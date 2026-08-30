#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline one-update DAPO training on pre-collected SWE trajectories.

Expected JSON format:

list[group]
group = list[trajectory]
trajectory = list[[input_token_ids, [[output_token_id, vllm_probability], ...], reward]]

The script enforces these offline-DAPO semantics:

* old log probabilities come only from the vLLM probabilities in dapo.json;
* KL, critic, reference model, reward model, and rollout are not used;
* one input-output step is one training sample;
* zero-advantage samples are removed before training;
* token-level loss uses one global valid-token denominator;
* Megatron performs exactly one optimizer update over the converted batch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


def _as_list(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    return value


def _read_int_list(value: Any, field_name: str, index: int) -> list[int]:
    value = _as_list(value)
    if not isinstance(value, list):
        raise TypeError(f"row {index}: {field_name} must be a list, got {type(value).__name__}")
    return [int(item) for item in value]


def _read_float_list(value: Any, field_name: str, index: int) -> list[float]:
    value = _as_list(value)
    if not isinstance(value, list):
        raise TypeError(f"row {index}: {field_name} must be a list, got {type(value).__name__}")
    return [float(item) for item in value]


def _safe_log_probability(probability: Any, min_probability: float, index_label: str) -> float:
    probability = float(probability)
    if not math.isfinite(probability):
        raise ValueError(f"{index_label}: vLLM probability must be finite, got {probability}")
    if probability > 1.0:
        raise ValueError(
            f"{index_label}: expected probability in [0, 1], got {probability}. "
            "If the data stores logprobs, convert them to probabilities before using this script."
        )
    probability = max(probability, min_probability)
    return math.log(probability)


class DAPOStepDataset(Dataset):
    """Dataset consumed by ``OfflineDAPOTrainer`` through no-padding batches."""

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer=None,
        config: dict | None = None,
        processor=None,
        max_samples: int = -1,
    ):
        del tokenizer, processor
        config = config or {}
        self.pad_mode = config.get("pad_mode", "no_padding")
        self.truncation = config.get("truncation", "error")
        self.max_length = int(config.get("max_length", 60000))
        if self.pad_mode != "no_padding":
            raise ValueError("DAPOStepDataset is intended for data.pad_mode=no_padding")
        if self.truncation != "error":
            raise ValueError("DAPOStepDataset expects conversion-time filtering; use data.truncation=error")

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        dataframes = [pd.read_parquet(path, dtype_backend="pyarrow") for path in parquet_files]
        self.dataframe = pd.concat(dataframes, ignore_index=True)
        if max_samples > 0:
            self.dataframe = self.dataframe.iloc[:max_samples].reset_index(drop=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]
        input_ids = _read_int_list(row["input_ids"], "input_ids", index)
        loss_mask = _read_int_list(row["loss_mask"], "loss_mask", index)
        old_log_probs = _read_float_list(row["old_log_probs"], "old_log_probs", index)
        advantages = _read_float_list(row["advantages"], "advantages", index)

        lengths = {len(input_ids), len(loss_mask), len(old_log_probs), len(advantages)}
        if len(lengths) != 1:
            raise ValueError(f"row {index}: input_ids, loss_mask, old_log_probs, and advantages lengths differ")
        if not input_ids:
            raise ValueError(f"row {index}: empty sequence")
        if len(input_ids) > self.max_length:
            raise ValueError(f"row {index}: sequence length {len(input_ids)} exceeds max_length={self.max_length}")

        loss_mask = [int(mask) for mask in loss_mask]
        old_log_probs = [float(value) for value in old_log_probs]
        advantages = [float(value) for value in advantages]

        # No token before position 0 exists to predict it. The loss later rolls
        # this mask by one position to align next-token labels with logprobs.
        loss_mask[0] = 0
        old_log_probs[0] = 0.0
        advantages[0] = 0.0

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        position_ids = torch.arange(input_ids_tensor.numel(), dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "old_log_probs": torch.tensor(old_log_probs, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
        }


def _parse_output_pairs(output_pairs: Any, label: str, min_probability: float) -> tuple[list[int], list[float]]:
    output_pairs = _as_list(output_pairs)
    if not isinstance(output_pairs, list):
        raise TypeError(f"{label}: output must be a list of [token_id, probability] pairs")

    output_ids: list[int] = []
    old_log_probs: list[float] = []
    for output_index, pair in enumerate(output_pairs):
        pair = _as_list(pair)
        if not isinstance(pair, list) or len(pair) != 2:
            raise TypeError(f"{label}, output {output_index}: expected [token_id, probability]")
        token_id, probability = pair
        output_ids.append(int(token_id))
        old_log_probs.append(_safe_log_probability(probability, min_probability, f"{label}, output {output_index}"))

    return output_ids, old_log_probs


def _step_penalty(step_idx: int, threshold: int, max_step: int) -> float:
    if max_step < threshold:
        raise ValueError(f"--max-steps must be greater than or equal to --step-penalty-threshold ({max_step} < {threshold})")
    if step_idx <= threshold or max_step == threshold:
        return 0.0
    return float(threshold - step_idx) / float(max_step - threshold)


def _group_mean_std(values: list[float], ddof: int) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) <= ddof:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    return mean, math.sqrt(max(var, 0.0))


def _dummy_row_from(row: dict[str, Any]) -> dict[str, Any]:
    input_ids = list(row["input_ids"])
    return {
        "input_ids": input_ids,
        "loss_mask": [0] * len(input_ids),
        "old_log_probs": [0.0] * len(input_ids),
        "advantages": [0.0] * len(input_ids),
        "seq_len": len(input_ids),
        "response_tokens": 0,
        "group_idx": -1,
        "traj_idx": -1,
        "step_idx": -1,
        "raw_reward": 0.0,
        "effective_reward": 0.0,
        "advantage": 0.0,
        "is_padding": True,
    }


def convert_json_to_parquet(
    json_path: Path,
    parquet_path: Path,
    *,
    num_groups: int,
    expected_trajs_per_group: int,
    max_input_length: int,
    max_total_length: int,
    max_steps: int,
    step_penalty_threshold: int,
    min_probability: float,
    std_epsilon: float,
    std_ddof: int,
    zero_adv_epsilon: float,
    divisibility: int,
) -> dict[str, int | float]:
    if max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if step_penalty_threshold < 0:
        raise ValueError("--step-penalty-threshold must be non-negative")
    if max_steps < step_penalty_threshold:
        raise ValueError("--max-steps must be greater than or equal to --step-penalty-threshold")
    if expected_trajs_per_group < 0:
        raise ValueError("--expected-trajs-per-group must be non-negative")

    with json_path.open("r", encoding="utf-8") as f:
        groups = json.load(f)
    if not isinstance(groups, list):
        raise TypeError("dapo.json must contain a list of task groups")
    if num_groups > 0:
        if len(groups) < num_groups:
            raise ValueError(f"dapo.json has {len(groups)} groups, but --num-groups={num_groups}")
        groups = groups[:num_groups]

    normalized_groups: list[list[Any]] = []
    for group_idx, group in enumerate(groups):
        group = _as_list(group)
        if not isinstance(group, list):
            raise TypeError(f"group {group_idx}: expected list, got {type(group).__name__}")
        normalized_groups.append(group)
    groups = normalized_groups

    detected_group_size = max((len(group) for group in groups), default=0)
    group_size = expected_trajs_per_group or detected_group_size

    rows: list[dict[str, Any]] = []
    stats: dict[str, int | float] = {
        "groups": len(groups),
        "group_size": group_size,
        "groups_without_valid_steps": 0,
        "groups_with_zero_advantage": 0,
        "groups_with_nonstandard_traj_count": 0,
        "trajectories": 0,
        "trajectories_truncated": 0,
        "steps_seen": 0,
        "steps_after_max_step": 0,
        "discard_input_too_long": 0,
        "discard_total_too_long": 0,
        "discard_empty_output": 0,
        "discard_zero_advantage": 0,
        "padding_rows": 0,
    }

    for group_idx, group in enumerate(groups):
        if len(group) != group_size:
            stats["groups_with_nonstandard_traj_count"] += 1
            raise ValueError(
                f"group {group_idx}: expected {group_size} trajectories, got {len(group)}"
            )

        trajectories: list[dict[str, Any]] = []
        for traj_idx, trajectory in enumerate(group):
            trajectory = _as_list(trajectory)
            if not isinstance(trajectory, list):
                raise TypeError(f"group {group_idx}, trajectory {traj_idx}: expected list")
            stats["trajectories"] += 1
            stats["steps_seen"] += len(trajectory)
            if not trajectory:
                # The current JSON format repeats a trajectory's final reward
                # on every step. An empty trajectory therefore cannot carry a
                # reward explicitly. SWE-agent only emits an empty trajectory
                # when no model step ran, which is an unsuccessful rollout.
                trajectories.append(
                    {
                        "raw_reward": 0.0,
                        "effective_reward": 0.0,
                        "candidates": [],
                    }
                )
                continue

            capped_steps = min(len(trajectory), max_steps)
            if len(trajectory) > max_steps:
                stats["trajectories_truncated"] += 1
                stats["steps_after_max_step"] += len(trajectory) - max_steps

            raw_reward: float | None = None
            traj_candidates: list[dict[str, Any]] = []
            for step_idx, step in enumerate(trajectory[:max_steps], start=1):
                step = _as_list(step)
                if not isinstance(step, list) or len(step) != 3:
                    raise TypeError(
                        f"group {group_idx}, trajectory {traj_idx}, step {step_idx}: "
                        "expected [input_token_ids, output_token_probability_pairs, reward]"
                    )

                input_ids_raw, output_pairs, reward = step
                reward = float(reward)
                if raw_reward is None:
                    raw_reward = reward
                elif abs(reward - raw_reward) > 1e-8:
                    raise ValueError(
                        f"group {group_idx}, trajectory {traj_idx}: reward differs across steps "
                        f"({raw_reward} vs {reward})"
                    )

                input_ids_raw = _as_list(input_ids_raw)
                if not isinstance(input_ids_raw, list):
                    raise TypeError(f"group {group_idx}, trajectory {traj_idx}, step {step_idx}: input must be a list")
                input_ids = [int(token_id) for token_id in input_ids_raw]
                if len(input_ids) > max_input_length:
                    stats["discard_input_too_long"] += 1
                    continue

                output_ids, output_old_log_probs = _parse_output_pairs(
                    output_pairs,
                    f"group {group_idx}, trajectory {traj_idx}, step {step_idx}",
                    min_probability,
                )
                if not output_ids:
                    stats["discard_empty_output"] += 1
                    continue

                total_length = len(input_ids) + len(output_ids)
                if max_total_length > 0 and total_length > max_total_length:
                    stats["discard_total_too_long"] += 1
                    continue

                traj_candidates.append(
                    {
                        "input_prompt_ids": input_ids,
                        "output_ids": output_ids,
                        "output_old_log_probs": output_old_log_probs,
                        "group_idx": group_idx,
                        "traj_idx": traj_idx,
                        "step_idx": step_idx,
                    }
                )

            if raw_reward is None:
                continue

            effective_reward = raw_reward + _step_penalty(
                capped_steps,
                threshold=step_penalty_threshold,
                max_step=max_steps,
            )
            trajectories.append(
                {
                    "raw_reward": raw_reward,
                    "effective_reward": effective_reward,
                    "candidates": traj_candidates,
                }
            )

        if not trajectories:
            stats["groups_without_valid_steps"] += 1
            continue

        # GRPO normalization is over trajectories, not input-output steps.
        # Each trajectory contributes its final reward exactly once regardless
        # of how many retained conversation steps it contains.
        rewards = [float(trajectory["effective_reward"]) for trajectory in trajectories]
        group_mean, group_std = _group_mean_std(rewards, ddof=std_ddof)

        if group_std <= std_epsilon:
            stats["groups_with_zero_advantage"] += 1
            stats["discard_zero_advantage"] += sum(
                len(trajectory["candidates"]) for trajectory in trajectories
            )
            continue

        if not any(trajectory["candidates"] for trajectory in trajectories):
            stats["groups_without_valid_steps"] += 1
            continue

        for trajectory in trajectories:
            advantage = (float(trajectory["effective_reward"]) - group_mean) / (group_std + std_epsilon)
            if abs(advantage) <= zero_adv_epsilon:
                stats["discard_zero_advantage"] += len(trajectory["candidates"])
                continue

            for candidate in trajectory["candidates"]:
                prompt_ids = candidate["input_prompt_ids"]
                output_ids = candidate["output_ids"]
                input_ids = prompt_ids + output_ids
                response_tokens = len(output_ids)
                loss_mask = [0] * len(prompt_ids) + [1] * response_tokens
                old_log_probs = [0.0] * len(prompt_ids) + candidate["output_old_log_probs"]
                advantages = [0.0] * len(prompt_ids) + [float(advantage)] * response_tokens
                loss_mask[0] = 0
                old_log_probs[0] = 0.0
                advantages[0] = 0.0

                rows.append(
                    {
                        "input_ids": input_ids,
                        "loss_mask": loss_mask,
                        "old_log_probs": old_log_probs,
                        "advantages": advantages,
                        "seq_len": len(input_ids),
                        "response_tokens": int(sum(loss_mask)),
                        "group_idx": int(candidate["group_idx"]),
                        "traj_idx": int(candidate["traj_idx"]),
                        "step_idx": int(candidate["step_idx"]),
                        "raw_reward": float(trajectory["raw_reward"]),
                        "effective_reward": float(trajectory["effective_reward"]),
                        "advantage": float(advantage),
                        "is_padding": False,
                    }
                )

    if not rows:
        raise ValueError("no nonzero-advantage training samples found after conversion")

    if divisibility <= 0:
        raise ValueError("divisibility must be positive")
    remainder = len(rows) % divisibility
    if remainder:
        padding_needed = divisibility - remainder
        template = rows[0]
        rows.extend(_dummy_row_from(template) for _ in range(padding_needed))
        stats["padding_rows"] = padding_needed

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)

    real_rows = sum(1 for row in rows if not row["is_padding"])
    max_seq_len = max(row["seq_len"] for row in rows)
    total_response_tokens = sum(row["response_tokens"] for row in rows)
    stats["rows"] = len(rows)
    stats["real_rows"] = real_rows
    stats["max_seq_len"] = max_seq_len
    stats["total_response_tokens"] = total_response_tokens

    print(
        f"wrote {len(rows)} DAPO rows to {parquet_path} "
        f"(real_rows={real_rows}, padding_rows={stats['padding_rows']}, "
        f"total_response_tokens={total_response_tokens}, max_seq_len={max_seq_len})",
        flush=True,
    )
    return stats


def dapo_policy_loss(
    *,
    clip_ratio_low: float,
    clip_ratio_high: float,
    model_output,
    data,
    dp_group=None,
):
    del dp_group
    log_prob = model_output["log_probs"]
    if not log_prob.is_nested:
        raise ValueError("dapo_policy_loss expects no-padding nested log_probs")

    log_prob_flat = log_prob.values()
    loss_mask_flat = torch.roll(data["loss_mask"].values().to(torch.bool), shifts=-1, dims=0)
    old_log_prob_flat = torch.roll(data["old_log_probs"].values().to(log_prob_flat.dtype), shifts=-1, dims=0)
    advantages_flat = torch.roll(data["advantages"].values().to(log_prob_flat.dtype), shifts=-1, dims=0)

    if loss_mask_flat.sum() == 0:
        zero = log_prob_flat.sum() * 0.0
        return zero, {"dapo/nonzero_tokens": 0.0}

    log_ratio = torch.clamp(log_prob_flat - old_log_prob_flat, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)

    pg_losses1 = -advantages_flat * ratio
    pg_losses2 = -advantages_flat * torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    batch_num_tokens = data["batch_num_tokens"]
    dp_size = data["dp_size"]
    loss = torch.masked_select(pg_losses, loss_mask_flat).sum() / batch_num_tokens * dp_size

    with torch.no_grad():
        clipped = (pg_losses2 > pg_losses1).to(torch.float32)
        mask_float = loss_mask_flat.to(torch.float32)
        denom = mask_float.sum().clamp_min(1.0)
        metrics = {
            "dapo/clipfrac": ((clipped * mask_float).sum() / denom).detach().item(),
            "dapo/approx_kl": ((-log_ratio * mask_float).sum() / denom).detach().item(),
            "dapo/ratio_mean": ((ratio * mask_float).sum() / denom).detach().item(),
            "dapo/nonzero_tokens": mask_float.sum().detach().item(),
        }
    return loss, metrics


class OfflineDAPOTrainer:
    def __init__(self, config):
        from verl.utils.profiler import log_gpu_memory_usage

        self.config = config
        log_gpu_memory_usage(f"rank {torch.distributed.get_rank()}: Before OfflineDAPOTrainer init")
        self.rank = torch.distributed.get_rank()
        self._build_config()
        self._build_dataset()
        self._build_engine() # The loss function is registered here!
        self._build_dataloader() # Here DistributedSampler API splits 1/nproc data points evenly to the current proc
        self._init_engine()
        self._build_ckpt_handler()
        if self.rank == 0:
            print(self.config)

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

    def _build_dataset(self):
        self.train_dataset = DAPOStepDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.model_config.tokenizer,
            config=self.config.data,
            processor=self.model_config.processor,
            max_samples=self.config.data.get("train_max_samples", -1),
        )

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorker
        from verl.workers.config import TrainingWorkerConfig

        loss_fn = partial(
            dapo_policy_loss,
            clip_ratio_low=float(self.config.dapo.clip_ratio_low),
            clip_ratio_high=float(self.config.dapo.clip_ratio_high),
        )

        worker_config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        self.training_client = TrainingWorker(config=worker_config)
        self.training_client.set_loss_fn(loss_fn=loss_fn)
        self.engine = self.training_client.engine

    def _build_dataloader(self):
        from verl.utils.dataset.dataset_utils import SFTTensorCollator
        from verl.utils.device import get_device_name

        dp_rank = self.engine.get_data_parallel_rank()
        dp_size = self.engine.get_data_parallel_size()
        if len(self.train_dataset) % dp_size != 0:
            raise ValueError(f"dataset rows ({len(self.train_dataset)}) must be divisible by dp_size ({dp_size})")

        self.global_batch_size = len(self.train_dataset)
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        if self.train_batch_size_per_dp <= 0:
            raise ValueError("per-DP batch is empty; add more nonzero-advantage samples")

        micro_batch_size = int(self.config.data.micro_batch_size_per_gpu)
        if self.train_batch_size_per_dp % micro_batch_size != 0:
            raise ValueError(
                f"per-DP batch size {self.train_batch_size_per_dp} must be divisible by "
                f"micro_batch_size_per_gpu={micro_batch_size}"
            )

        sampler = DistributedSampler(
            self.train_dataset,
            shuffle=False,
            num_replicas=dp_size,
            rank=dp_rank,
            drop_last=False,
        )
        self.collate_fn = SFTTensorCollator(self.config.data.pad_mode)
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=False,
            pin_memory_device=get_device_name(),
        )

    def _init_engine(self):
        self.total_training_steps = 1
        self.optimizer_config.total_training_steps = 1
        self.training_client.reset()

    def _build_ckpt_handler(self):
        from verl.utils.checkpoint import CheckpointHandler

        self.ckpt_handler = CheckpointHandler(
            engine=self.engine,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=self.config.trainer.max_ckpt_to_keep,
            default_hdfs_dir=self.config.trainer.default_hdfs_dir,
            resume_mode=self.config.trainer.resume_mode,
            resume_from_path=self.config.trainer.resume_from_path,
            lora_train_meta=None,
        )
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

    def _get_batch_seqlens(self, data):
        if data["input_ids"].is_nested:
            batch_seqlens = data["input_ids"].offsets().diff()
        else:
            batch_seqlens = data["attention_mask"].sum(dim=-1)
        batch_seqlens = batch_seqlens.to(self.config.trainer.device)

        dp_group = self.engine.get_data_parallel_group()
        dp_size = self.engine.get_data_parallel_size()
        if dp_size == 1 or dp_group is None:
            return batch_seqlens.tolist()

        output_tensor = torch.empty(
            (batch_seqlens.shape[0] * dp_size,),
            dtype=batch_seqlens.dtype,
            device=self.config.trainer.device,
        )
        torch.distributed.all_gather_into_tensor(output_tensor=output_tensor, input_tensor=batch_seqlens, group=dp_group)
        return output_tensor.tolist()

    def fit(self):
        """Run the single DAPO update for this worker process."""

        from tensordict.tensorclass import NonTensorData

        from verl.utils import tensordict_utils as tu
        from verl.utils.logger import log_with_rank
        from verl.utils.memory_utils import aggressive_empty_cache
        from verl.utils.metric.utils import reduce_metrics
        from verl.utils.tracking import Tracking

        # Only one rank should create experiment logs. All ranks still execute
        # the same data-loading and training path below.
        is_logging = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0
        tracking = None
        if is_logging:
            from omegaconf import OmegaConf

            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        # The script is designed for exactly one optimizer update. If a resumed
        # checkpoint already reached step 1, skip instead of replaying the batch.
        if self.resume_global_step >= 1:
            log_with_rank("Checkpoint already has global_step >= 1; skipping one-update DAPO run.", rank=0)
            return

        # Metadata consumed by verl/Megatron for no-padding batches,
        # microbatch splitting, and token-level loss normalization.
        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        aggressive_empty_cache(force_sync=True)

        # Each DP rank receives one dataloader batch: its shard of all converted
        # offline DAPO step samples. A second batch would imply a second update.
        data_iter = iter(self.train_dataloader)
        data = next(data_iter)
        try:
            next(data_iter)
            raise RuntimeError("OfflineDAPOTrainer expected exactly one dataloader batch")
        except StopIteration:
            pass

        data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
        batch_seqlens = self._get_batch_seqlens(data)

        # Attach per-step controls for TrainingWorker. global_token_num is
        # gathered across DP ranks and used by dapo_policy_loss's denominator.
        tu.assign_non_tensor(
            data,
            update_lr_scheduler=True,
            global_token_num=NonTensorData(batch_seqlens),
            disable_auto_offload=True,
        )

        # This single call performs zero-grad, microbatch forward/backward
        # accumulation, DP gradient sync, optimizer step, and LR scheduler step.
        output = self.training_client.train_batch(data=data)

        # Reduce metrics across ranks, rename common training metrics, and log a
        # compact summary from the output/source rank.
        if self.engine.is_mp_src_rank_with_outputs():
            metrics = tu.get(output, "metrics")
            metrics = reduce_metrics(metrics)
            renamed = {}
            for key, value in metrics.items():
                if key in {"loss", "grad_norm", "lr", "mfu"}:
                    renamed[f"train/{key}"] = value
                else:
                    renamed[key] = value
            renamed["train/global_sequences"] = self.global_batch_size
            renamed["train/global_tokens"] = sum(batch_seqlens)
            if tracking is not None:
                tracking.log(data=renamed, step=1)
            if self.engine.get_data_parallel_rank() == 0:
                print("DAPO metrics:", renamed, flush=True)

        aggressive_empty_cache(force_sync=True)

        # Save the post-update model at global step 1.
        self.ckpt_handler.save_checkpoint(step=1)


def run_dapo(config):
    from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group

    initialize_global_process_group()
    trainer = OfflineDAPOTrainer(config=config)
    trainer.fit()
    destroy_global_process_group()


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dapo-json", type=Path, default=Path("dapo.json"))
    parser.add_argument("--train-parquet", type=Path, default=Path("dapo.parquet"))
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--save-path", default="checkpoints/dapo/qwen3-8b-megatron")
    parser.add_argument("--num-groups", type=int, default=64, help="Number of SWE task groups to train on; 0 means all.")
    parser.add_argument(
        "--expected-trajs-per-group",
        type=int,
        default=0,
        help=(
            "Expected trajectories in every task group. Default: 0, which detects "
            "max(len(group)) from the selected dapo.json groups."
        ),
    )
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", default="29600")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-input-length", type=int, default=60000)
    parser.add_argument("--max-total-length", type=int, default=60000)
    parser.add_argument("--max-token-len-per-gpu", type=int, default=60000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--step-penalty-threshold", type=int, default=35)
    parser.add_argument(
        "--clip-ratio-low",
        type=float,
        default=0.2,
        help="Lower PPO/DAPO clip ratio delta; ratio lower bound is 1 - this value.",
    )
    parser.add_argument(
        "--clip-ratio-high",
        type=float,
        default=0.28,
        help="Upper PPO/DAPO clip ratio delta; ratio upper bound is 1 + this value.",
    )
    parser.add_argument("--lr", default="1e-6")
    parser.add_argument("--min-lr", default="1e-6")
    parser.add_argument("--weight-decay", default="0.1")
    parser.add_argument("--std-epsilon", type=float, default=1e-6)
    parser.add_argument("--std-ddof", type=int, choices=[0, 1], default=0)
    parser.add_argument("--zero-adv-epsilon", type=float, default=0.0)
    parser.add_argument("--min-probability", type=float, default=1e-12)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--project-name", default="offline-dapo")
    parser.add_argument("--experiment-name", default="qwen3-8b-swe-dapo-megatron")
    parser.add_argument("--resume-mode", choices=["auto", "disable", "resume_path"], default="disable")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _dp_size(args: argparse.Namespace) -> int:
    model_parallel = args.tp * args.pp * args.cp
    if args.nproc % model_parallel != 0:
        raise ValueError("nproc must be divisible by tp * pp * cp")
    return args.nproc // model_parallel


def _torchrun_command(args: argparse.Namespace, global_batch_size: int) -> list[str]:
    script_path = Path(__file__).resolve()
    return [
        "torchrun",
        "--nnodes",
        str(args.nnodes),
        "--nproc_per_node",
        str(args.nproc),
        "--node_rank",
        str(args.node_rank),
        "--master_addr",
        args.master_addr,
        "--master_port",
        args.master_port,
        str(script_path),
        f"data.train_files={args.train_parquet.resolve()}",
        "data.val_files=null",
        f"data.train_batch_size={global_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"data.max_token_len_per_gpu={args.max_token_len_per_gpu}",
        "data.use_dynamic_bsz=False",
        "data.pad_mode=no_padding",
        f"data.max_length={args.max_total_length}",
        "data.truncation=error",
        f"data.num_workers={args.num_workers}",
        "model=hf_model",
        f"model.path={args.model_path}",
        "model.trust_remote_code=True",
        # Qwen3.5 GDN layers do not support Megatron's packed THD format.
        # False selects the padded BSHD path while data.use_dynamic_bsz remains disabled.
        "model.use_remove_padding=False",
        "optim=megatron",
        f"optim.lr={args.lr}",
        f"optim.min_lr={args.min_lr}",
        f"optim.weight_decay={args.weight_decay}",
        "optim.betas=[0.9,0.95]",
        "optim.clip_grad=1.0",
        "optim.lr_warmup_steps=0",
        "optim.lr_warmup_steps_ratio=0.0",
        "optim.lr_decay_style=constant",
        "engine=megatron",
        f"engine.tensor_model_parallel_size={args.tp}",
        f"engine.pipeline_model_parallel_size={args.pp}",
        f"engine.context_parallel_size={args.cp}",
        "engine.use_distributed_optimizer=True",
        "engine.use_megatron_fsdp=True",
        "engine.use_mbridge=True",
        "engine.vanilla_mbridge=False",
        "engine.override_transformer_config.recompute_granularity=full",
        "engine.override_transformer_config.recompute_method=uniform",
        "engine.override_transformer_config.recompute_num_layers=1",
        f"trainer.default_local_dir={args.save_path}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        "trainer.logger=[console]",
        "trainer.total_epochs=1",
        "trainer.total_training_steps=1",
        "trainer.test_freq=-1",
        "trainer.save_freq=1",
        f"trainer.resume_mode={args.resume_mode}",
        "+dapo.clip_ratio_low=" + str(args.clip_ratio_low),
        "+dapo.clip_ratio_high=" + str(args.clip_ratio_high),
    ]


def launcher_main() -> int:
    args = _arg_parser().parse_args()
    if args.clip_ratio_low < 0 or args.clip_ratio_high < 0:
        raise ValueError("--clip-ratio-low and --clip-ratio-high must be non-negative")
    dp_size = _dp_size(args)
    divisibility = dp_size * args.micro_batch_size_per_gpu

    stats: dict[str, int | float] | None = None
    if not args.no_convert:
        stats = convert_json_to_parquet(
            args.dapo_json,
            args.train_parquet,
            num_groups=args.num_groups,
            expected_trajs_per_group=args.expected_trajs_per_group,
            max_input_length=args.max_input_length,
            max_total_length=args.max_total_length,
            max_steps=args.max_steps,
            step_penalty_threshold=args.step_penalty_threshold,
            min_probability=args.min_probability,
            std_epsilon=args.std_epsilon,
            std_ddof=args.std_ddof,
            zero_adv_epsilon=args.zero_adv_epsilon,
            divisibility=divisibility,
        )

    if args.convert_only:
        return 0

    if stats is None:
        row_count = len(pd.read_parquet(args.train_parquet))
    else:
        row_count = int(stats["rows"])

    if row_count % dp_size != 0:
        raise ValueError(f"parquet row count {row_count} must be divisible by dp_size={dp_size}")
    per_dp_batch_size = row_count // dp_size
    if per_dp_batch_size % args.micro_batch_size_per_gpu != 0:
        raise ValueError(
            f"per-DP batch size {per_dp_batch_size} must be divisible by "
            f"micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}"
        )
    grad_accum_microbatches = per_dp_batch_size // args.micro_batch_size_per_gpu

    cmd = _torchrun_command(args, global_batch_size=row_count)
    print(
        f"Offline DAPO plan: dp_size={dp_size}, global_step_samples={row_count}, "
        f"per_dp_batch_size={per_dp_batch_size}, "
        f"micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}, "
        f"microbatches_per_dp={grad_accum_microbatches}, optimizer_updates=1",
        flush=True,
    )
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    env = os.environ.copy()
    env["VERL_DAPO_WORKER"] = "1"
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("VERL_LOGGING_LEVEL", "INFO")

    print("Launching offline DAPO:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    return 0


def worker_main() -> int:
    import hydra

    from verl.utils.device import auto_set_device

    @hydra.main(config_path="../../../verl/trainer/config", config_name="sft_trainer_engine", version_base=None)
    def hydra_main(config):
        auto_set_device(config)
        run_dapo(config)

    hydra_main()
    return 0


def main() -> int:
    if os.environ.get("VERL_DAPO_WORKER") == "1":
        return worker_main()
    return launcher_main()


if __name__ == "__main__":
    sys.exit(main())
