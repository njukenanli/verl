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
"""Launch Megatron SFT on pre-tokenized RFT trajectories.

Expected JSON format:

[
  [
    [[token_id, ...], 0],
    [[token_id, ...], 1]
  ],
  ...
]

The second item is a loss mask flag: 0 means context-only, 1 means update on
those tokens. The complete trajectory is fed once; only the masked tokens
contribute to SFT loss.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


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


class RFTTrajectoryDataset(Dataset):
    """Dataset consumed by verl SFT through ``data.custom_cls``."""

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
            raise ValueError("RFTTrajectoryDataset is intended for data.pad_mode=no_padding")
        if self.truncation not in {"error", "left", "right"}:
            raise ValueError(f"Unknown truncation mode: {self.truncation}")

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

        if len(input_ids) != len(loss_mask):
            raise ValueError(f"row {index}: input_ids and loss_mask lengths differ")
        if not input_ids:
            raise ValueError(f"row {index}: empty trajectory")

        if len(input_ids) > self.max_length:
            if self.truncation == "error":
                raise ValueError(f"row {index}: sequence length {len(input_ids)} exceeds max_length={self.max_length}")
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            else:
                input_ids = input_ids[: self.max_length]
                loss_mask = loss_mask[: self.max_length]

        # No token before position 0 exists to predict it. This also prevents
        # no-padding packed batches from leaking a first-token mask across rows.
        loss_mask[0] = 0

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        loss_mask_tensor = torch.tensor(loss_mask, dtype=torch.bool)
        position_ids = torch.arange(input_ids_tensor.numel(), dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "position_ids": position_ids,
            "loss_mask": loss_mask_tensor,
        }


def _flatten_trajectory(trajectory: Any, index: int) -> tuple[list[int], list[int]]:
    trajectory = _as_list(trajectory)
    if not isinstance(trajectory, list):
        raise TypeError(f"trajectory {index}: expected list, got {type(trajectory).__name__}")

    input_ids: list[int] = []
    loss_mask: list[int] = []
    for segment_index, segment in enumerate(trajectory):
        segment = _as_list(segment)
        if not isinstance(segment, (list, tuple)) or len(segment) != 2:
            raise TypeError(f"trajectory {index}, segment {segment_index}: expected [token_ids, mask_flag]")

        token_ids, mask_flag = segment
        token_ids = _as_list(token_ids)
        if not isinstance(token_ids, list):
            raise TypeError(f"trajectory {index}, segment {segment_index}: token_ids must be a list")

        mask_flag = int(mask_flag)
        if mask_flag not in {0, 1}:
            raise ValueError(f"trajectory {index}, segment {segment_index}: mask flag must be 0 or 1")

        tokens = [int(token_id) for token_id in token_ids]
        input_ids.extend(tokens)
        loss_mask.extend([mask_flag] * len(tokens))

    if not input_ids:
        raise ValueError(f"trajectory {index}: empty trajectory")

    loss_mask[0] = 0
    return input_ids, loss_mask


def convert_json_to_parquet(json_path: Path, parquet_path: Path, max_length: int, truncation: str) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        trajectories = json.load(f)
    if not isinstance(trajectories, list):
        raise TypeError("rft.json must contain a list of trajectories")

    rows = []
    skipped = 0
    for index, trajectory in enumerate(trajectories):
        input_ids, loss_mask = _flatten_trajectory(trajectory, index)
        if len(input_ids) > max_length:
            if truncation == "error":
                raise ValueError(
                    f"trajectory {index}: sequence length {len(input_ids)} exceeds max_length={max_length}"
                )
            if truncation == "left":
                input_ids = input_ids[-max_length:]
                loss_mask = loss_mask[-max_length:]
            else:
                input_ids = input_ids[:max_length]
                loss_mask = loss_mask[:max_length]
            loss_mask[0] = 0

        loss_tokens = int(sum(loss_mask))
        if loss_tokens == 0:
            skipped += 1
            continue

        rows.append(
            {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "seq_len": len(input_ids),
                "loss_tokens": loss_tokens,
            }
        )

    if not rows:
        raise ValueError("no usable trajectories found after conversion")

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(parquet_path, index=False)
    max_seq_len = max(row["seq_len"] for row in rows)
    print(
        f"wrote {len(rows)} trajectories to {parquet_path} "
        f"(skipped_zero_loss={skipped}, max_seq_len={max_seq_len})",
        flush=True,
    )


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rft-json", type=Path, default=Path("rft.json"))
    parser.add_argument("--train-parquet", type=Path, default=Path("rft.parquet"))
    parser.add_argument("--model-path", default="Qwen/Qwen3-8B")
    parser.add_argument("--save-path", default="checkpoints/rft/qwen3-8b-megatron")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="localhost")
    parser.add_argument("--master-port", default="29500")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=60000)
    parser.add_argument("--max-token-len-per-gpu", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", default="1e-5")
    parser.add_argument("--min-lr", default="1e-6")
    parser.add_argument("--weight-decay", default="0.1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--project-name", default="rft-sft")
    parser.add_argument("--experiment-name", default="qwen3-8b-rft-megatron")
    parser.add_argument("--truncation", choices=["error", "left", "right"], default="error")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--no-convert", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _torchrun_command(args: argparse.Namespace) -> list[str]:
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
        "-m",
        "verl.trainer.sft_trainer",
        f"data.train_files={args.train_parquet.resolve()}",
        "data.val_files=null",
        f"data.train_batch_size={args.global_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}",
        f"data.max_token_len_per_gpu={args.max_token_len_per_gpu}",
        "data.use_dynamic_bsz=False",
        "data.pad_mode=no_padding",
        f"data.max_length={args.max_length}",
        f"data.truncation={args.truncation}",
        f"data.num_workers={args.num_workers}",
        f"data.custom_cls.path=file://{script_path}",
        "data.custom_cls.name=RFTTrajectoryDataset",
        "model=hf_model",
        f"model.path={args.model_path}",
        "model.trust_remote_code=True",
        "model.use_remove_padding=True",
        "optim=megatron",
        f"optim.lr={args.lr}",
        f"optim.min_lr={args.min_lr}",
        f"optim.weight_decay={args.weight_decay}",
        "optim.betas=[0.9,0.95]",
        "optim.clip_grad=1.0",
        "optim.lr_warmup_init=0",
        "optim.lr_warmup_steps_ratio=0.03",
        "optim.lr_decay_style=cosine",
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
        f"trainer.total_epochs={args.epochs}",
        "trainer.test_freq=-1",
        "trainer.save_freq=after_each_epoch",
    ]


def main() -> int:
    args = _arg_parser().parse_args()
    dp_size = args.nproc // (args.tp * args.pp * args.cp)
    if args.nproc % (args.tp * args.pp * args.cp) != 0:
        raise ValueError("nproc must be divisible by tp * pp * cp")
    if args.global_batch_size % dp_size != 0:
        raise ValueError("global batch size must be divisible by data-parallel size")
    local_batch_size = args.global_batch_size // dp_size
    if local_batch_size % args.micro_batch_size_per_gpu != 0:
        raise ValueError("per-DP batch size must be divisible by micro_batch_size_per_gpu")
    grad_accum_steps = local_batch_size // args.micro_batch_size_per_gpu

    if not args.no_convert:
        convert_json_to_parquet(args.rft_json, args.train_parquet, args.max_length, args.truncation)

    if args.convert_only:
        return 0

    cmd = _torchrun_command(args)
    print(
        f"RFT SFT plan: dp_size={dp_size}, local_batch_size={local_batch_size}, "
        f"micro_batch_size_per_gpu={args.micro_batch_size_per_gpu}, grad_accum_steps={grad_accum_steps}",
        flush=True,
    )
    if args.dry_run:
        print(" ".join(cmd))
        return 0

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("VERL_SFT_LOGGING_LEVEL", "INFO")

    print("Launching RFT SFT:", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
