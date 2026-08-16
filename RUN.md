# Environment

These notes target one x86_64 machine with 8 NVIDIA B200 GPUs. Prefer Docker
for B200. 

On the host:

```bash
docker pull verlai/verl:vllm017.latest

docker run --rm -it \
  --gpus all \
  --net=host \
  --ipc=host \
  --shm-size=512g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add=SYS_ADMIN \
  -v "$PWD:/workspace/verl" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/verl \
  --name verl-rft-rl \
  verlai/verl:vllm017.latest \
  bash
```

Inside the container:

```bash
pip install --no-deps -e .

python - <<'PY'
import torch
import pandas
import pyarrow
import megatron.core
import transformer_engine

print("cuda:", torch.version.cuda)
print("gpus:", torch.cuda.device_count())
PY
```

If the Docker image does not already contain the needed Megatron stack, install
the repo dependency set inside a CUDA/PyTorch base image:

```bash
USE_SGLANG=0 USE_MEGATRON=1 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

# RFT

RFT training data is `rft.json` in this format:

```text
list[trajectory]
trajectory = list[[token_ids, loss_flag]]
loss_flag = 0 for context-only tokens, 1 for tokens used in SFT loss
```

The launcher at `examples/sft/rft/rft.py` converts this JSON to parquet and then
runs verl SFT with the Megatron backend.

## Data Check

Place the collected trajectories at `rft.json`, or pass an absolute path with
`--rft-json`.

Run only the conversion step first:

```bash
python examples/sft/rft/rft.py \
  --rft-json rft.json \
  --train-parquet rft.parquet \
  --convert-only
```

This writes `rft.parquet` and skips trajectories with no supervised tokens.

## Dry Run

Print the exact `torchrun` command without launching training:

```bash
python examples/sft/rft/rft.py \
  --rft-json rft.json \
  --train-parquet rft.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/rft/qwen3-8b-megatron \
  --dry-run
```

Expected plan for the default 8-GPU setup:

```text
dp_size=8, local_batch_size=8, micro_batch_size_per_gpu=1, grad_accum_steps=8
```

That corresponds to global batch size 64:

## Training

Run the default single-node B200 plan:

```bash
python examples/sft/rft/rft.py \
  --rft-json rft.json \
  --train-parquet rft.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/rft/qwen3-8b-megatron \
  --nproc 8 \
  --tp 1 \
  --pp 1 \
  --cp 1 \
  --global-batch-size 64 \
  --micro-batch-size-per-gpu 1 \
  --max-length 60000 \
  --max-token-len-per-gpu 60000 \
  --epochs 1
```

For a second run after `rft.parquet` already exists, skip conversion:

```bash
python examples/sft/rft/rft.py \
  --no-convert \
  --train-parquet rft.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/rft/qwen3-8b-megatron
```

## Memory Knobs

The default command keeps one 60k-token trajectory resident per GPU and uses
Megatron-FSDP/distributed optimizer plus activation recompute.

If it OOMs, first try context parallelism:

```bash
python examples/sft/rft/rft.py \
  --no-convert \
  --train-parquet rft.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/rft/qwen3-8b-megatron-cp2 \
  --lr 1e-5 \
  --nproc 8 \
  --cp 2 \
  --global-batch-size 64 \
  --micro-batch-size-per-gpu 1
```

With `--cp 2`, data parallel size becomes 4, so batch 64 uses 16 accumulation
microbatches per data-parallel rank.

If the model path is local, point `--model-path` at the local HuggingFace model
directory to avoid network downloads.

# RL

The offline DAPO launcher is `examples/sft/rft/dapo.py`. It is intentionally
actor-only: no rollout worker, no reward model, no critic, no reference model,
and no KL. It consumes vLLM probabilities from `dapo.json` as the old policy
probabilities and does not recompute old logprobs from the actor.

Expected `dapo.json`:

```text
list[group]
group = list[trajectory]                         # one SWE task; size comes from the uploaded JSON
trajectory = list[[input_ids, output_pairs, r]]  # r is the same 0/1 reward for every step
output_pairs = list[[output_token_id, vllm_probability]]
```

An empty trajectory `[]` is a placeholder for a missing/failed sample. It
contributes reward 0 to trajectory-level normalization and contributes no
generated tokens to training.

By default, the converter detects the trajectory group size as
`max(len(group) for group in groups)` and verifies that every selected group has
that size. Pass a positive `--expected-trajs-per-group` only to enforce an
explicit override.

The converter builds one training sample per input-output step:

```text
input_ids = input_ids + output_token_ids
loss_mask = 0 on input tokens, 1 on output tokens
old_log_probs = log(vllm_probability) on output tokens
advantages = trajectory reward normalized once across the task's trajectories,
             then copied to every output token of that trajectory
```

It implements the DAPO requirements used here:

```text
token-level loss aggregation
drop zero-advantage samples before training
trajectory step penalty = (threshold - retained_num_steps) / (max_step - threshold) when max_step > threshold and retained_num_steps > threshold, else 0
setting max_step == threshold disables the length penalty
truncate trajectories after max_step=50; the same max_step is the penalty endpoint
discard i-o pairs with input length > 60000
discard i-o pairs with input + output length > 60000
single optimizer update over the whole converted batch
default clip deltas are 0.2 and 0.28, so ratio is clipped to [0.8, 1.28]
```

## RL Data Check

Convert only:

```bash
python examples/sft/rft/dapo.py \
  --dapo-json dapo.json \
  --train-parquet dapo.parquet \
  --num-groups 64 \
  --convert-only
```

Use `--num-groups 0` to consume every group in the JSON. The default assumes the
planned batch of 64 SWE tasks.

## RL Dry Run

```bash
python examples/sft/rft/dapo.py \
  --dapo-json dapo.json \
  --train-parquet dapo.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/dapo/qwen3-8b-megatron \
  --dry-run
```

For 8 GPUs with `tp=pp=cp=1`, the plan is:

```text
dp_size = 8
one i-o pair per GPU microbatch
microbatches_per_dp = converted_rows / 8
optimizer_updates = 1
```

The converter pads with zero-loss rows only when needed so the row count is
divisible by `dp_size * micro_batch_size_per_gpu`.

## RL Training

Run the single-node B200 plan:

```bash
python examples/sft/rft/dapo.py \
  --dapo-json dapo.json \
  --train-parquet dapo.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/dapo/qwen3-8b-megatron \
  --num-groups 64 \
  --nproc 8 \
  --tp 1 \
  --pp 1 \
  --cp 1 \
  --micro-batch-size-per-gpu 1 \
  --max-input-length 60000 \
  --max-total-length 60000 \
  --max-token-len-per-gpu 60000 \
  --max-steps 50 \
  --step-penalty-threshold 35 \
  --clip-ratio-low 0.2 \
  --clip-ratio-high 0.28 \
  --lr 1e-6 \
  --min-lr 1e-6
```

For a second run after `dapo.parquet` already exists:

```bash
python examples/sft/rft/dapo.py \
  --no-convert \
  --train-parquet dapo.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/dapo/qwen3-8b-megatron
```

## RL Memory Knobs

The most memory-efficient default is one input-output step per GPU microbatch:

```text
TP=1, PP=1, CP=1, DP=8, micro_batch_size_per_gpu=1
```

If 60k-token steps OOM, shard each long sequence over two GPUs:

```bash
python examples/sft/rft/dapo.py \
  --no-convert \
  --train-parquet dapo.parquet \
  --model-path Qwen/Qwen3-8B \
  --save-path checkpoints/dapo/qwen3-8b-megatron-cp2 \
  --nproc 8 \
  --cp 2 \
  --micro-batch-size-per-gpu 1
```

With `--cp 2`, data parallel size becomes 4, so each i-o pair spans a
context-parallel pair of GPUs and the number of microbatches per DP rank doubles.

## RL Notes

`output_probability_from_vllm` must be the probability of the sampled output
token at the matching output position. The script clamps probabilities below
`1e-12` before `log()`, rejects probabilities greater than 1, and never calls
actor logprob recomputation for the old policy.

The actor still computes current logprobs for `input_ids + output_ids`; those
current logprobs are required for gradients. Only the old probabilities are
forced to the collected vLLM values.
