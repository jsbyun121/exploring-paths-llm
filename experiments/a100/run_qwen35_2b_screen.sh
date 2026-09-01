#!/usr/bin/env bash

# Repeat the Stage-1 mechanism screen with Qwen3.5-2B in thinking mode.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
export ENABLE_THINKING="${ENABLE_THINKING:-true}"
# Qwen3.5 uses a hybrid Gated DeltaNet/attention architecture that is not
# supported by Liger's AutoModel patches.
export USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
# SDPA avoids coupling actor training to FlashAttention-2's PyTorch ABI while
# SGLang continues to use its model-specific rollout kernels.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-extra_buffer}"
export RELEASE_MEMORY_FOR_TRAINING="${RELEASE_MEMORY_FOR_TRAINING:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
# SGLang's TorchMemorySaver (used for colocated rollout/actor weights) cannot
# run with PyTorch's expandable-segments allocator.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# The longest thinking-mode GSM8K prompt is 242 tokens. A clipped 4,096-token
# response therefore yields 4,337 training tokens after the one-token shift.
# Keep just enough room for that singleton sequence while retaining the
# original rollout cap and a conservative full-vocabulary token pack.
export ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-4352}"
export ENTROPY_ACTOR_TOKEN_BUDGET="${ENTROPY_ACTOR_TOKEN_BUDGET:-4352}"
export ROLLOUT_GPU_FRACTION="${ROLLOUT_GPU_FRACTION:-0.20}"

exec "${SCRIPT_DIR}/run_screen.sh"
