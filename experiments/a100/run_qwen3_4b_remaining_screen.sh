#!/usr/bin/env bash

# Complete the Stage-1 Qwen3-4B-Thinking screen after the historical runs.
# Missing: fixed_half and rank_jsd_entropy. The historical rank_jsd run ended
# at W&B step 133 without a portable checkpoint, so rerun it through step 150.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
# This checkpoint supports only thinking mode and enables it in its default
# chat template, so no enable_thinking override is necessary.
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-auto}"
export RELEASE_MEMORY_FOR_TRAINING="${RELEASE_MEMORY_FOR_TRAINING:-true}"
export WANDB_MODE="${WANDB_MODE:-online}"

# SGLang's colocated memory saver is incompatible with expandable segments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"

# Match the original 80 GiB A100 screen configuration.
export ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-8192}"
export ENTROPY_ACTOR_TOKEN_BUDGET="${ENTROPY_ACTOR_TOKEN_BUDGET:-6144}"
export ROLLOUT_GPU_FRACTION="${ROLLOUT_GPU_FRACTION:-0.30}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
export MAX_STEPS="${MAX_STEPS:-150}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:-100}"

export METHODS="${METHODS:-fixed_half rank_jsd rank_jsd_entropy}"

exec "${SCRIPT_DIR}/run_screen.sh"
