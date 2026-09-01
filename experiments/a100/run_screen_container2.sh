#!/usr/bin/env bash

# Remaining Stage-1 objectives assigned to a second A100 container while the
# RTX PRO 6000 container finishes fixed_half.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
if [[ "${gpu_name}" != *A100* && "${ALLOW_NON_A100:-0}" != 1 ]]; then
    printf 'Expected an A100, but found: %s\n' "${gpu_name:-no visible GPU}" >&2
    printf 'Set ALLOW_NON_A100=1 only if this is intentional.\n' >&2
    exit 2
fi

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    python_bin="${REPO_ROOT}/.venv/bin/python"
else
    python_bin="$(command -v python3)"
fi

# The Blackwell environment intentionally omits flash-attn because FA2 does
# not support SM120. Install it lazily in the A100 container so actor training
# uses the faster, memory-efficient path that the historical A100 runs used.
if ! "${python_bin}" -c 'import flash_attn' >/dev/null 2>&1; then
    printf 'Installing flash-attn 2.8.3 for the A100 environment...\n'
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "${python_bin}" --no-build-isolation 'flash-attn==2.8.3'
    else
        "${python_bin}" -m pip install --no-build-isolation 'flash-attn==2.8.3'
    fi
fi

export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export ACTOR_TOKEN_BUDGET="${ACTOR_TOKEN_BUDGET:-8192}"
export ENTROPY_ACTOR_TOKEN_BUDGET="${ENTROPY_ACTOR_TOKEN_BUDGET:-6144}"
export ROLLOUT_GPU_FRACTION="${ROLLOUT_GPU_FRACTION:-0.30}"
# TorchMemorySaver cannot coexist with expandable_segments in the current
# dependency set used for colocated rollout/training.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
# A100 containers historically had working W&B connectivity. Override with
# WANDB_MODE=offline if the new container cannot reach api.wandb.ai.
export WANDB_MODE="${WANDB_MODE:-online}"
export METHODS="${METHODS:-rank_jsd rank_jsd_entropy}"

exec "${SCRIPT_DIR}/run_screen.sh"
