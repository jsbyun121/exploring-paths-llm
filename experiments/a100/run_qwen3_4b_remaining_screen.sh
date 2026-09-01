#!/usr/bin/env bash

# Complete the Stage-1 Qwen3-4B-Thinking screen after the historical runs.
# Missing: fixed_half and rank_jsd_entropy. The historical rank_jsd run ended
# at W&B step 133 without a portable checkpoint, so rerun it through step 150.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# FlashInfer JIT-compiles SM120 kernels and must see a CUDA >= 12.9 compiler.
# The host image still exposes CUDA 12.8 at /usr/local/cuda, while the project
# environment installs the CUDA 13 compiler alongside PyTorch.
for cuda_home in "${REPO_ROOT}"/.venv/lib/python*/site-packages/nvidia/cu13; do
    if [[ -x "${cuda_home}/bin/nvcc" ]]; then
        export CUDA_HOME="${cuda_home}"
        # NVIDIA's pip toolkit stores libraries in lib/, while FlashInfer's
        # generated Ninja files use the conventional CUDA lib64/ path.
        if [[ ! -e "${CUDA_HOME}/lib64" ]]; then
            ln -s lib "${CUDA_HOME}/lib64"
        fi
        if [[ ! -e "${CUDA_HOME}/lib/libcudart.so" ]]; then
            ln -s libcudart.so.13 "${CUDA_HOME}/lib/libcudart.so"
        fi
        export PATH="${CUDA_HOME}/bin:${PATH}"
        export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export LIBRARY_PATH="${CUDA_HOME}/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
        break
    fi
done

export MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
# This checkpoint supports only thinking mode and enables it in its default
# chat template, so no enable_thinking override is necessary.
export ENABLE_THINKING="${ENABLE_THINKING:-false}"
export USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-true}"
# PyTorch SDPA provides a fused Blackwell training path without FA2's compiled
# extension or FA4/CuTe-DSL's current SM120 backward incompatibilities. SGLang
# rollout still uses its own FlashInfer attention and CUDA graphs.
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export MAMBA_SCHEDULER_STRATEGY="${MAMBA_SCHEDULER_STRATEGY:-auto}"
export RELEASE_MEMORY_FOR_TRAINING="${RELEASE_MEMORY_FOR_TRAINING:-true}"
# CuTe-DSL 4.5.1 has an MLIR binding crash in FlashInfer's SM120 RMSNorm.
# Keep FlashInfer attention/CUDA graphs, but use its CUDA RMSNorm fallback.
export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
# Do not let transient W&B/API connectivity terminate a multi-hour GPU run.
# Offline runs retain the complete history and can be uploaded with wandb sync.
export WANDB_MODE="${WANDB_MODE:-offline}"

# SGLang's colocated memory saver is incompatible with expandable segments.
# Drop the deprecated spelling as well, since it may be inherited from an
# older shell or launcher and takes effect before the memory saver starts.
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"

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
