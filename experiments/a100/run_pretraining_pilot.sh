#!/usr/bin/env bash

# Section 4.4 pilot. Run only after the post-training rank-JSD gate passes.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_TF32_OVERRIDE="${NVIDIA_TF32_OVERRIDE:-1}"

MODEL="${MODEL:-Qwen/Qwen3-1.7B-Base}"
DATASET="${DATASET:-open-web-math/open-web-math}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/a100/pretraining}"
OBJECTIVES="${OBJECTIVES:-ce rank_jsd ce_rank_jsd}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-4}"
RANK_CAP="${RANK_CAP:-64}"
RANK_WEIGHT="${RANK_WEIGHT:-1.0}"

mkdir -p "${RUN_ROOT}/logs"

for objective in ${OBJECTIVES}; do
    name="${MODEL##*/}_${objective}_s${SEED}"
    output_dir="${RUN_ROOT}/${name}"
    log_file="${RUN_ROOT}/logs/${name}.log"
    if [[ -f "${output_dir}/latest/config.json" ]]; then
        printf 'Skipping completed run: %s\n' "${name}"
        continue
    fi

    python experiments/pretrain_rank_jsd.py \
        --model "${MODEL}" \
        --dataset "${DATASET}" \
        --objective "${objective}" \
        --rank-weight "${RANK_WEIGHT}" \
        --rank-cap "${RANK_CAP}" \
        --sequence-length "${SEQUENCE_LENGTH}" \
        --micro-batch-size "${MICRO_BATCH_SIZE}" \
        --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
        --max-steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --output-dir "${output_dir}" \
        2>&1 | tee "${log_file}"
done
