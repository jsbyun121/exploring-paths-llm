#!/usr/bin/env bash

# Evaluate one base or trained Hugging Face model with greedy and sampled runs.
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ $# -lt 1 ]]; then
    printf 'Usage: %s MODEL_OR_CHECKPOINT [OUTPUT_NAME]\n' "$0" >&2
    exit 2
fi

MODEL_PATH="$1"
OUTPUT_NAME="${2:-$(basename "${MODEL_PATH}")}"
EVAL_ROOT="${EVAL_ROOT:-${REPO_ROOT}/outputs/a100/evaluation}"
SAMPLE_K="${SAMPLE_K:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SEED="${SEED:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-max_split_size_mb:512}"
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${EVAL_ROOT}/${OUTPUT_NAME}"
EVAL_ARGS=(
    --model "${MODEL_PATH}"
    --dataset "${EVAL_DATASET:-openai/gsm8k}"
    --dataset-config "${EVAL_DATASET_CONFIG:-main}"
    --split "${EVAL_SPLIT:-train[:500]}"
    --prompt-style "${PROMPT_STYLE:-training}"
    --question-column "${QUESTION_COLUMN:-question}"
    --answer-column "${ANSWER_COLUMN:-answer}"
    --sample-k "${SAMPLE_K}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
    --output-dir "${EVAL_ROOT}/${OUTPUT_NAME}"
)
if [[ -n "${MAX_EXAMPLES:-}" ]]; then
    EVAL_ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi
if [[ "${SAVE_GENERATIONS:-0}" == "1" ]]; then
    EVAL_ARGS+=(--save-generations)
fi

python -m experiments.evaluate_reasoning "${EVAL_ARGS[@]}" \
    2>&1 | tee "${EVAL_ROOT}/${OUTPUT_NAME}.log"
