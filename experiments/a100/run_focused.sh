#!/usr/bin/env bash
# Bounded diagnostic: base/rank100 pass@8, then recover or rebuild CE100.
# No entropy sweep or automatic 300-step continuation.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MODEL="Qwen/Qwen3-4B-Thinking-2507"
export WANDB_MODE="${WANDB_MODE:-online}"
PROJECT="${PROJECT:-Exploring-Paths-Completion}"
FOCUSED_ROOT="${FOCUSED_ROOT:-${REPO_ROOT}/outputs/focused}"
EVAL_ROOT="${FOCUSED_ROOT}/evaluation"
RANK_CKPT="${RANK_CKPT:-${REPO_ROOT}/outputs/a100/qwen3-4b-thinking-2507_rank_jsd_h0p0_q0p0_s0/step100}"
RANK_MODEL="${REPO_ROOT}/outputs/a100/evaluation_models/rank_jsd_step100"
mkdir -p "${FOCUSED_ROOT}" "${EVAL_ROOT}"
if [[ -e "${FOCUSED_ROOT}/STOP" ]]; then
    printf 'Paused: remove %s/STOP before resuming.\n' "${FOCUSED_ROOT}" >&2
    exit 75
fi
exec 9>"${REPO_ROOT}/outputs/a100/.container2.lock"
if ! flock -n 9; then
    printf 'Another launcher holds the GPU lock.\n' >&2
    exit 1
fi
if pgrep -f 'RL2\.trainer\.(path|ppo)|experiments\.evaluate_reasoning' >/dev/null; then
    printf 'A training/evaluation process is already active.\n' >&2
    exit 1
fi

evaluate() {
    local model="$1" name="$2"
    if [[ -e "${FOCUSED_ROOT}/STOP" ]]; then
        printf 'Paused before starting %s.\n' "${name}"
        exit 75
    fi
    python -m experiments.evaluate_reasoning \
        --model "${model}" --split 'train[:500]' \
        --max-examples "${MAX_EXAMPLES:-200}" --sample-k 8 \
        --max-new-tokens 4096 --prompt-style training --seed 0 \
        --request-chunk-size 144 --gpu-memory-fraction 0.80 --save-generations \
        --wandb-project "${PROJECT}" --run-name "eval-${name}-val200-pass8" \
        --stop-file "${FOCUSED_ROOT}/STOP" \
        --output-dir "${EVAL_ROOT}/${name}" \
        2>&1 | tee -a "${FOCUSED_ROOT}/eval-${name}.log"
}

printf 'Phase 1: preserve rank-JSD step100 and evaluate base/rank100.\n'
CUDA_VISIBLE_DEVICES='' python -m experiments.export_path_checkpoint "${RANK_CKPT}" "${RANK_MODEL}"
evaluate "${MODEL}" base
evaluate "${RANK_MODEL}" rank_jsd_step100
python -m experiments.compare_evaluations "${EVAL_ROOT}/base" "${EVAL_ROOT}/rank_jsd_step100" \
    --output "${FOCUSED_ROOT}/base_vs_rank100.json"

# A recovered CE checkpoint may be supplied without retraining. Its completed
# training step must be 100 for the planned matched-step comparison.
if [[ -e "${FOCUSED_ROOT}/STOP" ]]; then
    printf 'Paused before the CE phase.\n'
    exit 75
fi
if [[ -n "${CE_MODEL_PATH:-}" ]]; then
    if [[ "${CE_MODEL_STEP:-}" != 100 ]]; then
        printf 'Set CE_MODEL_STEP=100 only after verifying the recovered model step.\n' >&2
        exit 1
    fi
    ce_model="${CE_MODEL_PATH}"
else
    printf 'Phase 2: no recovered CE100 supplied; rebuild only CE to step100.\n'
    export RUN_ROOT="${FOCUSED_ROOT}/training"
    export LOG_ROOT="${FOCUSED_ROOT}/training/logs"
    export MAX_STEPS=100 TEST_FREQ=25 SAVE_FREQ=10 KEEP_CHECKPOINTS=2 METHODS=positive_ce
    export ACTOR_TOKEN_BUDGET=6144
    # The original scheduler is constant (not constant_with_warmup), so
    # stopping at 100 does not alter its learning-rate schedule.
    export EXTRA_HYDRA_ARGS="${EXTRA_HYDRA_ARGS:+${EXTRA_HYDRA_ARGS} }trainer.stop_file=${FOCUSED_ROOT}/STOP"
    bash "${SCRIPT_DIR}/run_screen_container2.sh"
    ce_model="${RUN_ROOT}/qwen3-4b-thinking-2507_positive_ce_h0p0_q0p0_s0/latest"
    if [[ ! -f "${RUN_ROOT}/qwen3-4b-thinking-2507_positive_ce_h0p0_q0p0_s0/completed.json" ]]; then
        printf 'CE paused before step100; rerun this launcher to resume it.\n' >&2
        exit 1
    fi
fi
evaluate "${ce_model}" positive_ce_step100
python -m experiments.compare_evaluations "${EVAL_ROOT}/positive_ce_step100" "${EVAL_ROOT}/rank_jsd_step100" \
    --output "${FOCUSED_ROOT}/ce100_vs_rank100.json"
printf 'Focused diagnostic complete. Review ce100_vs_rank100.json before extending training.\n'
