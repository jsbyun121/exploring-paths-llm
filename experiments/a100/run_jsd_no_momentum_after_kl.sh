#!/usr/bin/env bash
# Queue Rank-JSD beta1=0 after Rank-KL and its final evaluations.
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
export FOCUSED_ROOT="${FOCUSED_ROOT:-${REPO_ROOT}/outputs/focused}"
EVAL_ROOT="${FOCUSED_ROOT}/evaluation"
RANK_CKPT="${RANK_CKPT:-${REPO_ROOT}/outputs/a100/qwen3-4b-thinking-2507_rank_jsd_h0p0_q0p0_s0/step100}"
RANK_MODEL="${REPO_ROOT}/outputs/a100/evaluation_models/rank_jsd_step100"
mkdir -p "${FOCUSED_ROOT}" "${EVAL_ROOT}"
if [[ -e "${FOCUSED_ROOT}/STOP" ]]; then
    printf 'Paused: remove %s/STOP before resuming.\n' "${FOCUSED_ROOT}" >&2
    exit 75
fi
exec 9>"${REPO_ROOT}/outputs/a100/.container2.lock"
printf 'Waiting for Rank-KL to finish and release the GPU lock.\n'
flock 9
python - <<'CHECK'
import json, os
from pathlib import Path
root = Path(os.environ["FOCUSED_ROOT"])
p = root / "rankjsd100_vs_rankkl100.json"
a = json.loads(p.read_text())
assert a['examples'] == 200, 'Expected completed matched 200-example comparison'
for name in ['positive_ce_step100', 'rank_jsd_step100', 'rank_kl_step100']:
    a = json.loads((root / 'evaluation' / name / 'summary.json').read_text())
    assert a['examples'] == 200 and a['sample_k'] == 8
print('Rank-KL evaluation completed; starting Rank-JSD with beta1=0.')
CHECK
[[ ! -e "${FOCUSED_ROOT}/STOP" ]] || exit 75
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


# Keep the final model and newest complete recovery point for prior runs.
python - <<'CLEAN'
import json, os, shutil
from pathlib import Path
root = Path(os.environ['FOCUSED_ROOT'])
removed = []
for method in ['positive_ce', 'rank_kl']:
    run = root / 'training' / f'qwen3-4b-thinking-2507_{method}_h0p0_q0p0_s0'
    assert json.loads((run / 'completed.json').read_text())['step'] == 100
    assert (run / 'latest/model.safetensors').is_file()
    assert (run / 'step100/.metadata').is_file()
    assert (run / 'step100/__0_0.distcp').is_file()
    old = run / 'step90'
    if old.exists():
        shutil.rmtree(old)
        removed.append(str(old))
# This isolated speed-test step used the superseded optimizer restore code;
# its measurement JSON and all original research checkpoints remain intact.
old = Path('outputs/speed_validation/qwen3-4b-thinking-2507_rank_jsd_h0p0_q0p0_s0/step101')
if old.exists():
    assert Path('outputs/analysis/speed_validation_summary.json').is_file()
    shutil.rmtree(old)
    removed.append(str(old))
(root / 'jsd_beta1_zero_cleanup.json').write_text(json.dumps({'removed':removed}, indent=2))
assert shutil.disk_usage(root).free > 80 * 1024**3, 'Insufficient disk for atomic checkpoints'
CLEAN

export RUN_ROOT="${FOCUSED_ROOT}/training_jsd_beta1_zero"
export LOG_ROOT="${RUN_ROOT}/logs"
export MAX_STEPS=100 TEST_FREQ=25 SAVE_FREQ=10 KEEP_CHECKPOINTS=2 METHODS=rank_jsd
export ACTOR_TOKEN_BUDGET=6144
export EXTRA_HYDRA_ARGS="actor.adam_betas=[0.0,0.999] trainer.experiment_name=qwen3-4b-thinking-2507_rank_jsd_beta1_zero_s0 trainer.stop_file=${FOCUSED_ROOT}/STOP"
bash "${SCRIPT_DIR}/run_screen_container2.sh"
rank_kl_dir="${RUN_ROOT}/qwen3-4b-thinking-2507_rank_jsd_h0p0_q0p0_s0"
[[ -f "${rank_kl_dir}/completed.json" ]] || { printf 'Rank-JSD beta1=0 paused or incomplete.\n'; exit 1; }
evaluate "${rank_kl_dir}/latest" rank_jsd_beta1_zero_step100
python -m experiments.compare_evaluations "${EVAL_ROOT}/positive_ce_step100" "${EVAL_ROOT}/rank_jsd_beta1_zero_step100" --output "${FOCUSED_ROOT}/ce100_vs_jsd_beta1_zero100.json"
python -m experiments.compare_evaluations "${EVAL_ROOT}/rank_jsd_step100" "${EVAL_ROOT}/rank_jsd_beta1_zero_step100" --output "${FOCUSED_ROOT}/rankjsd100_vs_jsd_beta1_zero100.json"
printf 'Rank-JSD beta1=0 training and matched comparisons complete.\n'
