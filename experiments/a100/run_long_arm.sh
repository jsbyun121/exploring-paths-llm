#!/usr/bin/env bash
# Optional per-arm time cap; zero means validation stopping only.
set -Eeuo pipefail
cd /workspace/repos/exploring-paths-llm
arm="${1:?Pass ce, jsd, dr_grpo or sapo}"
seconds="${2:-0}"
case "$arm" in ce|jsd|dr_grpo|sapo) ;; *) exit 2;; esac
[[ "$seconds" =~ ^[0-9]+$ ]] || exit 2
[[ -f outputs/focused/jsd_beta1_zero_vs_kl_beta1_zero100.json ]]
[[ ! -e outputs/long/STOP ]]
source experiments/a100/common.sh
export WANDB_MODE=online
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=4
exec 9>outputs/a100/.container2.lock
flock -n 9 || { echo 'GPU busy'; exit 1; }
python - <<'PY'
import shutil
assert shutil.disk_usage('outputs').free>64*1024**3, 'Need 64 GiB for checkpoint and best-model atomic staging; archive verified old runs first.'
PY
resume=()
if compgen -G "outputs/long/${arm}/step*/.metadata" > /dev/null; then resume+=(trainer.load_ckpt_from=latest); fi
torchrun --standalone --nproc_per_node=1 -m experiments.train_long \
 --config-path "$(pwd)/experiments/a100/long_comparison" --config-name "$arm" \
 "trainer.long_budget_seconds=$seconds" "${resume[@]}"
