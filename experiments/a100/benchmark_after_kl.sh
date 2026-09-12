#!/usr/bin/env bash
set -Eeuo pipefail
cd /workspace/repos/exploring-paths-llm
source experiments/a100/common.sh
exec 9>outputs/a100/.container2.lock
flock 9
[[ -f outputs/focused/jsd_beta1_zero_vs_kl_beta1_zero100.json ]]
[[ ! -e outputs/focused/STOP ]]
if pgrep -f 'RL2\.trainer\.(path|ppo)|experiments\.train_long|experiments\.evaluate_reasoning' >/dev/null; then exit 1; fi
export PYTORCH_ALLOC_CONF=max_split_size_mb:512
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=4
# Read-only forward/backward profiling: no optimizer updates or model export.
torchrun --standalone --nproc_per_node=1 -m experiments.benchmark_policy_actor --repeats 2
