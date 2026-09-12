#!/usr/bin/env bash
# Run on the Mac only after the sequential A100 training queue has released the GPU/disk.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${A100_HOST:-a100}"
remote_repo="/workspace/repos/exploring-paths-llm"
remote_models="$remote_repo/outputs/forgetting/models"
remote_recovery="$remote_repo/outputs/forgetting/recovery/jsd_step220"

step120="$repo/snapshots/d49e1f087c2a91313f02c3a23118752693014e5289d3188c00a5bf74e61af014/files/best"
step180="$repo/snapshots/73ff744a4957a264a4ec34bcdce3f6ae683919548710f02bf2925a7194795e1c/files/best"
step220="$repo/snapshots/25e70705e5cc93a03fea75fea1a7726ececb7cf8bb6e39a633449461c2479b24/files"

[[ "$(jq -r .step "$step120/training_state.json")" == 120 ]]
[[ "$(jq -r .step "$step180/training_state.json")" == 180 ]]
[[ "$(jq -r .step "$step220/long_state.json")" == 220 ]]
[[ "$(jq -r .step "$step220/best/training_state.json")" == 180 ]] || {
  echo "Refusing: step220 best-model guard changed unexpectedly" >&2
  exit 1
}

if ssh -T "$host" "pgrep -f '[e]xperiments.train_long' >/dev/null"; then
  echo "Refusing to stage while A100 long training is active" >&2
  exit 1
fi

ssh -T "$host" "mkdir -p '$remote_models/jsd_step120/best' '$remote_models/jsd_step180/best' '$remote_recovery'"
rsync -rt --partial --inplace --checksum "$step120/" "$host:$remote_models/jsd_step120/best/"
rsync -rt --partial --inplace --checksum "$step180/" "$host:$remote_models/jsd_step180/best/"

# Upload only the actual step-220 recovery files, never its step-180 best export.
rsync -rt --partial --inplace --checksum \
  "$step220/.metadata" "$step220/__0_0.distcp" "$step220/recovery.json" \
  "$host:$remote_recovery/"
ssh -T "$host" "cd '$remote_repo' && CUDA_VISIBLE_DEVICES= .venv/bin/python -m experiments.export_path_checkpoint '$remote_recovery' '$remote_models/jsd_step220/best'"

ssh -T "$host" "cd '$remote_repo' && bash experiments/forgetting/run_suite.sh --dry-run"
echo "JSD models staged. The uploaded step220 recovery remains intact for audit."
