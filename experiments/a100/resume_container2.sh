#!/usr/bin/env bash

# Resume only rank-JSD. Entropy sweeps are deliberately no longer queued.
# Usage: bash experiments/a100/resume_container2.sh
# With step100, the default bounded continuation is steps 101-125.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/a100}"
export MODEL="Qwen/Qwen3-4B-Thinking-2507"
export SEED="${SEED:-0}"
export MAX_STEPS="${MAX_STEPS:-125}"
export METHODS="rank_jsd"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
RUN_NAME="qwen3-4b-thinking-2507_rank_jsd_h0p0_q0p0_s${SEED:-0}"
RUN_DIR="${RUN_ROOT}/${RUN_NAME}"
LOCK_FILE="${RUN_ROOT}/.container2.lock"

mkdir -p "${RUN_ROOT}"

# Keep accidental double launches from sharing the same GPU and output tree.
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    printf 'Another container-2 launcher already holds %s\n' "${LOCK_FILE}" >&2
    exit 1
fi

if pgrep -af 'RL2\.trainer\.(path|ppo)' >/dev/null; then
    printf 'A training process is already running; refusing to launch a duplicate.\n' >&2
    pgrep -af 'RL2\.trainer\.(path|ppo)' >&2
    exit 1
fi

if [[ -f "${RUN_DIR}/completed.json" ]]; then
    printf 'Checking the completed rank_jsd run against the requested target.\n'
else
    # Match the trainer's checkpoint selection: only completed DCP writes
    # have .metadata. Refuse to silently restart this experiment from zero.
    latest_step=""
    if [[ -d "${RUN_DIR}" ]]; then
        latest_step="$(find "${RUN_DIR}" -mindepth 2 -maxdepth 2 -type f -name .metadata -printf '%h\n' \
            | sed -n 's|.*/step\([0-9][0-9]*\)$|\1|p' | sort -n | tail -n 1)"
    fi
    if [[ -z "${latest_step}" ]] || (( 10#${latest_step} < 100 )); then
        printf 'Expected a completed rank_jsd checkpoint at step 100 or later under %s.\n' "${RUN_DIR}" >&2
        exit 1
    fi
    printf 'Resuming rank_jsd from step%s; next training step is %s (target: %s).\n' \
        "${latest_step}" "$((10#${latest_step} + 1))" "${MAX_STEPS}"
fi

printf 'Only rank_jsd is enabled. Target: %s; checkpoint interval: %s.\n' "${MAX_STEPS}" "${SAVE_FREQ}"

exec "${SCRIPT_DIR}/run_screen_container2.sh"
