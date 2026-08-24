#!/usr/bin/env bash

# Remaining Stage-1 objectives assigned to the original A100 container.
# The already-running fixed_half_legacy job should be allowed to finish before
# launching this script; it will then be detected as complete and skipped.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If this shard is launched while the pre-split legacy job is still active,
# wait without claiming GPU memory. The original all-method launcher was
# stopped separately so it cannot advance to the other shard's objectives.
legacy_pattern='RL2\.trainer\.path .*actor\.path\.objective=fixed_half_legacy'
if pgrep -f "${legacy_pattern}" >/dev/null; then
    printf 'Waiting for the existing fixed_half_legacy trainer to finish...\n'
    while pgrep -f "${legacy_pattern}" >/dev/null; do
        sleep 30
    done
fi

# Retire any stopped pre-split launcher after its active child has exited. A
# pending TERM is delivered when CONT releases the stopped shell.
while read -r launcher_pid launcher_state launcher_command; do
    if [[ "${launcher_state}" == T* && "${launcher_command}" == "bash experiments/a100/run_screen.sh" ]]; then
        kill -TERM "${launcher_pid}" 2>/dev/null || true
        kill -CONT "${launcher_pid}" 2>/dev/null || true
    fi
done < <(ps -eo pid=,stat=,args=)

export METHODS="${METHODS:-fixed_half_legacy fixed_half}"
exec "${SCRIPT_DIR}/run_screen.sh"
