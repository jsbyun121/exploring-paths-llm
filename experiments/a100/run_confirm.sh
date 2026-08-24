#!/usr/bin/env bash

# Stage 2: confirm only the shortlisted objective(s) with three seeds.
# Override, for example:
#   CONDITIONS='rank_jsd:-0.03 positive_ce:0.0' bash .../run_confirm.sh
set -Eeuo pipefail

export MAX_STEPS="${MAX_STEPS:-300}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
source "$(dirname "$0")/common.sh"

CONDITIONS="${CONDITIONS:-rank_jsd:-0.03 positive_ce:0.0}"
SEEDS="${SEEDS:-0 1 2}"
TOKEN_ENTROPY_QUANTILES="${TOKEN_ENTROPY_QUANTILES:-0.0}"

for condition in ${CONDITIONS}; do
    objective="${condition%%:*}"
    coefficient="${condition#*:}"
    if [[ "${objective}" == "${coefficient}" ]]; then
        printf 'Condition must use objective:entropy_coef syntax: %s\n' "${condition}" >&2
        exit 2
    fi
    for quantile in ${TOKEN_ENTROPY_QUANTILES}; do
        for seed in ${SEEDS}; do
            run_path "${objective}" "${coefficient}" "${seed}" "${quantile}"
        done
    done
done
