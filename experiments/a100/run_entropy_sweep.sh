#!/usr/bin/env bash

# Section 4.2: determine whether fixed entropy penalties stabilize each loss.
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

SEED="${SEED:-0}"
OBJECTIVES="${OBJECTIVES:-bernoulli_kl_legacy bernoulli_kl_detached rank_jsd}"
COEFFICIENTS="${COEFFICIENTS:-0.0 -0.01 -0.03 -0.1}"

for objective in ${OBJECTIVES}; do
    for coefficient in ${COEFFICIENTS}; do
        run_path "${objective}" "${coefficient}" "${SEED}"
    done
done
