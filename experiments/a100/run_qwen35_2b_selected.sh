#!/usr/bin/env bash

# Fast Qwen3.5-2B pilot after collecting the DrGRPO baseline through step 50.
# Positive CE was the strongest path objective in the historical 4B screen;
# rank-JSD with entropy is the remaining high-value mechanism check.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export METHODS="${METHODS:-positive_ce rank_jsd_entropy}"
export MAX_STEPS="${MAX_STEPS:-50}"
export TEST_FREQ="${TEST_FREQ:-25}"
export SAVE_FREQ="${SAVE_FREQ:-25}"

exec "${SCRIPT_DIR}/run_qwen35_2b_screen.sh"
