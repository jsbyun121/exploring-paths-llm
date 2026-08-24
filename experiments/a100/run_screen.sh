#!/usr/bin/env bash

# Stage 1: one-seed mechanism and implementation audit.
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

SEED="${SEED:-0}"
METHODS="${METHODS:-drgrpo positive_ce bernoulli_kl_legacy bernoulli_kl_detached fixed_half_legacy fixed_half rank_jsd rank_jsd_entropy}"

for method in ${METHODS}; do
    case "${method}" in
        drgrpo)
            run_drgrpo "${SEED}"
            ;;
        rank_jsd_entropy)
            run_path rank_jsd "${RANK_ENTROPY_COEF:--0.03}" "${SEED}"
            ;;
        positive_ce|bernoulli_kl_legacy|bernoulli_kl_detached|fixed_half_legacy|fixed_half|rank_jsd)
            run_path "${method}" 0.0 "${SEED}"
            ;;
        *)
            printf 'Unknown method: %s\n' "${method}" >&2
            exit 2
            ;;
    esac
done
