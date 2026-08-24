#!/usr/bin/env bash

# Remaining Stage-1 objectives assigned to the second A100 container.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export METHODS="${METHODS:-rank_jsd rank_jsd_entropy}"
exec "${SCRIPT_DIR}/run_screen.sh"
