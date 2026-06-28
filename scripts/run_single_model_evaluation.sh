#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/evaluation/single_model/default.yaml"
DEFAULT_VERL_ROOT="/root/autodl-tmp/verl"

export PYTHONPATH="${PROJECT_ROOT}:${DEFAULT_VERL_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

python scripts/evaluate_single_model.py --config "${DEFAULT_CONFIG}" "$@"
