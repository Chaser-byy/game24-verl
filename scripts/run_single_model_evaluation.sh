#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PROJECT_ROOT}/configs/evaluation/single_model/default.yaml"
DEFAULT_VERL_ROOT="/root/autodl-tmp/verl"

export PYTHONPATH="${PROJECT_ROOT}:${DEFAULT_VERL_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

SINGLE_EVAL_CONFIG="${SINGLE_EVAL_CONFIG:-${DEFAULT_CONFIG}}"
SINGLE_EVAL_PASS_N_VALUE="${SINGLE_EVAL_PASS_N:-${SINGLE_EVAL_SAMPLE_N:-}}"
SINGLE_EVAL_DO_SAMPLE_DISPLAY="${SINGLE_EVAL_DO_SAMPLE:-auto/config default}"
if [[ -z "${SINGLE_EVAL_DO_SAMPLE:-}" && -n "${SINGLE_EVAL_PASS_N_VALUE}" && "${SINGLE_EVAL_PASS_N_VALUE}" != "1" ]]; then
  SINGLE_EVAL_DO_SAMPLE_DISPLAY="auto(true because sample_pass_n>1)"
fi

ARGS=(--config "${SINGLE_EVAL_CONFIG}")

if [[ -n "${SINGLE_EVAL_DATA_SPLIT:-}" ]]; then
  ARGS+=(--data-split "${SINGLE_EVAL_DATA_SPLIT}")
fi
if [[ -n "${SINGLE_EVAL_BATCH_SIZE:-}" ]]; then
  ARGS+=(--batch-size "${SINGLE_EVAL_BATCH_SIZE}")
fi
if [[ -n "${SINGLE_EVAL_MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "${SINGLE_EVAL_MAX_NEW_TOKENS}")
fi
if [[ -n "${SINGLE_EVAL_SEED:-}" ]]; then
  ARGS+=(--seed "${SINGLE_EVAL_SEED}")
fi
if [[ -n "${SINGLE_EVAL_PASS_N_VALUE}" ]]; then
  ARGS+=(--sample-pass-n "${SINGLE_EVAL_PASS_N_VALUE}")
fi
if [[ -n "${SINGLE_EVAL_DO_SAMPLE:-}" ]]; then
  ARGS+=(--do-sample "${SINGLE_EVAL_DO_SAMPLE}")
fi
if [[ -n "${SINGLE_EVAL_TEMPERATURE:-}" ]]; then
  ARGS+=(--temperature "${SINGLE_EVAL_TEMPERATURE}")
fi
if [[ -n "${SINGLE_EVAL_TOP_P:-}" ]]; then
  ARGS+=(--top-p "${SINGLE_EVAL_TOP_P}")
fi
if [[ -n "${SINGLE_EVAL_TOP_K:-}" ]]; then
  ARGS+=(--top-k "${SINGLE_EVAL_TOP_K}")
fi
if [[ -n "${SINGLE_EVAL_OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${SINGLE_EVAL_OUTPUT_DIR}")
fi
if [[ "${SINGLE_EVAL_OVERWRITE:-false}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

cat <<EOF
Single-model evaluation launch
  config=${SINGLE_EVAL_CONFIG}
  data_split=${SINGLE_EVAL_DATA_SPLIT:-config default}
  batch_size=${SINGLE_EVAL_BATCH_SIZE:-config default}
  max_new_tokens=${SINGLE_EVAL_MAX_NEW_TOKENS:-config default}
  seed=${SINGLE_EVAL_SEED:-config default}
  do_sample=${SINGLE_EVAL_DO_SAMPLE_DISPLAY}
  sample_pass_n=${SINGLE_EVAL_PASS_N_VALUE:-config default}
  temperature=${SINGLE_EVAL_TEMPERATURE:-config default}
  top_p=${SINGLE_EVAL_TOP_P:-config default}
  top_k=${SINGLE_EVAL_TOP_K:-config default}
  output_dir=${SINGLE_EVAL_OUTPUT_DIR:-auto}
EOF

python scripts/evaluate_single_model.py "${ARGS[@]}" "$@"
