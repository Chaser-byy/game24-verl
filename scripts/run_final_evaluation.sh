#!/usr/bin/env bash
set -euo pipefail

RAW_MODEL_PATH="${RAW_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
SFT_MODEL_PATH="${SFT_MODEL_PATH:-}"
GRPO_RUN_DIR="${GRPO_RUN_DIR:-}"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-data/game24}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-outputs/evaluation}"

EVAL_MODE="${EVAL_MODE:-quick}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_SEED="${EVAL_SEED:-2026}"
EVAL_SAMPLE_N="${EVAL_SAMPLE_N:-8}"
EVAL_SAMPLE_TEMPERATURE="${EVAL_SAMPLE_TEMPERATURE:-0.7}"
EVAL_SAMPLE_TOP_P="${EVAL_SAMPLE_TOP_P:-0.95}"
EVAL_TORCH_DTYPE="${EVAL_TORCH_DTYPE:-bfloat16}"
EVAL_DIAGNOSTIC_LIMIT="${EVAL_DIAGNOSTIC_LIMIT:-20}"

if [[ -z "${SFT_MODEL_PATH}" ]]; then
  echo "ERROR: SFT_MODEL_PATH must point to the exported Hugging Face SFT model directory." >&2
  exit 2
fi

if [[ -z "${GRPO_RUN_DIR}" ]]; then
  echo "ERROR: GRPO_RUN_DIR must point to a GRPO run directory containing global_step_xxx checkpoints." >&2
  exit 2
fi

cat <<CONFIG
Game24 strict final evaluation
  EVAL_MODE=${EVAL_MODE}
  RAW_MODEL_PATH=${RAW_MODEL_PATH}
  SFT_MODEL_PATH=${SFT_MODEL_PATH}
  GRPO_RUN_DIR=${GRPO_RUN_DIR}
  EVAL_DATA_DIR=${EVAL_DATA_DIR}
  EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}
  EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS}
  EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE}
  EVAL_SEED=${EVAL_SEED}
  EVAL_SAMPLE_N=${EVAL_SAMPLE_N}
  EVAL_SAMPLE_TEMPERATURE=${EVAL_SAMPLE_TEMPERATURE}
  EVAL_SAMPLE_TOP_P=${EVAL_SAMPLE_TOP_P}
  EVAL_TORCH_DTYPE=${EVAL_TORCH_DTYPE}
  EVAL_DIAGNOSTIC_LIMIT=${EVAL_DIAGNOSTIC_LIMIT}
CONFIG

python scripts/final_evaluation.py \
  --mode "${EVAL_MODE}" \
  --raw-model-path "${RAW_MODEL_PATH}" \
  --sft-model-path "${SFT_MODEL_PATH}" \
  --grpo-run-dir "${GRPO_RUN_DIR}" \
  --data-dir "${EVAL_DATA_DIR}" \
  --output-dir "${EVAL_OUTPUT_DIR}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --seed "${EVAL_SEED}" \
  --sample-n "${EVAL_SAMPLE_N}" \
  --sample-temperature "${EVAL_SAMPLE_TEMPERATURE}" \
  --sample-top-p "${EVAL_SAMPLE_TOP_P}" \
  --torch-dtype "${EVAL_TORCH_DTYPE}" \
  --diagnostic-limit "${EVAL_DIAGNOSTIC_LIMIT}"
