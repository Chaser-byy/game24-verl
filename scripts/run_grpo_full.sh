#!/usr/bin/env bash
set -euo pipefail

# Target verl version: v0.7.1
# Full-parameter GRPO after SFT. This script is intentionally separate from
# scripts/run_grpo_lora.sh and does not pass LoRA alpha/target_modules settings.

FULL_GRPO_MODEL_PATH="${FULL_GRPO_MODEL_PATH:-}"
FULL_GRPO_TRAIN_FILE="${FULL_GRPO_TRAIN_FILE:-data/game24/train.parquet}"
FULL_GRPO_VAL_FILE="${FULL_GRPO_VAL_FILE:-data/game24/val.parquet}"
FULL_GRPO_REWARD_FILE="${FULL_GRPO_REWARD_FILE:-game24/reward.py}"
FULL_GRPO_OUTPUT_DIR="${FULL_GRPO_OUTPUT_DIR:-outputs/game24-grpo-full-param}"
FULL_GRPO_PROJECT_NAME="${FULL_GRPO_PROJECT_NAME:-game24-grpo-full-param}"
FULL_GRPO_EXPERIMENT_NAME="${FULL_GRPO_EXPERIMENT_NAME:-game24-grpo-full-param}"
FULL_GRPO_LOGGER="${FULL_GRPO_LOGGER:-console}"

FULL_GRPO_TRAIN_BATCH_SIZE="${FULL_GRPO_TRAIN_BATCH_SIZE:-8}"
FULL_GRPO_ROLLOUT_N="${FULL_GRPO_ROLLOUT_N:-8}"
FULL_GRPO_MAX_PROMPT_LENGTH="${FULL_GRPO_MAX_PROMPT_LENGTH:-192}"
FULL_GRPO_MAX_RESPONSE_LENGTH="${FULL_GRPO_MAX_RESPONSE_LENGTH:-192}"
FULL_GRPO_LEARNING_RATE="${FULL_GRPO_LEARNING_RATE:-3e-7}"
FULL_GRPO_TOTAL_EPOCHS="${FULL_GRPO_TOTAL_EPOCHS:-30}"
FULL_GRPO_TOTAL_TRAINING_STEPS="${FULL_GRPO_TOTAL_TRAINING_STEPS:-200}"
FULL_GRPO_SAVE_FREQ="${FULL_GRPO_SAVE_FREQ:-25}"
FULL_GRPO_TEST_FREQ="${FULL_GRPO_TEST_FREQ:-25}"
FULL_GRPO_VAL_BEFORE_TRAIN="${FULL_GRPO_VAL_BEFORE_TRAIN:-true}"
FULL_GRPO_GPU_MEMORY_UTILIZATION="${FULL_GRPO_GPU_MEMORY_UTILIZATION:-0.35}"
FULL_GRPO_N_GPUS="${FULL_GRPO_N_GPUS:-1}"
FULL_GRPO_DTYPE="${FULL_GRPO_DTYPE:-bfloat16}"
FULL_GRPO_TEMPERATURE="${FULL_GRPO_TEMPERATURE:-1.0}"
FULL_GRPO_TOP_P="${FULL_GRPO_TOP_P:-0.95}"
FULL_GRPO_ATTN_IMPLEMENTATION="${FULL_GRPO_ATTN_IMPLEMENTATION:-sdpa}"
FULL_GRPO_USE_REMOVE_PADDING="${FULL_GRPO_USE_REMOVE_PADDING:-false}"
FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING="${FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING:-true}"
FULL_GRPO_USE_KL_LOSS="${FULL_GRPO_USE_KL_LOSS:-true}"
FULL_GRPO_KL_LOSS_COEF="${FULL_GRPO_KL_LOSS_COEF:-0.001}"
FULL_GRPO_KL_LOSS_TYPE="${FULL_GRPO_KL_LOSS_TYPE:-low_var_kl}"
FULL_GRPO_CHECKPOINT_SAVE_CONTENTS="${FULL_GRPO_CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}"

FULL_GRPO_PPO_MINI_BATCH_SIZE="${FULL_GRPO_PPO_MINI_BATCH_SIZE:-8}"
FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU="${FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"

fail() {
  echo "ERROR: $*" >&2
  exit 2
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  if ! is_positive_integer "${value}"; then
    fail "${name} must be a positive integer, got '${value}'"
  fi
}

require_boolean() {
  local name="$1"
  local value="$2"
  case "${value}" in
    true | false | True | False) ;;
    *) fail "${name} must be true or false, got '${value}'" ;;
  esac
}

if [[ -z "${FULL_GRPO_MODEL_PATH}" ]]; then
  cat >&2 <<ERROR
FULL_GRPO_MODEL_PATH must point to the exported Hugging Face SFT model directory.
Example:
  FULL_GRPO_MODEL_PATH=/root/autodl-tmp/outputs/game24-sft-full/global_step_363/huggingface bash scripts/run_grpo_full.sh
ERROR
  exit 2
fi

require_positive_integer "FULL_GRPO_TRAIN_BATCH_SIZE" "${FULL_GRPO_TRAIN_BATCH_SIZE}"
require_positive_integer "FULL_GRPO_ROLLOUT_N" "${FULL_GRPO_ROLLOUT_N}"
require_positive_integer "FULL_GRPO_MAX_PROMPT_LENGTH" "${FULL_GRPO_MAX_PROMPT_LENGTH}"
require_positive_integer "FULL_GRPO_MAX_RESPONSE_LENGTH" "${FULL_GRPO_MAX_RESPONSE_LENGTH}"
require_positive_integer "FULL_GRPO_TOTAL_TRAINING_STEPS" "${FULL_GRPO_TOTAL_TRAINING_STEPS}"
require_positive_integer "FULL_GRPO_SAVE_FREQ" "${FULL_GRPO_SAVE_FREQ}"
require_positive_integer "FULL_GRPO_TEST_FREQ" "${FULL_GRPO_TEST_FREQ}"
require_positive_integer "FULL_GRPO_N_GPUS" "${FULL_GRPO_N_GPUS}"
require_positive_integer "FULL_GRPO_PPO_MINI_BATCH_SIZE" "${FULL_GRPO_PPO_MINI_BATCH_SIZE}"
require_positive_integer "FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU" "${FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
require_boolean "FULL_GRPO_VAL_BEFORE_TRAIN" "${FULL_GRPO_VAL_BEFORE_TRAIN}"
require_boolean "FULL_GRPO_USE_REMOVE_PADDING" "${FULL_GRPO_USE_REMOVE_PADDING}"
require_boolean "FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING" "${FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING}"
require_boolean "FULL_GRPO_USE_KL_LOSS" "${FULL_GRPO_USE_KL_LOSS}"

if (( FULL_GRPO_PPO_MINI_BATCH_SIZE > FULL_GRPO_TRAIN_BATCH_SIZE )); then
  fail "FULL_GRPO_PPO_MINI_BATCH_SIZE (${FULL_GRPO_PPO_MINI_BATCH_SIZE}) cannot be greater than FULL_GRPO_TRAIN_BATCH_SIZE (${FULL_GRPO_TRAIN_BATCH_SIZE})"
fi

if (( FULL_GRPO_TRAIN_BATCH_SIZE % FULL_GRPO_PPO_MINI_BATCH_SIZE != 0 )); then
  fail "FULL_GRPO_TRAIN_BATCH_SIZE (${FULL_GRPO_TRAIN_BATCH_SIZE}) must be divisible by FULL_GRPO_PPO_MINI_BATCH_SIZE (${FULL_GRPO_PPO_MINI_BATCH_SIZE})"
fi

ACTOR_MICRO_GLOBAL=$((FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS))
if (( FULL_GRPO_PPO_MINI_BATCH_SIZE % ACTOR_MICRO_GLOBAL != 0 )); then
  fail "FULL_GRPO_PPO_MINI_BATCH_SIZE (${FULL_GRPO_PPO_MINI_BATCH_SIZE}) must be divisible by FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS (${ACTOR_MICRO_GLOBAL})"
fi

ROLLOUT_BATCH_SIZE=$((FULL_GRPO_TRAIN_BATCH_SIZE * FULL_GRPO_ROLLOUT_N))
LOG_PROB_MICRO_GLOBAL=$((FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS))
REF_LOG_PROB_MICRO_GLOBAL=$((FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS))

if (( ROLLOUT_BATCH_SIZE % LOG_PROB_MICRO_GLOBAL != 0 )); then
  fail "FULL_GRPO_TRAIN_BATCH_SIZE * FULL_GRPO_ROLLOUT_N (${ROLLOUT_BATCH_SIZE}) must be divisible by FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS (${LOG_PROB_MICRO_GLOBAL})"
fi

if (( ROLLOUT_BATCH_SIZE % REF_LOG_PROB_MICRO_GLOBAL != 0 )); then
  fail "FULL_GRPO_TRAIN_BATCH_SIZE * FULL_GRPO_ROLLOUT_N (${ROLLOUT_BATCH_SIZE}) must be divisible by FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU * FULL_GRPO_N_GPUS (${REF_LOG_PROB_MICRO_GLOBAL})"
fi

if [[ "${FULL_GRPO_LOGGER}" == \[* ]]; then
  TRAINER_LOGGER="${FULL_GRPO_LOGGER}"
else
  TRAINER_LOGGER="[\"${FULL_GRPO_LOGGER}\"]"
fi

parameter_summary() {
  python - "${FULL_GRPO_MODEL_PATH}" <<'PY'
import sys

model_path = sys.argv[1]
try:
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    total = sum(parameter.numel() for parameter in model.parameters())
except Exception as exc:
    raise SystemExit(f"failed to count parameters from {model_path!r}: {type(exc).__name__}: {exc}") from exc

print(f"  FULL_GRPO_TOTAL_PARAMETERS={total}")
print(f"  FULL_GRPO_TRAINABLE_PARAMETERS={total}")
print("  FULL_GRPO_TRAINABLE_PARAMETER_RATIO=1.000000")
PY
}

PARAMETER_SUMMARY="$(parameter_summary)"

cat <<CONFIG
Game24 verl full-parameter GRPO configuration after SFT warm start
  Target verl version: v0.7.1
  LORA_ENABLED=false
  FULL_PARAMETER_TRAINING=true
  actor_rollout_ref.model.lora_rank=0
  LoRA adapter loading=false
  LoRA alpha/target_modules overrides=not passed
${PARAMETER_SUMMARY}
  FULL_GRPO_MODEL_PATH=${FULL_GRPO_MODEL_PATH}
  FULL_GRPO_TRAIN_FILE=${FULL_GRPO_TRAIN_FILE}
  FULL_GRPO_VAL_FILE=${FULL_GRPO_VAL_FILE}
  FULL_GRPO_REWARD_FILE=${FULL_GRPO_REWARD_FILE}
  FULL_GRPO_OUTPUT_DIR=${FULL_GRPO_OUTPUT_DIR}
  FULL_GRPO_PROJECT_NAME=${FULL_GRPO_PROJECT_NAME}
  FULL_GRPO_EXPERIMENT_NAME=${FULL_GRPO_EXPERIMENT_NAME}
  FULL_GRPO_LOGGER=${FULL_GRPO_LOGGER}
  TRAINER_LOGGER=${TRAINER_LOGGER}
  FULL_GRPO_TRAIN_BATCH_SIZE=${FULL_GRPO_TRAIN_BATCH_SIZE}
  FULL_GRPO_ROLLOUT_N=${FULL_GRPO_ROLLOUT_N}
  FULL_GRPO_ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}
  FULL_GRPO_PPO_MINI_BATCH_SIZE=${FULL_GRPO_PPO_MINI_BATCH_SIZE}
  FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=${FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}
  FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  FULL_GRPO_MAX_PROMPT_LENGTH=${FULL_GRPO_MAX_PROMPT_LENGTH}
  FULL_GRPO_MAX_RESPONSE_LENGTH=${FULL_GRPO_MAX_RESPONSE_LENGTH}
  FULL_GRPO_LEARNING_RATE=${FULL_GRPO_LEARNING_RATE}
  FULL_GRPO_TOTAL_TRAINING_STEPS=${FULL_GRPO_TOTAL_TRAINING_STEPS}
  FULL_GRPO_SAVE_FREQ=${FULL_GRPO_SAVE_FREQ}
  FULL_GRPO_TEST_FREQ=${FULL_GRPO_TEST_FREQ}
  FULL_GRPO_VAL_BEFORE_TRAIN=${FULL_GRPO_VAL_BEFORE_TRAIN}
  FULL_GRPO_CHECKPOINT_SAVE_CONTENTS=${FULL_GRPO_CHECKPOINT_SAVE_CONTENTS}
  FULL_GRPO_GPU_MEMORY_UTILIZATION=${FULL_GRPO_GPU_MEMORY_UTILIZATION}
  FULL_GRPO_N_GPUS=${FULL_GRPO_N_GPUS}
  FULL_GRPO_DTYPE=${FULL_GRPO_DTYPE}
  FULL_GRPO_TEMPERATURE=${FULL_GRPO_TEMPERATURE}
  FULL_GRPO_TOP_P=${FULL_GRPO_TOP_P}
  FULL_GRPO_ATTN_IMPLEMENTATION=${FULL_GRPO_ATTN_IMPLEMENTATION}
  FULL_GRPO_USE_REMOVE_PADDING=${FULL_GRPO_USE_REMOVE_PADDING}
  FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING=${FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING}
  FULL_GRPO_USE_KL_LOSS=${FULL_GRPO_USE_KL_LOSS}
  FULL_GRPO_KL_LOSS_COEF=${FULL_GRPO_KL_LOSS_COEF}
  FULL_GRPO_KL_LOSS_TYPE=${FULL_GRPO_KL_LOSS_TYPE}
CONFIG

python -m verl.trainer.main_ppo \
  data.train_files="${FULL_GRPO_TRAIN_FILE}" \
  data.val_files="${FULL_GRPO_VAL_FILE}" \
  data.train_batch_size="${FULL_GRPO_TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${FULL_GRPO_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${FULL_GRPO_MAX_RESPONSE_LENGTH}" \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${FULL_GRPO_MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${FULL_GRPO_ENABLE_GRADIENT_CHECKPOINTING}" \
  ++actor_rollout_ref.model.override_config.attn_implementation="${FULL_GRPO_ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${FULL_GRPO_USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.actor.optim.lr="${FULL_GRPO_LEARNING_RATE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${FULL_GRPO_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss="${FULL_GRPO_USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${FULL_GRPO_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.kl_loss_type="${FULL_GRPO_KL_LOSS_TYPE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.dtype="${FULL_GRPO_DTYPE}" \
  actor_rollout_ref.actor.checkpoint.save_contents="${FULL_GRPO_CHECKPOINT_SAVE_CONTENTS}" \
  actor_rollout_ref.actor.checkpoint.load_contents="${FULL_GRPO_CHECKPOINT_SAVE_CONTENTS}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.dtype="${FULL_GRPO_DTYPE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${FULL_GRPO_ROLLOUT_N}" \
  actor_rollout_ref.rollout.dtype="${FULL_GRPO_DTYPE}" \
  actor_rollout_ref.rollout.temperature="${FULL_GRPO_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${FULL_GRPO_TOP_P}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${FULL_GRPO_N_GPUS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${FULL_GRPO_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  reward.custom_reward_function.path="${FULL_GRPO_REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name="${FULL_GRPO_PROJECT_NAME}" \
  trainer.experiment_name="${FULL_GRPO_EXPERIMENT_NAME}" \
  trainer.default_local_dir="${FULL_GRPO_OUTPUT_DIR}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.n_gpus_per_node="${FULL_GRPO_N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${FULL_GRPO_SAVE_FREQ}" \
  trainer.test_freq="${FULL_GRPO_TEST_FREQ}" \
  trainer.val_before_train="${FULL_GRPO_VAL_BEFORE_TRAIN}" \
  trainer.total_training_steps="${FULL_GRPO_TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs="${FULL_GRPO_TOTAL_EPOCHS}" \
  trainer.resume_mode=disable
