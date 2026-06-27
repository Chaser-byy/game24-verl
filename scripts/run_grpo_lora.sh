#!/usr/bin/env bash
set -euo pipefail

# Target verl version: v0.7.1
# This script is not intended for verl v0.8.0 or the main branch.
# Recommended route: run full-parameter SFT first, export a Hugging Face model,
# then set GRPO_MODEL_PATH to that exported SFT model directory before launching GRPO.

MODEL_PATH="${GRPO_MODEL_PATH:-${MODEL_PATH:-}}"
GRPO_TRAIN_FILE="${GRPO_TRAIN_FILE:-data/game24/train.parquet}"
GRPO_VAL_FILE="${GRPO_VAL_FILE:-data/game24/val.parquet}"
GRPO_REWARD_FILE="${GRPO_REWARD_FILE:-game24/reward.py}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-outputs/game24-grpo-smoke}"
GRPO_PROJECT_NAME="${GRPO_PROJECT_NAME:-game24-grpo}"
GRPO_EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME:-game24-grpo-smoke}"
GRPO_LOGGER="${GRPO_LOGGER:-console}"

GRPO_TRAIN_BATCH_SIZE="${GRPO_TRAIN_BATCH_SIZE:-16}"
GRPO_ROLLOUT_N="${GRPO_ROLLOUT_N:-16}"
GRPO_MAX_PROMPT_LENGTH="${GRPO_MAX_PROMPT_LENGTH:-192}"
GRPO_MAX_RESPONSE_LENGTH="${GRPO_MAX_RESPONSE_LENGTH:-256}"
GRPO_LORA_RANK="${GRPO_LORA_RANK:-64}"
GRPO_LORA_ALPHA="${GRPO_LORA_ALPHA:-64}"
GRPO_LEARNING_RATE="${GRPO_LEARNING_RATE:-1e-6}"
GRPO_TOTAL_EPOCHS="${GRPO_TOTAL_EPOCHS:-8}"
GRPO_TOTAL_TRAINING_STEPS="${GRPO_TOTAL_TRAINING_STEPS:-400}"
GRPO_SAVE_FREQ="${GRPO_SAVE_FREQ:-50}"
GRPO_TEST_FREQ="${GRPO_TEST_FREQ:-25}"
GRPO_GPU_MEMORY_UTILIZATION="${GRPO_GPU_MEMORY_UTILIZATION:-0.45}"
GRPO_N_GPUS="${GRPO_N_GPUS:-${N_GPUS:-1}}"
GRPO_DTYPE="${GRPO_DTYPE:-${DTYPE:-bfloat16}}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
GRPO_TOP_P="${GRPO_TOP_P:-0.95}"

GRPO_PPO_MINI_BATCH_SIZE="${GRPO_PPO_MINI_BATCH_SIZE:-${GRPO_TRAIN_BATCH_SIZE}}"
GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU="${GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

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

if [[ -z "${MODEL_PATH}" ]]; then
  cat >&2 <<ERROR
MODEL_PATH must point to the exported Hugging Face SFT model directory.
Run scripts/run_sft.sh first, export/merge the SFT checkpoint if needed, then retry:
  GRPO_MODEL_PATH=/path/to/exported-sft-hf-model bash scripts/run_grpo_lora.sh
ERROR
  exit 2
fi

require_positive_integer "GRPO_TRAIN_BATCH_SIZE" "${GRPO_TRAIN_BATCH_SIZE}"
require_positive_integer "GRPO_PPO_MINI_BATCH_SIZE" "${GRPO_PPO_MINI_BATCH_SIZE}"
require_positive_integer "GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU" "${GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"

if (( GRPO_PPO_MINI_BATCH_SIZE > GRPO_TRAIN_BATCH_SIZE )); then
  fail "GRPO_PPO_MINI_BATCH_SIZE (${GRPO_PPO_MINI_BATCH_SIZE}) cannot be greater than GRPO_TRAIN_BATCH_SIZE (${GRPO_TRAIN_BATCH_SIZE})"
fi

if (( GRPO_TRAIN_BATCH_SIZE % GRPO_PPO_MINI_BATCH_SIZE != 0 )); then
  fail "GRPO_TRAIN_BATCH_SIZE (${GRPO_TRAIN_BATCH_SIZE}) must be divisible by GRPO_PPO_MINI_BATCH_SIZE (${GRPO_PPO_MINI_BATCH_SIZE})"
fi

if [[ "${GRPO_LOGGER}" == \[* ]]; then
  TRAINER_LOGGER="${GRPO_LOGGER}"
else
  TRAINER_LOGGER="[\"${GRPO_LOGGER}\"]"
fi

cat <<CONFIG
Game24 verl GRPO LoRA configuration after SFT warm start
  Target verl version: v0.7.1
  MODEL_PATH=${MODEL_PATH}
  GRPO_TRAIN_FILE=${GRPO_TRAIN_FILE}
  GRPO_VAL_FILE=${GRPO_VAL_FILE}
  GRPO_REWARD_FILE=${GRPO_REWARD_FILE}
  GRPO_OUTPUT_DIR=${GRPO_OUTPUT_DIR}
  GRPO_PROJECT_NAME=${GRPO_PROJECT_NAME}
  GRPO_EXPERIMENT_NAME=${GRPO_EXPERIMENT_NAME}
  GRPO_LOGGER=${GRPO_LOGGER}
  TRAINER_LOGGER=${TRAINER_LOGGER}
  GRPO_TRAIN_BATCH_SIZE=${GRPO_TRAIN_BATCH_SIZE}
  GRPO_ROLLOUT_N=${GRPO_ROLLOUT_N}
  GRPO_PPO_MINI_BATCH_SIZE=${GRPO_PPO_MINI_BATCH_SIZE}
  GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=${GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}
  GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  GRPO_MAX_PROMPT_LENGTH=${GRPO_MAX_PROMPT_LENGTH}
  GRPO_MAX_RESPONSE_LENGTH=${GRPO_MAX_RESPONSE_LENGTH}
  GRPO_LORA_RANK=${GRPO_LORA_RANK}
  GRPO_LORA_ALPHA=${GRPO_LORA_ALPHA}
  GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE}
  GRPO_TOTAL_EPOCHS=${GRPO_TOTAL_EPOCHS}
  GRPO_TOTAL_TRAINING_STEPS=${GRPO_TOTAL_TRAINING_STEPS}
  GRPO_SAVE_FREQ=${GRPO_SAVE_FREQ}
  GRPO_TEST_FREQ=${GRPO_TEST_FREQ}
  GRPO_GPU_MEMORY_UTILIZATION=${GRPO_GPU_MEMORY_UTILIZATION}
  GRPO_N_GPUS=${GRPO_N_GPUS}
  GRPO_DTYPE=${GRPO_DTYPE}
  GRPO_TEMPERATURE=${GRPO_TEMPERATURE}
  GRPO_TOP_P=${GRPO_TOP_P}
CONFIG

EXTRA_ARGS=()
if [[ -n "${GRPO_TOTAL_TRAINING_STEPS}" ]]; then
  EXTRA_ARGS+=("trainer.total_training_steps=${GRPO_TOTAL_TRAINING_STEPS}")
fi

python -m verl.trainer.main_ppo \
  data.train_files="${GRPO_TRAIN_FILE}" \
  data.val_files="${GRPO_VAL_FILE}" \
  data.train_batch_size="${GRPO_TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${GRPO_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${GRPO_MAX_RESPONSE_LENGTH}" \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="${GRPO_LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${GRPO_LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr="${GRPO_LEARNING_RATE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${GRPO_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.dtype="${GRPO_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.dtype="${GRPO_DTYPE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${GRPO_ROLLOUT_N}" \
  actor_rollout_ref.rollout.dtype="${GRPO_DTYPE}" \
  actor_rollout_ref.rollout.temperature="${GRPO_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${GRPO_TOP_P}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GRPO_N_GPUS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GRPO_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  reward.custom_reward_function.path="${GRPO_REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name="${GRPO_PROJECT_NAME}" \
  trainer.experiment_name="${GRPO_EXPERIMENT_NAME}" \
  trainer.default_local_dir="${GRPO_OUTPUT_DIR}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.n_gpus_per_node="${GRPO_N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${GRPO_SAVE_FREQ}" \
  trainer.test_freq="${GRPO_TEST_FREQ}" \
  trainer.total_epochs="${GRPO_TOTAL_EPOCHS}" \
  trainer.resume_mode=disable \
  "${EXTRA_ARGS[@]}"
