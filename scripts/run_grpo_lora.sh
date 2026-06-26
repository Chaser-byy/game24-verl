#!/usr/bin/env bash
set -euo pipefail

# Target verl version: v0.7.1
# This script is not intended for verl v0.8.0 or the main branch.

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-data/game24/train.parquet}"
VAL_FILE="${VAL_FILE:-data/game24/val.parquet}"
REWARD_FILE="${REWARD_FILE:-game24/reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/game24-grpo-lora}"
PROJECT_NAME="${PROJECT_NAME:-game24-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25-1p5b-grpo-lora}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-8}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-25}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
N_GPUS="${N_GPUS:-1}"
DTYPE="${DTYPE:-bfloat16}"

cat <<CONFIG
Game24 verl GRPO LoRA configuration
  Target verl version: v0.7.1
  MODEL_PATH=${MODEL_PATH}
  TRAIN_FILE=${TRAIN_FILE}
  VAL_FILE=${VAL_FILE}
  REWARD_FILE=${REWARD_FILE}
  OUTPUT_DIR=${OUTPUT_DIR}
  PROJECT_NAME=${PROJECT_NAME}
  EXPERIMENT_NAME=${EXPERIMENT_NAME}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
  ROLLOUT_N=${ROLLOUT_N}
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH}
  MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}
  LORA_RANK=${LORA_RANK}
  LORA_ALPHA=${LORA_ALPHA}
  LEARNING_RATE=${LEARNING_RATE}
  TOTAL_EPOCHS=${TOTAL_EPOCHS}
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
  SAVE_FREQ=${SAVE_FREQ}
  TEST_FREQ=${TEST_FREQ}
  GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
  N_GPUS=${N_GPUS}
  DTYPE=${DTYPE}
CONFIG

EXTRA_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  EXTRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

python -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.dtype="${DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.dtype="${DTYPE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.dtype="${DTYPE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  reward.custom_reward_function.path="${REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.logger='["console","wandb"]' \
  trainer.n_gpus_per_node="${N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.resume_mode=disable \
  "${EXTRA_ARGS[@]}"
