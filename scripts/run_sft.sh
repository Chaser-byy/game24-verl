#!/usr/bin/env bash
set -euo pipefail

# Target verl version: v0.7.1
# This script runs full-parameter SFT before GRPO warm start.

MODEL_PATH="${MODEL_PATH:-${SFT_MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-data/game24-sft/sft_train.parquet}"
SFT_VAL_FILE="${SFT_VAL_FILE:-data/game24-sft/sft_val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${SFT_OUTPUT_DIR:-outputs/game24-sft-full}}"
PROJECT_NAME="${PROJECT_NAME:-game24-verl}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${SFT_EXPERIMENT_NAME:-qwen25-1p5b-sft-full}}"

N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-${SFT_TRAIN_BATCH_SIZE:-32}}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${SFT_MICRO_BATCH_SIZE:-4}}"
MAX_LENGTH="${MAX_LENGTH:-${SFT_MAX_LENGTH:-512}}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${SFT_MAX_TOKEN_LEN_PER_GPU:-8192}}"
LEARNING_RATE="${LEARNING_RATE:-${SFT_LEARNING_RATE:-5e-6}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${SFT_TOTAL_EPOCHS:-1}}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-${SFT_TOTAL_TRAINING_STEPS:-}}"
SAVE_FREQ="${SAVE_FREQ:-${SFT_SAVE_FREQ:--1}}"
TEST_FREQ="${TEST_FREQ:-${SFT_TEST_FREQ:--1}}"
DTYPE="${DTYPE:-bfloat16}"
LOGGER="${LOGGER:-[\"console\",\"wandb\"]}"
CHECKPOINT_SAVE_CONTENTS="${CHECKPOINT_SAVE_CONTENTS:-${SFT_CHECKPOINT_SAVE_CONTENTS:-[\"model\",\"optimizer\",\"extra\",\"hf_model\"]}}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION-${SFT_ATTENTION_IMPLEMENTATION-sdpa}}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING-${SFT_USE_REMOVE_PADDING-False}}"

HYDRA_ARGS=(
  "data.train_files=${SFT_TRAIN_FILE}"
  "data.val_files=${SFT_VAL_FILE}"
  "data.messages_key=messages"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE}"
  "data.max_length=${MAX_LENGTH}"
  "data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "data.use_dynamic_bsz=True"
  "data.pad_mode=no_padding"
  "data.truncation=error"
  "model.path=${MODEL_PATH}"
  "model.trust_remote_code=True"
  "model.enable_gradient_checkpointing=True"
  "model.lora_rank=0"
  "engine.strategy=fsdp"
  "engine.dtype=${DTYPE}"
  "engine.param_offload=False"
  "engine.optimizer_offload=False"
  "optim.lr=${LEARNING_RATE}"
  "checkpoint.save_contents=${CHECKPOINT_SAVE_CONTENTS}"
  "checkpoint.load_contents=${CHECKPOINT_SAVE_CONTENTS}"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.default_local_dir=${OUTPUT_DIR}"
  "trainer.logger=${LOGGER}"
  "trainer.n_gpus_per_node=${N_GPUS}"
  "trainer.nnodes=1"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.resume_mode=disable"
)

# override_config is an existing structured Hydra dict, but attn_implementation
# is a dynamic Hugging Face config key. Use per-key ++ so it works whether the
# nested key is absent in verl v0.7.1 or predeclared by a future/user config.
if [[ -n "${ATTENTION_IMPLEMENTATION}" ]]; then
  HYDRA_ARGS+=("++model.override_config.attn_implementation=${ATTENTION_IMPLEMENTATION}")
fi

if [[ -n "${USE_REMOVE_PADDING}" ]]; then
  HYDRA_ARGS+=("model.use_remove_padding=${USE_REMOVE_PADDING}")
fi

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
  HYDRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

cat <<CONFIG
Game24 verl SFT configuration
  Target verl version: v0.7.1
  MODEL_PATH=${MODEL_PATH}
  SFT_TRAIN_FILE=${SFT_TRAIN_FILE}
  SFT_VAL_FILE=${SFT_VAL_FILE}
  OUTPUT_DIR=${OUTPUT_DIR}
  PROJECT_NAME=${PROJECT_NAME}
  EXPERIMENT_NAME=${EXPERIMENT_NAME}
  N_GPUS=${N_GPUS}
  TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}
  MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}
  MAX_LENGTH=${MAX_LENGTH}
  MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU}
  LEARNING_RATE=${LEARNING_RATE}
  TOTAL_EPOCHS=${TOTAL_EPOCHS}
  TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS}
  SAVE_FREQ=${SAVE_FREQ}
  TEST_FREQ=${TEST_FREQ}
  DTYPE=${DTYPE}
  ATTENTION_IMPLEMENTATION=${ATTENTION_IMPLEMENTATION:-<not set>}
  USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-<not set>}
  LOGGER=${LOGGER}
  CHECKPOINT_SAVE_CONTENTS=${CHECKPOINT_SAVE_CONTENTS}
CONFIG

printf 'Hydra overrides:\n'
printf '  %s\n' "${HYDRA_ARGS[@]}"

torchrun --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
  -m verl.trainer.sft_trainer \
  "${HYDRA_ARGS[@]}"
