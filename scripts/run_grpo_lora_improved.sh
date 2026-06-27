#!/usr/bin/env bash
set -euo pipefail

# Target verl version: v0.7.1
# Improved LoRA GRPO entry. This starts from the exported SFT Hugging Face model,
# never resumes old GRPO checkpoints, and keeps old LoRA/full-parameter scripts intact.

IMPROVED_GRPO_MODEL_PATH="${IMPROVED_GRPO_MODEL_PATH:-}"
IMPROVED_GRPO_TRAIN_FILE="${IMPROVED_GRPO_TRAIN_FILE:-data/game24/train.parquet}"
IMPROVED_GRPO_VAL_FILE="${IMPROVED_GRPO_VAL_FILE:-data/game24/val.parquet}"
IMPROVED_GRPO_TEST_FILE="${IMPROVED_GRPO_TEST_FILE:-data/game24/test.parquet}"
IMPROVED_GRPO_HARD100_FILE="${IMPROVED_GRPO_HARD100_FILE:-data/game24/tot_hard100.parquet}"
IMPROVED_GRPO_SFT_TRAIN_FILE="${IMPROVED_GRPO_SFT_TRAIN_FILE:-data/game24-sft/sft_train.parquet}"
IMPROVED_GRPO_REWARD_FILE="${IMPROVED_GRPO_REWARD_FILE:-game24/reward_strict.py}"
IMPROVED_GRPO_REWARD_MANAGER_FILE="${IMPROVED_GRPO_REWARD_MANAGER_FILE:-game24/strict_dapo_reward_manager.py}"
IMPROVED_GRPO_OUTPUT_DIR="${IMPROVED_GRPO_OUTPUT_DIR:-outputs/game24-grpo-lora-improved}"
IMPROVED_GRPO_PROJECT_NAME="${IMPROVED_GRPO_PROJECT_NAME:-game24-grpo-lora-improved}"
IMPROVED_GRPO_EXPERIMENT_NAME="${IMPROVED_GRPO_EXPERIMENT_NAME:-game24-grpo-lora-improved}"
IMPROVED_GRPO_LOGGER="${IMPROVED_GRPO_LOGGER:-console}"

IMPROVED_GRPO_TRAIN_BATCH_SIZE="${IMPROVED_GRPO_TRAIN_BATCH_SIZE:-8}"
IMPROVED_GRPO_GEN_BATCH_SIZE="${IMPROVED_GRPO_GEN_BATCH_SIZE:-24}"
IMPROVED_GRPO_MAX_GEN_BATCHES="${IMPROVED_GRPO_MAX_GEN_BATCHES:-4}"
IMPROVED_GRPO_ROLLOUT_N="${IMPROVED_GRPO_ROLLOUT_N:-16}"
IMPROVED_GRPO_MAX_PROMPT_LENGTH="${IMPROVED_GRPO_MAX_PROMPT_LENGTH:-192}"
IMPROVED_GRPO_MAX_RESPONSE_LENGTH="${IMPROVED_GRPO_MAX_RESPONSE_LENGTH:-192}"

IMPROVED_GRPO_LORA_RANK="${IMPROVED_GRPO_LORA_RANK:-64}"
IMPROVED_GRPO_LORA_ALPHA="${IMPROVED_GRPO_LORA_ALPHA:-64}"
IMPROVED_GRPO_TARGET_MODULES="${IMPROVED_GRPO_TARGET_MODULES:-all-linear}"
IMPROVED_GRPO_LORA_DROPOUT="${IMPROVED_GRPO_LORA_DROPOUT:-0}"

IMPROVED_GRPO_PPO_MINI_BATCH_SIZE="${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE:-4}"
IMPROVED_GRPO_PPO_EPOCHS="${IMPROVED_GRPO_PPO_EPOCHS:-2}"
IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU="${IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU:-32}"
IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"
IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-64}"

IMPROVED_GRPO_LEARNING_RATE="${IMPROVED_GRPO_LEARNING_RATE:-5e-6}"
IMPROVED_GRPO_GRAD_CLIP="${IMPROVED_GRPO_GRAD_CLIP:-1.0}"
IMPROVED_GRPO_TEMPERATURE="${IMPROVED_GRPO_TEMPERATURE:-1.0}"
IMPROVED_GRPO_TOP_P="${IMPROVED_GRPO_TOP_P:-1.0}"
IMPROVED_GRPO_TOP_K="${IMPROVED_GRPO_TOP_K:--1}"
IMPROVED_GRPO_USE_KL_LOSS="${IMPROVED_GRPO_USE_KL_LOSS:-false}"
IMPROVED_GRPO_KL_LOSS_COEF="${IMPROVED_GRPO_KL_LOSS_COEF:-0}"
IMPROVED_GRPO_CLIP_RATIO_LOW="${IMPROVED_GRPO_CLIP_RATIO_LOW:-0.20}"
IMPROVED_GRPO_CLIP_RATIO_HIGH="${IMPROVED_GRPO_CLIP_RATIO_HIGH:-0.28}"
IMPROVED_GRPO_CLIP_RATIO_C="${IMPROVED_GRPO_CLIP_RATIO_C:-10.0}"
IMPROVED_GRPO_LOSS_AGG_MODE="${IMPROVED_GRPO_LOSS_AGG_MODE:-seq-mean-token-mean}"

IMPROVED_GRPO_ATTN_IMPLEMENTATION="${IMPROVED_GRPO_ATTN_IMPLEMENTATION:-sdpa}"
IMPROVED_GRPO_USE_REMOVE_PADDING="${IMPROVED_GRPO_USE_REMOVE_PADDING:-false}"
IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING="${IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING:-false}"
IMPROVED_GRPO_GPU_MEMORY_UTILIZATION="${IMPROVED_GRPO_GPU_MEMORY_UTILIZATION:-0.45}"
IMPROVED_GRPO_N_GPUS="${IMPROVED_GRPO_N_GPUS:-1}"
IMPROVED_GRPO_DTYPE="${IMPROVED_GRPO_DTYPE:-bfloat16}"

IMPROVED_GRPO_TOTAL_EPOCHS="${IMPROVED_GRPO_TOTAL_EPOCHS:-30}"
IMPROVED_GRPO_TOTAL_TRAINING_STEPS="${IMPROVED_GRPO_TOTAL_TRAINING_STEPS:-200}"
IMPROVED_GRPO_SAVE_FREQ="${IMPROVED_GRPO_SAVE_FREQ:-20}"
IMPROVED_GRPO_TEST_FREQ="${IMPROVED_GRPO_TEST_FREQ:-10}"
IMPROVED_GRPO_VAL_BEFORE_TRAIN="${IMPROVED_GRPO_VAL_BEFORE_TRAIN:-true}"
IMPROVED_GRPO_AUDIT_OUTPUT="${IMPROVED_GRPO_AUDIT_OUTPUT:-${IMPROVED_GRPO_OUTPUT_DIR}/data_boundary_audit.json}"

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

if [[ -z "${IMPROVED_GRPO_MODEL_PATH}" ]]; then
  cat >&2 <<ERROR
IMPROVED_GRPO_MODEL_PATH must point to the exported Hugging Face SFT model directory.
This improved run starts fresh from SFT and never loads an old GRPO LoRA adapter.
ERROR
  exit 2
fi

require_positive_integer "IMPROVED_GRPO_TRAIN_BATCH_SIZE" "${IMPROVED_GRPO_TRAIN_BATCH_SIZE}"
require_positive_integer "IMPROVED_GRPO_GEN_BATCH_SIZE" "${IMPROVED_GRPO_GEN_BATCH_SIZE}"
require_positive_integer "IMPROVED_GRPO_MAX_GEN_BATCHES" "${IMPROVED_GRPO_MAX_GEN_BATCHES}"
require_positive_integer "IMPROVED_GRPO_ROLLOUT_N" "${IMPROVED_GRPO_ROLLOUT_N}"
require_positive_integer "IMPROVED_GRPO_PPO_MINI_BATCH_SIZE" "${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE}"
require_positive_integer "IMPROVED_GRPO_PPO_EPOCHS" "${IMPROVED_GRPO_PPO_EPOCHS}"
require_positive_integer "IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU" "${IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU" "${IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
require_positive_integer "IMPROVED_GRPO_TOTAL_TRAINING_STEPS" "${IMPROVED_GRPO_TOTAL_TRAINING_STEPS}"
require_positive_integer "IMPROVED_GRPO_SAVE_FREQ" "${IMPROVED_GRPO_SAVE_FREQ}"
require_positive_integer "IMPROVED_GRPO_TEST_FREQ" "${IMPROVED_GRPO_TEST_FREQ}"
require_positive_integer "IMPROVED_GRPO_N_GPUS" "${IMPROVED_GRPO_N_GPUS}"
require_boolean "IMPROVED_GRPO_USE_KL_LOSS" "${IMPROVED_GRPO_USE_KL_LOSS}"
require_boolean "IMPROVED_GRPO_USE_REMOVE_PADDING" "${IMPROVED_GRPO_USE_REMOVE_PADDING}"
require_boolean "IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING" "${IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING}"
require_boolean "IMPROVED_GRPO_VAL_BEFORE_TRAIN" "${IMPROVED_GRPO_VAL_BEFORE_TRAIN}"

if (( IMPROVED_GRPO_PPO_MINI_BATCH_SIZE > IMPROVED_GRPO_TRAIN_BATCH_SIZE )); then
  fail "IMPROVED_GRPO_PPO_MINI_BATCH_SIZE (${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE}) cannot be greater than IMPROVED_GRPO_TRAIN_BATCH_SIZE (${IMPROVED_GRPO_TRAIN_BATCH_SIZE})"
fi

if (( IMPROVED_GRPO_TRAIN_BATCH_SIZE % IMPROVED_GRPO_PPO_MINI_BATCH_SIZE != 0 )); then
  fail "IMPROVED_GRPO_TRAIN_BATCH_SIZE (${IMPROVED_GRPO_TRAIN_BATCH_SIZE}) must be divisible by IMPROVED_GRPO_PPO_MINI_BATCH_SIZE (${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE})"
fi

if (( IMPROVED_GRPO_GEN_BATCH_SIZE < IMPROVED_GRPO_TRAIN_BATCH_SIZE )); then
  fail "IMPROVED_GRPO_GEN_BATCH_SIZE (${IMPROVED_GRPO_GEN_BATCH_SIZE}) must be at least IMPROVED_GRPO_TRAIN_BATCH_SIZE (${IMPROVED_GRPO_TRAIN_BATCH_SIZE})"
fi

ROLLOUT_TRAJECTORIES=$((IMPROVED_GRPO_TRAIN_BATCH_SIZE * IMPROVED_GRPO_ROLLOUT_N))
if (( ROLLOUT_TRAJECTORIES % IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  fail "effective rollout trajectories (${ROLLOUT_TRAJECTORIES}) must be divisible by IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU (${IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU})"
fi

GEN_ROLLOUT_TRAJECTORIES=$((IMPROVED_GRPO_GEN_BATCH_SIZE * IMPROVED_GRPO_ROLLOUT_N))
if (( GEN_ROLLOUT_TRAJECTORIES % IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  fail "generated rollout trajectories (${GEN_ROLLOUT_TRAJECTORIES}) must be divisible by IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU (${IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU})"
fi

if (( GEN_ROLLOUT_TRAJECTORIES % IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU != 0 )); then
  fail "generated rollout trajectories (${GEN_ROLLOUT_TRAJECTORIES}) must be divisible by IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU (${IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU})"
fi

mkdir -p "${IMPROVED_GRPO_OUTPUT_DIR}"

python scripts/audit_game24_boundaries.py \
  --train-file "${IMPROVED_GRPO_TRAIN_FILE}" \
  --val-file "${IMPROVED_GRPO_VAL_FILE}" \
  --test-file "${IMPROVED_GRPO_TEST_FILE}" \
  --hard100-file "${IMPROVED_GRPO_HARD100_FILE}" \
  --sft-train-file "${IMPROVED_GRPO_SFT_TRAIN_FILE}" \
  --output-json "${IMPROVED_GRPO_AUDIT_OUTPUT}"

if [[ "${IMPROVED_GRPO_LOGGER}" == \[* ]]; then
  TRAINER_LOGGER="${IMPROVED_GRPO_LOGGER}"
else
  TRAINER_LOGGER="[\"${IMPROVED_GRPO_LOGGER}\"]"
fi

cat <<CONFIG
Game24 improved LoRA GRPO configuration after SFT warm start
  Target verl version: v0.7.1
  Fresh start from SFT: true
  Resume old GRPO checkpoint: false
  Load old GRPO LoRA adapter: false
  Strict reward: score=1.0 iff verify_solution(...).is_correct else 0.0
  Reward manager: game24_strict_dapo
  Group metric: acc (strict is_correct)
  IMPROVED_GRPO_MODEL_PATH=${IMPROVED_GRPO_MODEL_PATH}
  IMPROVED_GRPO_TRAIN_FILE=${IMPROVED_GRPO_TRAIN_FILE}
  IMPROVED_GRPO_VAL_FILE=${IMPROVED_GRPO_VAL_FILE}
  Frozen audit-only files: test=${IMPROVED_GRPO_TEST_FILE}, hard100=${IMPROVED_GRPO_HARD100_FILE}
  IMPROVED_GRPO_SFT_TRAIN_FILE=${IMPROVED_GRPO_SFT_TRAIN_FILE}
  IMPROVED_GRPO_OUTPUT_DIR=${IMPROVED_GRPO_OUTPUT_DIR}
  IMPROVED_GRPO_PROJECT_NAME=${IMPROVED_GRPO_PROJECT_NAME}
  IMPROVED_GRPO_EXPERIMENT_NAME=${IMPROVED_GRPO_EXPERIMENT_NAME}
  TRAINER_LOGGER=${TRAINER_LOGGER}
  IMPROVED_GRPO_TRAIN_BATCH_SIZE=${IMPROVED_GRPO_TRAIN_BATCH_SIZE}
  IMPROVED_GRPO_GEN_BATCH_SIZE=${IMPROVED_GRPO_GEN_BATCH_SIZE}
  IMPROVED_GRPO_MAX_GEN_BATCHES=${IMPROVED_GRPO_MAX_GEN_BATCHES}
  IMPROVED_GRPO_ROLLOUT_N=${IMPROVED_GRPO_ROLLOUT_N}
  IMPROVED_GRPO_MAX_PROMPT_LENGTH=${IMPROVED_GRPO_MAX_PROMPT_LENGTH}
  IMPROVED_GRPO_MAX_RESPONSE_LENGTH=${IMPROVED_GRPO_MAX_RESPONSE_LENGTH}
  IMPROVED_GRPO_LORA_RANK=${IMPROVED_GRPO_LORA_RANK}
  IMPROVED_GRPO_LORA_ALPHA=${IMPROVED_GRPO_LORA_ALPHA}
  IMPROVED_GRPO_TARGET_MODULES=${IMPROVED_GRPO_TARGET_MODULES}
  IMPROVED_GRPO_LORA_DROPOUT=${IMPROVED_GRPO_LORA_DROPOUT} (not passed: no verified verl v0.7.1 model config key)
  IMPROVED_GRPO_PPO_MINI_BATCH_SIZE=${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE}
  IMPROVED_GRPO_PPO_EPOCHS=${IMPROVED_GRPO_PPO_EPOCHS}
  IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=${IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}
  IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
  IMPROVED_GRPO_LEARNING_RATE=${IMPROVED_GRPO_LEARNING_RATE}
  IMPROVED_GRPO_GRAD_CLIP=${IMPROVED_GRPO_GRAD_CLIP} (not passed: no verified verl v0.7.1 actor config key)
  IMPROVED_GRPO_TEMPERATURE=${IMPROVED_GRPO_TEMPERATURE}
  IMPROVED_GRPO_TOP_P=${IMPROVED_GRPO_TOP_P}
  IMPROVED_GRPO_TOP_K=${IMPROVED_GRPO_TOP_K}
  IMPROVED_GRPO_USE_KL_LOSS=${IMPROVED_GRPO_USE_KL_LOSS}
  IMPROVED_GRPO_KL_LOSS_COEF=${IMPROVED_GRPO_KL_LOSS_COEF}
  IMPROVED_GRPO_CLIP_RATIO_LOW=${IMPROVED_GRPO_CLIP_RATIO_LOW}
  IMPROVED_GRPO_CLIP_RATIO_HIGH=${IMPROVED_GRPO_CLIP_RATIO_HIGH}
  IMPROVED_GRPO_CLIP_RATIO_C=${IMPROVED_GRPO_CLIP_RATIO_C}
  IMPROVED_GRPO_LOSS_AGG_MODE=${IMPROVED_GRPO_LOSS_AGG_MODE}
  IMPROVED_GRPO_ATTN_IMPLEMENTATION=${IMPROVED_GRPO_ATTN_IMPLEMENTATION}
  IMPROVED_GRPO_USE_REMOVE_PADDING=${IMPROVED_GRPO_USE_REMOVE_PADDING}
  IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING=${IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING}
  IMPROVED_GRPO_TOTAL_TRAINING_STEPS=${IMPROVED_GRPO_TOTAL_TRAINING_STEPS}
  IMPROVED_GRPO_TEST_FREQ=${IMPROVED_GRPO_TEST_FREQ}
  IMPROVED_GRPO_SAVE_FREQ=${IMPROVED_GRPO_SAVE_FREQ}
  Validation decoding: actor_rollout_ref.rollout.val_kwargs.do_sample=False, n=1, temperature=0
  Data boundary audit: ${IMPROVED_GRPO_AUDIT_OUTPUT}
CONFIG

python -m verl.trainer.main_ppo \
  data.train_files="${IMPROVED_GRPO_TRAIN_FILE}" \
  data.val_files="${IMPROVED_GRPO_VAL_FILE}" \
  data.train_batch_size="${IMPROVED_GRPO_TRAIN_BATCH_SIZE}" \
  ++data.gen_batch_size="${IMPROVED_GRPO_GEN_BATCH_SIZE}" \
  data.max_prompt_length="${IMPROVED_GRPO_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${IMPROVED_GRPO_MAX_RESPONSE_LENGTH}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0 \
  actor_rollout_ref.model.path="${IMPROVED_GRPO_MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing="${IMPROVED_GRPO_ENABLE_GRADIENT_CHECKPOINTING}" \
  ++actor_rollout_ref.model.override_config.attn_implementation="${IMPROVED_GRPO_ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${IMPROVED_GRPO_USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.lora_rank="${IMPROVED_GRPO_LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${IMPROVED_GRPO_LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules="${IMPROVED_GRPO_TARGET_MODULES}" \
  actor_rollout_ref.actor.optim.lr="${IMPROVED_GRPO_LEARNING_RATE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${IMPROVED_GRPO_PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs="${IMPROVED_GRPO_PPO_EPOCHS}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_kl_loss="${IMPROVED_GRPO_USE_KL_LOSS}" \
  actor_rollout_ref.actor.kl_loss_coef="${IMPROVED_GRPO_KL_LOSS_COEF}" \
  actor_rollout_ref.actor.clip_ratio_low="${IMPROVED_GRPO_CLIP_RATIO_LOW}" \
  actor_rollout_ref.actor.clip_ratio_high="${IMPROVED_GRPO_CLIP_RATIO_HIGH}" \
  actor_rollout_ref.actor.clip_ratio_c="${IMPROVED_GRPO_CLIP_RATIO_C}" \
  actor_rollout_ref.actor.loss_agg_mode="${IMPROVED_GRPO_LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.dtype="${IMPROVED_GRPO_DTYPE}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.fsdp_config.dtype="${IMPROVED_GRPO_DTYPE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${IMPROVED_GRPO_ROLLOUT_N}" \
  actor_rollout_ref.rollout.dtype="${IMPROVED_GRPO_DTYPE}" \
  actor_rollout_ref.rollout.temperature="${IMPROVED_GRPO_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${IMPROVED_GRPO_TOP_P}" \
  actor_rollout_ref.rollout.top_k="${IMPROVED_GRPO_TOP_K}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${IMPROVED_GRPO_N_GPUS}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${IMPROVED_GRPO_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  reward.custom_reward_function.path="${IMPROVED_GRPO_REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  reward.reward_manager.source=register \
  reward.reward_manager.name=game24_strict_dapo \
  reward.reward_manager.module.path="${IMPROVED_GRPO_REWARD_MANAGER_FILE}" \
  reward.reward_manager.module.name=Game24StrictDAPORewardManager \
  trainer.project_name="${IMPROVED_GRPO_PROJECT_NAME}" \
  trainer.experiment_name="${IMPROVED_GRPO_EXPERIMENT_NAME}" \
  trainer.default_local_dir="${IMPROVED_GRPO_OUTPUT_DIR}" \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.n_gpus_per_node="${IMPROVED_GRPO_N_GPUS}" \
  trainer.nnodes=1 \
  trainer.save_freq="${IMPROVED_GRPO_SAVE_FREQ}" \
  trainer.test_freq="${IMPROVED_GRPO_TEST_FREQ}" \
  trainer.val_before_train="${IMPROVED_GRPO_VAL_BEFORE_TRAIN}" \
  trainer.total_training_steps="${IMPROVED_GRPO_TOTAL_TRAINING_STEPS}" \
  trainer.total_epochs="${IMPROVED_GRPO_TOTAL_EPOCHS}" \
  trainer.resume_mode=disable
