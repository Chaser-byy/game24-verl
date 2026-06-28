# game24-verl

`game24-verl` is a fresh, standalone Game of 24 task layer for external `verl` v0.7.1 training. It does not copy, migrate, or depend on any older TRL or GRPO project.

The recommended route is now optimized for final correctness:

```text
exact solver SFT data
-> Qwen2.5-1.5B-Instruct short full-parameter SFT
-> verl GRPO-LoRA warm-started from the exported SFT model
```

Pure GRPO from the original base model is no longer the main workflow. The existing verifier, reward, data preparation, and evaluation code remain in place because GRPO and evaluation still depend on them.

## Fixed Targets

- verl version: `v0.7.1`
- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT entry: `torchrun -m verl.trainer.sft_trainer`
- GRPO entry: `python -m verl.trainer.main_ppo`
- GRPO algorithm: `algorithm.adv_estimator=grpo`
- GRPO rollout backend: vLLM
- GRPO parameter-efficient training: LoRA

The scripts are pinned conceptually to `verl` v0.7.1 because the Hydra config surface and checkpoint layout can change between `verl` releases. Do not assume compatibility with `verl` v0.8.0 or the main branch before checking upstream config names on the server.

## Project Layout

```text
game24/
  prompt.py        Prompt builder shared by data, SFT, and evaluation
  verifier.py      Strict AST + Fraction verifier, no eval
  solver.py        Exact Fraction-based 24-point solver for SFT trajectories
  trajectory.py    Compact XML assistant responses from solver derivations
  reward.py        verl custom reward entry point
  reward_strict.py Strict 0/1 reward entry point for improved GRPO
  strict_dapo_reward_manager.py  DAPO-style strict group diagnostics
  metrics.py       Evaluation metric aggregation helpers
scripts/
  prepare_data.py     Server-side RL parquet generation
  build_sft_data.py   Exact-solver SFT parquet generation from train IDs only
  audit_game24_boundaries.py  Canonical split leakage audit
  run_sft.sh          Full-parameter SFT launch script for verl v0.7.1
  run_grpo_lora.sh    GRPO-LoRA launch script starting from exported SFT model
  run_grpo_lora_improved.sh  Improved strict-reward LoRA GRPO launch script
  run_grpo_full_param.sh  Full-parameter GRPO launch script starting from exported SFT model
  run_grpo_full.sh    Compatibility wrapper for run_grpo_full_param.sh
  evaluate_single_model.py  Strict val/test evaluation for one full-parameter checkpoint
  run_single_model_evaluation.sh  Shell entry for single-model strict evaluation
  evaluate.py         Server-side vLLM evaluation entry
  final_evaluation.py Strict raw/SFT/GRPO final evaluation and checkpoint selection
  run_final_evaluation.sh  Shell entry for final evaluation
configs/
  default.env.example
  evaluation/single_model/default.yaml
```

## Task Definition

The model receives four integers and must produce:

```xml
<think>
free-form reasoning
</think>
<answer>
final expression
</answer>
```

Only the expression inside `<answer>` is scored by the verifier. The verifier requires exactly one answer tag, accepts only integer constants, `+`, `-`, `*`, `/`, and parentheses, checks that the leaf integer multiset exactly matches the four input numbers, evaluates with `fractions.Fraction`, and requires the result to equal exactly `24`.

## Recommended Flow

1. Generate leak-free train/val/test/ToT-100 data:

```bash
python scripts/prepare_data.py --output-dir data/game24
```

2. Build exact-solver SFT trajectories using only `train.parquet` problem IDs:

```bash
python scripts/build_sft_data.py \
  --processed-data-dir data/game24 \
  --output-dir data/game24-sft
```

3. Run one epoch of full-parameter SFT:

```bash
bash scripts/run_sft.sh
```

4. Export a Hugging Face-loadable SFT model. If `checkpoint.save_contents=["model","optimizer","extra","hf_model"]` produces a usable HF directory in your server setup, use that. Otherwise use the official verl merger, for example:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir outputs/game24-sft-full/global_step_*/actor \
  --target_dir outputs/game24-sft-hf
```

5. Evaluate the SFT model Pass@1 and Pass@8 before RL:

```bash
python scripts/evaluate.py \
  --model outputs/game24-sft-hf \
  --data-file data/game24 \
  --n 8 \
  --temperature 0.7 \
  --top-p 0.95
```

6. Point GRPO to the exported SFT model:

```bash
GRPO_MODEL_PATH=outputs/game24-sft-hf bash scripts/run_grpo_lora.sh
```

7. Run GRPO-LoRA with rollout `n=16`.

8. Select the best checkpoint by validation greedy Pass@1 strict exact accuracy.

9. Report final results on ordinary `test.parquet`, `tot_hard100.parquet`, and `unsolvable.parquet`.

## Why SFT First

SFT changes the model and is used here as a GRPO warm start. The SFT stage is intentionally short, defaulting to one epoch, to teach stable expression format and exact arithmetic patterns while reducing overfitting and preserving later exploration. Multiple exact solutions, input digit permutations, and prompt templates keep the SFT data diverse without using validation, test, or ToT-100 answers.

The final objective is highest 24-point correctness. This project does not maintain a pure-GRPO control route as a first-class workflow.

## Data Preparation

`scripts/prepare_data.py` loads `nlile/24-game` and `test-time-compute/game-of-24`, adapts likely field names, deduplicates by `tuple(sorted(numbers))`, reserves the ToT fixed index range `[900, 1000)`, first reserves an ordinary non-overlapping test split from the remaining ToT records, then splits train/validation from the remaining `nlile/24-game` solvable IDs with a fixed seed. If the upstream datasets do not include unsolvable puzzles, the script fills `unsolvable.parquet` by enumerating classic 1-13 four-card combinations and checking them with the exact solver. It writes:

```text
train.parquet
val.parquet
test.parquet
tot_hard100.parquet
unsolvable.parquet
dataset_stats.json
```

No reference answer is added to the RL prompt.

The defaults are sized for the classic 1362 solvable 24-point combinations:

```text
--val-size 128
--test-size 256
```

## SFT Data

`scripts/build_sft_data.py` reads the prepared parquet files and uses only IDs from `train.parquet`. It explicitly checks:

```text
SFT train IDs intersect project val IDs = empty
SFT train IDs intersect project test IDs = empty
SFT train IDs intersect ToT-100 IDs = empty
```

It then cuts about 5% of original train problem IDs into `sft_val.parquet`, solves each remaining problem exactly, keeps up to 8 diverse solutions by default, uses up to 4 input permutations and up to 3 prompt templates, and writes:

```text
sft_train.parquet
sft_val.parquet
sft_stats.json
```

Default record count is roughly:

```text
solved SFT train problem IDs * up to 8 solutions * up to 2 prompt/permutation variants
```

For the classic 24-point problem scale, this is intended to land around 10k-30k high-quality samples, depending on how many unique train IDs are actually solvable and how many diverse exact solutions the solver finds.

Trajectory text keeps remaining-value lists compact, such as `5/6`, but wraps non-integer fractions when they are operation operands, such as `20 / (5/6)` or `(6/5) * 20`. Before writing each SFT row, the generator checks every structured solver step with `Fraction` arithmetic and skips any inconsistent candidate while counting it in `sft_stats.json`.

## SFT Configuration

`scripts/run_sft.sh` defaults to full-parameter SFT, not LoRA:

- `MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct`
- `N_GPUS=1`
- `TRAIN_BATCH_SIZE=32`
- `MICRO_BATCH_SIZE=4`
- `MAX_LENGTH=512`
- `LEARNING_RATE=5e-6`
- `TOTAL_EPOCHS=1`
- `SAVE_FREQ=-1`
- `TEST_FREQ=-1`
- `DTYPE=bfloat16`
- `SFT_ATTENTION_IMPLEMENTATION=sdpa`
- `SFT_USE_REMOVE_PADDING=False`

The script sets `data.messages_key=messages`, FSDP, BF16, gradient checkpointing, and `checkpoint.save_contents=["model","optimizer","extra","hf_model"]`. It also prints the exact Hydra overrides array before launching `torchrun`; the printed `data.train_files`, `data.val_files`, `++model.override_config.attn_implementation`, and `model.use_remove_padding` lines are the values actually passed to Hydra.

`SFT_ATTENTION_IMPLEMENTATION=sdpa` emits `++model.override_config.attn_implementation=sdpa`. The per-key `++` is intentional: in verl v0.7.1, `model.override_config` is an existing structured Hydra dict, but `attn_implementation` is a dynamic Hugging Face config key that may not already exist inside it. Plain `model.override_config.attn_implementation=sdpa` fails Hydra struct validation when the nested key is absent. The narrow `++` adds or updates only that one key and does not disable structure checks elsewhere. Set `SFT_ATTENTION_IMPLEMENTATION=` to omit the attention override entirely.

The default `sdpa` attention backend and `SFT_USE_REMOVE_PADDING=False` avoid requiring `flash_attn` for smoke tests and default server runs. `data.use_dynamic_bsz=True` and `data.pad_mode=no_padding` remain part of the SFT data pipeline configuration; they are not vLLM settings. Set `SFT_ATTENTION_IMPLEMENTATION=flash_attention_2 SFT_USE_REMOVE_PADDING=True` only on environments where FlashAttention2 has been installed and separately verified. The script never installs FlashAttention2 automatically. A `torch_dtype` deprecation warning from Transformers is not the root cause of the FlashAttention/Hydra failures described here.

For smoke runs, set `SFT_TRAIN_FILE` and `SFT_VAL_FILE` to the small parquet paths before launching. The script uses those same variables in both the configuration printout and the Hydra overrides, so verify the log shows the smoke paths and that verl reports smoke-scale `dataset len:` values before trusting a short run.

## GRPO After SFT

`scripts/run_grpo_lora.sh` requires:

```bash
GRPO_MODEL_PATH=/path/to/exported-sft-hf-model
```

It does not search checkpoints and does not default back to `Qwen/Qwen2.5-1.5B-Instruct`.

Default GRPO-LoRA settings:

- `GRPO_PROJECT_NAME=game24-grpo`
- `GRPO_EXPERIMENT_NAME=game24-grpo-smoke`
- `GRPO_LOGGER=console`
- `GRPO_TRAIN_BATCH_SIZE=4`
- `GRPO_ROLLOUT_N=4`
- `GRPO_PPO_MINI_BATCH_SIZE=4`
- `GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=1`
- `GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1`
- `GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1`
- `GRPO_MAX_PROMPT_LENGTH=192`
- `GRPO_MAX_RESPONSE_LENGTH=256`
- `GRPO_LORA_RANK=64`
- `GRPO_LORA_ALPHA=64`
- `GRPO_LEARNING_RATE=1e-6`
- `GRPO_TOTAL_TRAINING_STEPS=5`
- `GRPO_TEMPERATURE=1.0`
- `GRPO_TOP_P=0.95`
- `GRPO_ATTN_IMPLEMENTATION=sdpa`
- `GRPO_USE_REMOVE_PADDING=false`
- `GRPO_SAVE_FREQ=50`
- `GRPO_TEST_FREQ=25`
- `GRPO_GPU_MEMORY_UTILIZATION=0.45`
- `GRPO_N_GPUS=1`

The GRPO script still uses vLLM rollout, KL loss, checkpoint saving, console logging by default, and the custom reward function `game24/reward.py:compute_score`. GRPO-specific environment variables take priority and are used by default so that sourced SFT settings such as `PROJECT_NAME`, `EXPERIMENT_NAME`, or `LOGGER` do not leak into GRPO runs.

For smoke runs, keep all per-GPU micro-batch settings at `1`. After the job is stable, increase `GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU`, `GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`, and `GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU` gradually to improve throughput. Do not rely on the verl default `actor_rollout_ref.actor.ppo_mini_batch_size`; set `GRPO_PPO_MINI_BATCH_SIZE` explicitly, keep it positive, ensure it is no larger than `GRPO_TRAIN_BATCH_SIZE`, and ensure `GRPO_TRAIN_BATCH_SIZE` is divisible by it.

GRPO defaults `GRPO_ATTN_IMPLEMENTATION=sdpa` and `GRPO_USE_REMOVE_PADDING=false` so the FSDP Actor/Reference Transformers path does not require `flash-attn` during smoke runs. The script passes the attention setting with Hydra's narrow dynamic-key syntax, `++actor_rollout_ref.model.override_config.attn_implementation=...`, because `attn_implementation` is not present in verl v0.7.1's structured `override_config` by default. This setting is for the Transformers Actor/Reference model path only; `actor_rollout_ref.rollout.name=vllm` remains unchanged and no vLLM attention backend override is added.

A800 runs can separately test `GRPO_ATTN_IMPLEMENTATION=flash_attention_2 GRPO_USE_REMOVE_PADDING=true` after FlashAttention2 is installed and verified, but it is not required for the current smoke path. Always check the GRPO smoke log for `GRPO_ATTN_IMPLEMENTATION=sdpa` and `GRPO_USE_REMOVE_PADDING=false` in the printed launch summary before debugging deeper worker errors.

## Improved LoRA GRPO

`scripts/run_grpo_lora_improved.sh` is an isolated LoRA GRPO experiment intended to improve strict greedy Pass@1. It does not modify or replace the original `scripts/run_grpo_lora.sh` baseline, and it uses only `IMPROVED_GRPO_*` variables. It starts fresh from the exported SFT Hugging Face model and does not resume old GRPO checkpoints or load old GRPO LoRA adapters.

The main differences from the original LoRA baseline are:

- strict 0/1 training reward in `game24/reward_strict.py`;
- DAPO-style reward manager `game24/strict_dapo_reward_manager.py` for strict `acc` and group diagnostics;
- `IMPROVED_GRPO_ROLLOUT_N=16`;
- `IMPROVED_GRPO_GEN_BATCH_SIZE=24` candidate prompts versus `IMPROVED_GRPO_TRAIN_BATCH_SIZE=8` target effective prompts;
- actor KL loss disabled and KL coefficient set to `0`;
- PPO epochs increased to `2`;
- asymmetric clipping through `clip_ratio_low=0.20`, `clip_ratio_high=0.28`, and `clip_ratio_c=10.0`;
- `loss_agg_mode=seq-mean-token-mean`;
- validation every 10 steps with greedy `do_sample=False`, `n=1`.

The strict reward returns `score=1.0` only when `verify_solution()` reports `is_correct=True`; every other response gets `score=0.0`. Diagnostic fields such as `format_valid`, `parse_valid`, `number_usage_valid`, `response_length`, and strict `acc` are logged separately and do not create shaped reward.

The reward manager groups rollouts by verl `uid` when present, otherwise by `tuple(sorted(numbers))` and target. It records:

```text
k_correct
k_correct_hist_0 ... k_correct_hist_16
all_wrong_rate
all_correct_rate
mixed_group_rate
generated_prompt_count
accepted_prompt_count
acceptance_rate
generation_rounds
zero_reward_std_rate
response_length
```

It also prints batch-level strict metrics in the log, including `strict_exact=correct/total`, strict accuracy, format rate, parse rate, number usage rate, mean response length, and mixed group rate. For a 64-question validation split this gives raw counts such as `12/64`, not only percentages.

In verl v0.7.1, the trainer reads `data.gen_batch_size` even though it is not declared in `legacy_data.yaml`, so the script passes it with Hydra's dynamic-field syntax: `++data.gen_batch_size=24`. The script does not invent higher-version `algorithm.filter_groups.*` keys. All-wrong and all-correct groups have zero reward standard deviation under strict 0/1 GRPO, so they produce no policy-gradient signal when actor KL loss is disabled; the group diagnostics make that visible in logs. If a future server-side verl v0.7.1 build exposes explicit group-filtering keys, add them only after confirming that exact build's config surface.

Before launch, `scripts/audit_game24_boundaries.py` checks canonical IDs using `tuple(sorted(numbers))` and fails on any overlap between:

```text
train vs val
train vs test
train vs hard100
SFT train vs project val
SFT train vs project test
SFT train vs hard100
```

`test`, `tot_hard100`, and `unsolvable` are not used for training, rollout generation, dynamic candidate generation, or checkpoint selection. `test` and `hard100` are read only by the launch-time leakage audit. Hyperparameters and checkpoint selection should be adjusted only from project validation behavior; test and hard100 remain frozen.

Default improved settings:

- `IMPROVED_GRPO_TRAIN_BATCH_SIZE=8`
- `IMPROVED_GRPO_GEN_BATCH_SIZE=24`
- `IMPROVED_GRPO_MAX_GEN_BATCHES=4`
- `IMPROVED_GRPO_ROLLOUT_N=16`
- `IMPROVED_GRPO_MAX_PROMPT_LENGTH=192`
- `IMPROVED_GRPO_MAX_RESPONSE_LENGTH=192`
- `IMPROVED_GRPO_LORA_RANK=64`
- `IMPROVED_GRPO_LORA_ALPHA=64`
- `IMPROVED_GRPO_TARGET_MODULES=all-linear`
- `IMPROVED_GRPO_PPO_MINI_BATCH_SIZE=4`
- `IMPROVED_GRPO_PPO_EPOCHS=2`
- `IMPROVED_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=32`
- `IMPROVED_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=64`
- `IMPROVED_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=64`
- `IMPROVED_GRPO_LEARNING_RATE=5e-6`
- `IMPROVED_GRPO_TEMPERATURE=1.0`
- `IMPROVED_GRPO_TOP_P=1.0`
- `IMPROVED_GRPO_TOP_K=-1`
- `IMPROVED_GRPO_TOTAL_TRAINING_STEPS=200`
- `IMPROVED_GRPO_TEST_FREQ=10`
- `IMPROVED_GRPO_SAVE_FREQ=20`

`lora_dropout` and explicit gradient clipping are printed in the launch summary but not passed as Hydra overrides because the checked verl v0.7.1 model/actor configs do not expose verified keys for them. This avoids silently creating unused fields.

Start the formal improved LoRA run:

```bash
IMPROVED_GRPO_MODEL_PATH=/root/autodl-tmp/outputs/game24-sft-full/global_step_363/huggingface \
bash scripts/run_grpo_lora_improved.sh
```

## Full-Parameter GRPO

`scripts/run_grpo_full_param.sh` is the formal full-parameter GRPO path for verl v0.7.1 after SFT. It is separate from `scripts/run_grpo_lora.sh` and uses only `FULL_GRPO_*` environment variables. `scripts/run_grpo_full.sh` remains as a compatibility wrapper that calls the same implementation.

Default SFT warm start:

```bash
FULL_GRPO_MODEL_PATH=/root/autodl-tmp/outputs/game24-sft-full/global_step_363/huggingface
```

Default formal training settings:

- `FULL_GRPO_TRAIN_BATCH_SIZE=8`
- `FULL_GRPO_ROLLOUT_N=8`
- `FULL_GRPO_MAX_PROMPT_LENGTH=192`
- `FULL_GRPO_MAX_RESPONSE_LENGTH=192`
- `FULL_GRPO_PPO_MINI_BATCH_SIZE=8`
- `FULL_GRPO_PPO_EPOCHS=1`
- `FULL_GRPO_PPO_MICRO_BATCH_SIZE_PER_GPU=4`
- `FULL_GRPO_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16`
- `FULL_GRPO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=16`
- `FULL_GRPO_LEARNING_RATE=3e-7`
- `FULL_GRPO_TOTAL_TRAINING_STEPS=400`
- `FULL_GRPO_SAVE_FREQ=25`
- `FULL_GRPO_VAL_FREQ=25`
- `FULL_GRPO_MAX_ACTOR_CKPT_TO_KEEP=1`
- `FULL_GRPO_VAL_BEFORE_TRAIN=true`
- `FULL_GRPO_ATTN_IMPLEMENTATION=sdpa`
- `FULL_GRPO_USE_REMOVE_PADDING=false`
- `FULL_GRPO_GPU_MEMORY_UTILIZATION=0.35`
- `FULL_GRPO_USE_KL_LOSS=true`
- `FULL_GRPO_KL_LOSS_COEF=0.001`

LoRA is disabled with verl's `actor_rollout_ref.model.lora_rank=0`; the script does not pass `lora_alpha`, `target_modules`, or a LoRA adapter path. The launch summary prints that LoRA is disabled, full-parameter training is enabled, and the trainable and total parameter counts are equal.

Training uses `FULL_GRPO_TRAIN_FILE=data/game24/train.parquet` and internal validation uses `FULL_GRPO_VAL_FILE=data/game24/val.parquet`. verl runs validation before training and every 25 steps through `trainer.val_before_train=true` and `trainer.test_freq=25`; the project-level environment variable is named `FULL_GRPO_VAL_FREQ` to avoid confusing this with final held-out test evaluation. The script does not run `test.parquet`, ToT hard-100, unsolvable evaluation, Pass@8, or `scripts/run_final_evaluation.sh`.

The default output directory is timestamped so it does not overwrite earlier experiments:

```text
/root/autodl-tmp/outputs/game24-grpo-full-param-<timestamp>/global_step_xxx/
```

Every 25 steps, the actor checkpoint is configured with:

```text
actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]
trainer.max_actor_ckpt_to_keep=1
```

This preserves the latest full actor state, optimizer state, extra/RNG state, and a Hugging Face model artifact when supported by the local verl setup. In verl v0.7.1 the actor retention manager saves the new `global_step_N/actor` first, then removes older tracked actor paths with `shutil.rmtree`. That means only the newest checkpoint remains complete; older `global_step_*` directories may retain small trainer files such as `data.pt`, but the old actor model and optimizer `.pt` files should not accumulate. Keeping only one checkpoint saves disk, but it also means you cannot roll back to an earlier validation-best weight.

To retain two complete actor checkpoints instead:

```bash
FULL_GRPO_MAX_ACTOR_CKPT_TO_KEEP=2 bash scripts/run_grpo_full_param.sh
```

The launch script prints `df -h /root/autodl-tmp`, `du -sh "$FULL_GRPO_OUTPUT_DIR"`, the checkpoint save frequency, and the final remaining `global_step_*` directories after a normal training exit.

The latest checkpoint can be used for interrupted-training resume by explicitly setting:

```bash
FULL_GRPO_RESUME_MODE=resume_path \
FULL_GRPO_RESUME_FROM_PATH=/root/autodl-tmp/outputs/game24-grpo-full-param-<timestamp>/global_step_xxx \
bash scripts/run_grpo_full_param.sh
```

If the server only writes the sharded FSDP actor state, use verl's merger on the saved actor directory:

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /root/autodl-tmp/outputs/game24-grpo-full-param-<timestamp>/global_step_xxx/actor \
  --target_dir /root/autodl-tmp/outputs/game24-grpo-full-param-<timestamp>/global_step_xxx/huggingface
```

Start the formal 400-step full-parameter run:

```bash
FULL_GRPO_LOG_FILE=/root/autodl-tmp/logs/game24-grpo-full-param-$(date +%Y%m%d_%H%M%S).log \
bash scripts/run_grpo_full_param.sh
```

## Single-Model Strict Evaluation

Use this entry when you want to evaluate one full-parameter GRPO checkpoint directly from a training run directory. It does not require a manually exported Hugging Face model, does not use `SFT_MODEL_PATH` or `GRPO_RUN_DIR`, does not load LoRA adapters, and does not run multi-checkpoint selection.

Default config:

```text
configs/evaluation/single_model/default.yaml
```

Measure the latest full-parameter run:

```bash
bash scripts/run_single_model_evaluation.sh
```

By default, the evaluator searches `/root/autodl-tmp/outputs` for the newest recoverable run matching:

```text
game24-grpo-full-param-continue-*
game24-grpo-full-param-*
```

It skips evaluation/SFT/LoRA metadata directories, selects the highest numeric `global_step_*` with a recoverable actor checkpoint, then runs greedy strict Pass@1 on `val.parquet` and `test.parquet`.

Checkpoint handling is automatic:

```text
actor/huggingface_merged with config + weights -> load directly
actor/huggingface with config + weights -> load directly
raw verl FSDP actor shards -> run python -m verl.model_merger merge automatically
actor/huggingface with only tokenizer/config -> not treated as a complete model
```

For raw FSDP checkpoints, the merge source is `global_step_N/actor` and the cached target is:

```text
global_step_N/actor/huggingface_merged
```

The evaluator uses the official verl v0.7.1 merger command:

```text
python -m verl.model_merger merge --backend fsdp --local_dir <actor> --target_dir <actor>/huggingface_merged
```

There is no separate direct-FSDP Transformers inference path in this evaluator. The stable path is automatic merge to Hugging Face format, then one model load for val and test. The merged model is cached by default and reused on later runs; original checkpoints, SFT models, and other experiments are not deleted.

Dry-run shows the resolved run/checkpoint/format without loading a GPU model or merging:

```bash
bash scripts/run_single_model_evaluation.sh --dry-run
```

Evaluate a specific training directory by copying the config:

```bash
cp configs/evaluation/single_model/default.yaml configs/evaluation/single_model/full_param_step500.yaml
```

Then edit:

```yaml
model:
  source: run_name
  run_name: game24-grpo-full-param-continue-400-to-500-具体时间
  step: 500
```

Run it:

```bash
bash scripts/run_single_model_evaluation.sh \
  --config configs/evaluation/single_model/full_param_step500.yaml
```

Outputs are written under:

```text
/root/autodl-tmp/outputs/single-model-evaluation/<run-name>/step_<N>/<timestamp>/
```

The directory contains `results.json`, split summaries, full prediction JSONL files, `prompt_audit.json`, `resolved_config.yaml`, and `evaluation.log`. Strict exact accuracy is computed only from `verification.is_correct` in `game24/verifier.py`; `reward_mean` is reported separately and is not accuracy.

## Final Evaluation

Use the final evaluation entry after GRPO training has produced LoRA checkpoints. The default mode is a fast diagnostic pass:

```text
EVAL_MODE=quick
-> evaluate every GRPO global_step_xxx checkpoint on val with greedy Pass@1
-> choose the highest strict exact accuracy, tie-breaking to the earlier step
-> evaluate SFT and the selected GRPO LoRA on val and test with greedy Pass@1
```

The strict exact accuracy is `VerificationResult.is_correct` from `game24/verifier.py`: exactly one `<answer>` tag, a parsable AST-whitelisted arithmetic expression, exact input number multiset usage, and exact `Fraction` value equal to 24. `reward_mean` is reported separately and must not be read as accuracy because the reward function contains partial rewards.

The quick mode intentionally skips raw Qwen, Pass@8, hard100, and unsolvable so the first server check can answer whether SFT and GRPO use the same prompt path and whether strict validation behaves as expected. It writes only the first `EVAL_DIAGNOSTIC_LIMIT=20` prediction rows per model/dataset while still computing metrics on the full split.

For the full report, set:

```bash
EVAL_MODE=full
```

Full mode keeps the val-only GRPO checkpoint selection, then evaluates raw Qwen, SFT, and the selected GRPO LoRA on `test`, `tot_hard100`, and `unsolvable` with greedy Pass@1 and sampling Pass@8.

GRPO checkpoints are loaded as PEFT adapters:

```text
base model = exported SFT Hugging Face directory
adapter = global_step_xxx/actor/lora_adapter
```

Do not treat `actor/huggingface` as a complete standalone GRPO model unless a separate export step has explicitly produced one.

Checkpoint selection loads the SFT base model once, then loads/switches one PEFT LoRA adapter per `global_step_xxx` using a single active adapter name. This avoids repeatedly reading the full SFT model from disk and prevents adapter stacking.

Generation is batched. `EVAL_BATCH_SIZE=32` means one `model.generate()` call per batch of prompts, and Pass@8 uses `num_return_sequences=8` in that same batched generation call rather than looping eight times in Python. Each batch is verified and written to JSONL immediately, with progress printed after every batch.

Run the quick evaluation on the server:

```bash
EVAL_MODE=quick \
RAW_MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct \
SFT_MODEL_PATH=/root/autodl-tmp/outputs/game24-sft-full/global_step_363/huggingface \
GRPO_RUN_DIR=/root/autodl-tmp/outputs/YOUR_GRPO_RUN_DIR \
EVAL_DATA_DIR=data/game24 \
EVAL_OUTPUT_DIR=outputs/evaluation \
EVAL_BATCH_SIZE=32 \
EVAL_MAX_NEW_TOKENS=192 \
bash scripts/run_final_evaluation.sh
```

The script discovers parquet files under `EVAL_DATA_DIR` instead of hard-coding filenames. Quick mode requires `train`, `val`, and `test`; full mode additionally requires `hard100` and `unsolvable`.

Quick mode writes:

```text
outputs/evaluation/checkpoint_selection.csv
outputs/evaluation/quick_results.json
outputs/evaluation/prompt_audit.json
outputs/evaluation/predictions/sft_val.jsonl
outputs/evaluation/predictions/grpo_val.jsonl
outputs/evaluation/predictions/sft_test.jsonl
outputs/evaluation/predictions/grpo_test.jsonl
```

Full mode writes:

```text
outputs/evaluation/checkpoint_selection.csv
outputs/evaluation/final_results.json
outputs/evaluation/prompt_audit.json
outputs/evaluation/predictions/*.jsonl
```

`checkpoint_selection.csv` includes `step`, `exact_correct`, `total`, `exact_accuracy`, `format_rate`, `number_usage_rate`, `parse_rate`, and `reward_mean`. Final summaries include greedy Pass@1 and sampling Pass@8 metrics for each model and dataset. For Pass@8, the reported solved rate is the fraction of problems with at least one strictly correct sample; the average correct sample count is reported separately.

Every prediction JSONL row stores the model name, dataset name, problem ID, input numbers, full generated text, extracted answer, strict verification dictionary, failure reason, and reward.

`prompt_audit.json` reads one row from the training parquet and records the original training messages, the rendered training prompt, the rendered evaluation prompt, token IDs for both, whether the token IDs are identical, the tokenizer chat template, EOS token, padding side, input number order, and `EVAL_MAX_NEW_TOKENS`. A mismatch fails the run because evaluation must use the training parquet prompt instead of inventing a different prompt.

## Single-Model Evaluation

Evaluate one parquet file:

```bash
python scripts/evaluate.py \
  --model outputs/game24-sft-hf \
  --data-file data/game24/test.parquet \
  --n 1 \
  --temperature 0 \
  --output-jsonl outputs/eval/test.jsonl \
  --summary-json outputs/eval/test_summary.json
```

Evaluate `test`, `tot_hard100`, and `unsolvable` together by passing the directory:

```bash
python scripts/evaluate.py \
  --model outputs/game24-sft-hf \
  --data-file data/game24 \
  --n 8 \
  --temperature 0.7 \
  --top-p 0.95
```

Metrics include greedy Pass@1, sampling Pass@N, strict exact accuracy, format valid rate, parse valid rate, correct number multiset rate, average response length, and response truncation rate. For `unsolvable.parquet`, the script reports failed-answer hallucination rate instead of treating accuracy as the main score.

## Server Validation Checklist

This repository was authored locally only. No tests, data downloads, SFT data generation, training, inference, GPU checks, or dependency installation were run in this Ubuntu workspace.

Verify these items on the GPU server:

- Exact `verl==0.7.1` install and both trainer entries.
- SFT Hydra keys: `data.messages_key`, FSDP `engine.*`, BF16 dtype, checkpoint `save_contents`, `++model.override_config.attn_implementation`, and `model.use_remove_padding`.
- SFT smoke logs: printed `SFT_TRAIN_FILE`/`SFT_VAL_FILE`, printed Hydra `data.train_files`/`data.val_files`, and verl `dataset len:` values must all indicate the intended small files.
- SFT attention logs: printed `ATTENTION_IMPLEMENTATION` and `USE_REMOVE_PADDING` must match the Hydra overrides, Hydra must not fail with a struct error, and model loading must not request missing FlashAttention2 when using `sdpa`.
- Whether `hf_model` checkpoint content produces a directly loadable SFT directory in v0.7.1.
- If needed, the official `python -m verl.model_merger merge --backend fsdp` path and expected actor checkpoint directory.
- PPO/GRPO Hydra keys for LoRA, vLLM rollout, rollout temperature/top-p, custom reward path/name, logger, checkpoint, KL loss, and whether a separate Transformers actor/ref attention override is needed.
- Actual field names and sizes in `nlile/24-game` and `test-time-compute/game-of-24`.
- Dataset leakage checks printed by `prepare_data.py` and `build_sft_data.py`.
- WandB environment and model access credentials.
- Single 80GB GPU memory fit for one-epoch SFT and rollout `n=16` GRPO.
- For tiny SFT smoke runs, keep the final printed Hydra overrides with the log, run only a few training steps, and inspect loss, peak memory, and checkpoint output before starting a full run.

Suggested first server command:

```bash
python scripts/prepare_data.py --output-dir data/game24
```
