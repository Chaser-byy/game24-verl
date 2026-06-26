# game24-verl

`game24-verl` is a fresh, standalone Game of 24 task layer for external `verl` v0.7.1 training. It does not copy, migrate, or depend on any older TRL or GRPO project.

The project only contains task-specific code: prompt construction, strict answer verification, a custom reward function, data preparation, training launch configuration, evaluation, and documentation. It does not vendor or modify `verl`, vLLM, Ray, FSDP, KL loss, checkpointing, or GRPO internals.

## Fixed Training Target

- verl version: `v0.7.1`
- training entry: `python -m verl.trainer.main_ppo`
- algorithm: GRPO via `algorithm.adv_estimator=grpo`
- model: `Qwen/Qwen2.5-1.5B-Instruct`
- parameter-efficient training: LoRA
- rollout backend: vLLM

The scripts are pinned conceptually to `verl` v0.7.1 because the Hydra config surface can change between `verl` releases. Do not assume this project is compatible with `verl` v0.8.0 or the main branch before checking the upstream config names on the server.

## Project Layout

```text
game24/
  prompt.py        Prompt builder shared by training data and evaluation
  verifier.py      Strict AST + Fraction verifier, no eval
  reward.py        verl custom reward entry point
  metrics.py       Evaluation metric aggregation helpers
scripts/
  prepare_data.py  Server-side parquet generation
  run_grpo_lora.sh GRPO LoRA launch script for external verl v0.7.1
  evaluate.py      Server-side vLLM evaluation entry
configs/
  default.env.example
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

Only the expression inside `<answer>` is verified. The verifier requires exactly one answer tag, accepts only integer constants, `+`, `-`, `*`, `/`, and parentheses, checks that the leaf integer multiset exactly matches the four input numbers, evaluates with `fractions.Fraction`, and requires the result to equal exactly `24`.

## Data Flow

On the GPU/server environment, install the project dependencies and run:

```bash
python scripts/prepare_data.py --output-dir data/game24
```

The script loads `nlile/24-game` and `test-time-compute/game-of-24`, adapts likely field names, deduplicates by `tuple(sorted(numbers))`, reserves the ToT fixed index range `[900, 1000)`, removes those IDs from training candidates, splits train/validation with a fixed seed, builds an ordinary test split from non-overlapping ToT records, and writes:

```text
train.parquet
val.parquet
test.parquet
tot_hard100.parquet
unsolvable.parquet
dataset_stats.json
```

No reference answer is added to the prompt.

## Training Flow

Copy `configs/default.env.example` to your server environment file if useful, edit paths as needed, then launch:

```bash
bash scripts/run_grpo_lora.sh
```

The script prints its final configuration and calls external `verl`:

```bash
python -m verl.trainer.main_ppo
```

It sets GRPO, LoRA rank/alpha, `target_modules=all-linear`, vLLM rollout with `n=8`, BF16 FSDP/rollout dtype, gradient checkpointing, KL loss, checkpoint save/test frequency, console and WandB logging, single-node GPU settings, and the custom reward function `game24/reward.py:compute_score`.

## Evaluation

Evaluate one parquet file:

```bash
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-file data/game24/test.parquet \
  --n 1 \
  --temperature 0 \
  --output-jsonl outputs/eval/test.jsonl \
  --summary-json outputs/eval/test_summary.json
```

Evaluate the standard `test`, `tot_hard100`, and `unsolvable` files together by passing the directory:

```bash
python scripts/evaluate.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/game24-grpo-lora/global_step_200/actor \
  --data-file data/game24 \
  --n 8 \
  --temperature 0.7 \
  --top-p 0.95
```

Metrics include greedy Pass@1, sampling Pass@N, strict exact accuracy, format valid rate, parse valid rate, correct number multiset rate, average response length, and response truncation rate. For `unsolvable.parquet`, the script reports failed-answer hallucination rate instead of treating accuracy as the main score.

## Default Training Configuration

- `TRAIN_BATCH_SIZE=32`
- `ROLLOUT_N=8`
- `MAX_PROMPT_LENGTH=192`
- `MAX_RESPONSE_LENGTH=256`
- `LORA_RANK=64`
- `LORA_ALPHA=64`
- `LEARNING_RATE=2e-6`
- `TOTAL_EPOCHS=8`
- `SAVE_FREQ=50`
- `TEST_FREQ=25`
- `GPU_MEMORY_UTILIZATION=0.45`
- `N_GPUS=1`
- `DTYPE=bfloat16`

## Server Validation Checklist

This repository was authored locally only. It has not run tests, data downloads, training, inference, GPU checks, or dependency installation in this Ubuntu workspace.

Verify these items on the GPU server:

- Exact `verl==0.7.1` install and `python -m verl.trainer.main_ppo` availability.
- Hydra override names for LoRA, vLLM rollout, BF16 dtype, custom reward path/name, logger, checkpoint, and KL loss.
- vLLM LoRA adapter loading behavior used by `scripts/evaluate.py`.
- Actual field names in `nlile/24-game` and `test-time-compute/game-of-24`.
- Dataset sizes, especially `test-time-compute/game-of-24` rows `[900, 1000)`.
- WandB environment and model access credentials.
- Single 80GB GPU memory fit with the default vLLM memory utilization.

Suggested first server command:

```bash
python scripts/prepare_data.py --output-dir data/game24
```

Recommended training progression:

1. Generate data.
2. Evaluate the base model.
3. Run a 10-step smoke test with `TOTAL_TRAINING_STEPS=10`.
4. Run a 50-step small training job.
5. Run a 200-300 step training job after checking metrics and samples.
