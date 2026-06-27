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
  metrics.py       Evaluation metric aggregation helpers
scripts/
  prepare_data.py     Server-side RL parquet generation
  build_sft_data.py   Exact-solver SFT parquet generation from train IDs only
  run_sft.sh          Full-parameter SFT launch script for verl v0.7.1
  run_grpo_lora.sh    GRPO-LoRA launch script starting from exported SFT model
  evaluate.py         Server-side vLLM evaluation entry
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
MODEL_PATH=outputs/game24-sft-hf bash scripts/run_grpo_lora.sh
```

7. Run GRPO-LoRA with rollout `n=16`.

8. Select the best checkpoint by validation greedy Pass@1.

9. Report final results on ordinary `test.parquet` and `tot_hard100.parquet`.

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

The script sets `data.messages_key=messages`, FSDP, BF16, gradient checkpointing, `model.override_config.attn_implementation=sdpa`, and `checkpoint.save_contents=["model","optimizer","extra","hf_model"]`. The default `sdpa` attention backend and `SFT_USE_REMOVE_PADDING=False` avoid requiring `flash_attn` for smoke tests and default server runs. Set `SFT_ATTENTION_IMPLEMENTATION=flash_attention_2 SFT_USE_REMOVE_PADDING=True` only on environments where FlashAttention2 is installed. The actual checkpoint directory layout must still be validated on the server.

## GRPO After SFT

`scripts/run_grpo_lora.sh` requires:

```bash
MODEL_PATH=/path/to/exported-sft-hf-model
```

It does not search checkpoints and does not default back to `Qwen/Qwen2.5-1.5B-Instruct`.

Default GRPO-LoRA settings:

- `TRAIN_BATCH_SIZE=16`
- `ROLLOUT_N=16`
- `MAX_PROMPT_LENGTH=192`
- `MAX_RESPONSE_LENGTH=256`
- `LORA_RANK=64`
- `LORA_ALPHA=64`
- `LEARNING_RATE=1e-6`
- `TOTAL_TRAINING_STEPS=400`
- `TEMPERATURE=1.0`
- `TOP_P=0.95`
- `SAVE_FREQ=50`
- `TEST_FREQ=25`
- `GPU_MEMORY_UTILIZATION=0.45`
- `N_GPUS=1`

The GRPO script still uses vLLM rollout, KL loss, checkpoint saving, console and WandB logging, and the custom reward function `game24/reward.py:compute_score`.

## Evaluation

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
- SFT Hydra keys: `data.messages_key`, FSDP `engine.*`, BF16 dtype, checkpoint `save_contents`, and `model.use_remove_padding`.
- Whether `hf_model` checkpoint content produces a directly loadable SFT directory in v0.7.1.
- If needed, the official `python -m verl.model_merger merge --backend fsdp` path and expected actor checkpoint directory.
- PPO/GRPO Hydra keys for LoRA, vLLM rollout, rollout temperature/top-p, custom reward path/name, logger, checkpoint, and KL loss.
- Actual field names and sizes in `nlile/24-game` and `test-time-compute/game-of-24`.
- Dataset leakage checks printed by `prepare_data.py` and `build_sft_data.py`.
- WandB environment and model access credentials.
- Single 80GB GPU memory fit for one-epoch SFT and rollout `n=16` GRPO.

Suggested first server command:

```bash
python scripts/prepare_data.py --output-dir data/game24
```
