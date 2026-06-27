#!/usr/bin/env python3
"""Strict final evaluation for raw, SFT, and GRPO-LoRA Game24 models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from game24.prompt import build_chat_prompt
from game24.reward import compute_score
from game24.verifier import verify_solution


@dataclass(frozen=True)
class Problem:
    problem_id: str
    numbers: list[int]
    target: int
    solvable: bool
    prompt_messages: list[dict[str, str]]


@dataclass(frozen=True)
class Checkpoint:
    step: int
    checkpoint_dir: Path
    adapter_dir: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    base_path: str
    adapter_path: str | None = None


def _decode_nested(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return _decode_nested(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, Mapping):
        return {key: _decode_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_nested(item) for item in value]
    return value


def _load_messages(value: Any) -> list[dict[str, str]] | None:
    value = _decode_nested(value)
    if value is None:
        return None
    if isinstance(value, list):
        messages: list[dict[str, str]] = []
        for item in value:
            if not isinstance(item, Mapping) or "role" not in item or "content" not in item:
                return None
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
        return messages
    return None


def _load_ground_truth(row: pd.Series) -> dict[str, Any]:
    reward_model = _decode_nested(row.get("reward_model"))
    if isinstance(reward_model, Mapping) and "ground_truth" in reward_model:
        ground_truth = _decode_nested(reward_model["ground_truth"])
        if isinstance(ground_truth, Mapping):
            return dict(ground_truth)

    extra_info = _decode_nested(row.get("extra_info"))
    if isinstance(extra_info, Mapping) and "numbers" in extra_info:
        return {
            "numbers": extra_info["numbers"],
            "target": extra_info.get("target", 24),
            "solvable": extra_info.get("solvable", True),
        }

    raise ValueError("row does not contain reward_model.ground_truth or extra_info.numbers")


def _problem_id_from_row(row: pd.Series, numbers: Sequence[int]) -> str:
    extra_info = _decode_nested(row.get("extra_info"))
    if isinstance(extra_info, Mapping) and extra_info.get("problem_id") is not None:
        return str(extra_info["problem_id"])
    return "_".join(str(number) for number in sorted(int(item) for item in numbers))


def _problem_from_row(row: pd.Series) -> Problem:
    truth = _load_ground_truth(row)
    numbers = [int(number) for number in _decode_nested(truth["numbers"])]
    target = int(truth.get("target", 24))
    solvable = bool(truth.get("solvable", True))
    messages = _load_messages(row.get("prompt")) or build_chat_prompt(numbers, target=target)

    return Problem(
        problem_id=_problem_id_from_row(row, numbers),
        numbers=numbers,
        target=target,
        solvable=solvable,
        prompt_messages=messages,
    )


def load_problems(path: Path) -> list[Problem]:
    frame = pd.read_parquet(path)
    return [_problem_from_row(row) for _, row in frame.iterrows()]


def _split_from_file(path: Path) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    if "unsolvable" in normalized:
        return "unsolvable"
    if "tot" in normalized or "hard100" in normalized or "hard_100" in normalized:
        return "hard100"
    if normalized in {"val", "valid", "validation"} or normalized.endswith("_val"):
        return "val"
    if normalized == "test" or normalized.endswith("_test"):
        return "test"
    return None


def _split_from_rows(path: Path) -> str | None:
    try:
        frame = pd.read_parquet(path, columns=["extra_info"])
    except Exception:
        return None
    for value in frame["extra_info"].head(20):
        extra_info = _decode_nested(value)
        if isinstance(extra_info, Mapping):
            split = str(extra_info.get("split", "")).lower()
            if split in {"val", "valid", "validation"}:
                return "val"
            if split == "test":
                return "test"
            if split in {"tot_hard100", "hard100", "tot"}:
                return "hard100"
            if split == "unsolvable":
                return "unsolvable"
    return None


def discover_data_files(data_dir: Path) -> dict[str, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"evaluation data directory does not exist: {data_dir}")

    discovered: dict[str, Path] = {}
    candidates = sorted(data_dir.rglob("*.parquet"))
    for path in candidates:
        split = _split_from_file(path) or _split_from_rows(path)
        if split is not None:
            discovered.setdefault(split, path)

    required = {"val", "test", "hard100", "unsolvable"}
    missing = sorted(required - set(discovered))
    if missing:
        available = [str(path) for path in candidates]
        raise FileNotFoundError(
            f"missing required evaluation split(s): {missing}; parquet candidates under {data_dir}: {available}"
        )
    return discovered


def discover_checkpoints(grpo_run_dir: Path) -> list[Checkpoint]:
    if not grpo_run_dir.exists():
        raise FileNotFoundError(f"GRPO run directory does not exist: {grpo_run_dir}")

    checkpoints: list[Checkpoint] = []
    for checkpoint_dir in sorted(grpo_run_dir.glob("global_step_*")):
        match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir.name)
        if match is None or not checkpoint_dir.is_dir():
            continue
        adapter_dir = checkpoint_dir / "actor" / "lora_adapter"
        if (adapter_dir / "adapter_model.safetensors").exists() and (adapter_dir / "adapter_config.json").exists():
            checkpoints.append(Checkpoint(int(match.group(1)), checkpoint_dir, adapter_dir))

    if not checkpoints:
        raise FileNotFoundError(
            f"no GRPO LoRA checkpoints found under {grpo_run_dir}; expected global_step_xxx/actor/lora_adapter"
        )
    return sorted(checkpoints, key=lambda item: item.step)


def _prompt_text(problem: Problem, tokenizer: Any) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(problem.prompt_messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{message['role']}: {message['content']}" for message in problem.prompt_messages) + "\nassistant:"


def load_model_and_tokenizer(spec: ModelSpec, torch_dtype: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(spec.base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.base_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if spec.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter_path)
    model.eval()
    return model, tokenizer


def unload_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def generate_responses(
    model: Any,
    tokenizer: Any,
    problems: Sequence[Problem],
    *,
    n: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> list[list[str]]:
    import torch
    from transformers import set_seed

    set_seed(seed)
    grouped: list[list[str]] = [[] for _ in problems]
    prompts = [_prompt_text(problem, tokenizer) for problem in problems]

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": n,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
        for batch_index in range(len(batch_prompts)):
            item_start = batch_index * n
            grouped[start + batch_index].extend(decoded[item_start : item_start + n])

    return grouped


def _ground_truth(problem: Problem) -> dict[str, Any]:
    return {"numbers": problem.numbers, "target": problem.target, "solvable": problem.solvable}


def evaluate_generations(
    *,
    model_name: str,
    dataset_name: str,
    problems: Sequence[Problem],
    responses: Sequence[Sequence[str]],
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    first_exact: list[bool] = []
    first_format: list[bool] = []
    first_parse: list[bool] = []
    first_numbers: list[bool] = []
    first_rewards: list[float] = []
    correct_sample_counts: list[int] = []

    for problem, problem_responses in zip(problems, responses, strict=True):
        truth_json = json.dumps(_ground_truth(problem), separators=(",", ":"))
        correct_samples = 0
        for sample_index, response in enumerate(problem_responses):
            verification = verify_solution(response, problem.numbers, target=problem.target)
            reward = float(compute_score("game24", response, truth_json))
            correct_samples += int(verification.is_correct)
            if sample_index == 0:
                first_exact.append(verification.is_correct)
                first_format.append(verification.format_valid)
                first_parse.append(verification.parse_valid)
                first_numbers.append(verification.numbers_valid)
                first_rewards.append(reward)

            records.append(
                {
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "mode": mode,
                    "problem_id": problem.problem_id,
                    "numbers": problem.numbers,
                    "target": problem.target,
                    "solvable": problem.solvable,
                    "sample_index": sample_index,
                    "generation_text": response,
                    "answer": verification.expression,
                    "verification": verification.to_dict(),
                    "failure_reason": verification.error_reason,
                    "reward": reward,
                }
            )
        correct_sample_counts.append(correct_samples)

    total = len(problems)
    sample_n = max((len(item) for item in responses), default=0)
    exact_correct = sum(first_exact)
    summary: dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "mode": mode,
        "total": total,
        "sample_n": sample_n,
        "exact_correct": exact_correct,
        "exact_accuracy": exact_correct / total if total else 0.0,
        "format_rate": sum(first_format) / total if total else 0.0,
        "parse_rate": sum(first_parse) / total if total else 0.0,
        "number_usage_rate": sum(first_numbers) / total if total else 0.0,
        "reward_mean": sum(first_rewards) / total if total else 0.0,
    }
    if sample_n > 1:
        pass_at_n_correct = sum(count > 0 for count in correct_sample_counts)
        summary.update(
            {
                "pass_at_n_correct": pass_at_n_correct,
                "pass_at_n_rate": pass_at_n_correct / total if total else 0.0,
                "average_correct_samples_per_problem": sum(correct_sample_counts) / total if total else 0.0,
            }
        )
    if dataset_name == "unsolvable":
        summary["hallucinated_exact_correct"] = summary.get("pass_at_n_correct", exact_correct)
        summary["hallucinated_exact_rate"] = summary.get("pass_at_n_rate", summary["exact_accuracy"])
    return summary, records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_checkpoint_selection(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "checkpoint_dir",
        "adapter_dir",
        "exact_correct",
        "total",
        "exact_accuracy",
        "format_rate",
        "number_usage_rate",
        "parse_rate",
        "reward_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def evaluate_checkpoint_on_val(
    *,
    checkpoint: Checkpoint,
    sft_model_path: str,
    val_problems: Sequence[Problem],
    output_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    torch_dtype: str,
) -> dict[str, Any]:
    spec = ModelSpec(f"grpo_step_{checkpoint.step}", sft_model_path, str(checkpoint.adapter_dir))
    model, tokenizer = load_model_and_tokenizer(spec, torch_dtype=torch_dtype)
    try:
        responses = generate_responses(
            model,
            tokenizer,
            val_problems,
            n=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
        )
        summary, records = evaluate_generations(
            model_name=spec.name,
            dataset_name="val",
            problems=val_problems,
            responses=responses,
            mode="checkpoint_selection_greedy_pass1",
        )
        write_jsonl(output_dir / "predictions" / f"checkpoint_selection__global_step_{checkpoint.step}__val.jsonl", records)
    finally:
        unload_model(model)

    return {
        "step": checkpoint.step,
        "checkpoint_dir": str(checkpoint.checkpoint_dir),
        "adapter_dir": str(checkpoint.adapter_dir),
        "exact_correct": summary["exact_correct"],
        "total": summary["total"],
        "exact_accuracy": summary["exact_accuracy"],
        "format_rate": summary["format_rate"],
        "number_usage_rate": summary["number_usage_rate"],
        "parse_rate": summary["parse_rate"],
        "reward_mean": summary["reward_mean"],
    }


def run_model_eval(
    *,
    spec: ModelSpec,
    datasets: Mapping[str, Sequence[Problem]],
    output_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    sample_n: int,
    sample_temperature: float,
    sample_top_p: float,
    torch_dtype: str,
) -> list[dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(spec, torch_dtype=torch_dtype)
    summaries: list[dict[str, Any]] = []
    try:
        for dataset_name, problems in datasets.items():
            greedy = generate_responses(
                model,
                tokenizer,
                problems,
                n=1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                seed=seed,
            )
            summary, records = evaluate_generations(
                model_name=spec.name,
                dataset_name=dataset_name,
                problems=problems,
                responses=greedy,
                mode="greedy_pass1",
            )
            summaries.append(summary)
            write_jsonl(output_dir / "predictions" / f"{sanitize_name(spec.name)}__{dataset_name}__greedy_pass1.jsonl", records)

            sampling = generate_responses(
                model,
                tokenizer,
                problems,
                n=sample_n,
                do_sample=True,
                temperature=sample_temperature,
                top_p=sample_top_p,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                seed=seed,
            )
            summary, records = evaluate_generations(
                model_name=spec.name,
                dataset_name=dataset_name,
                problems=problems,
                responses=sampling,
                mode=f"sampling_pass{sample_n}",
            )
            summaries.append(summary)
            write_jsonl(output_dir / "predictions" / f"{sanitize_name(spec.name)}__{dataset_name}__sampling_pass{sample_n}.jsonl", records)
    finally:
        unload_model(model)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict final Game24 evaluation with val-only checkpoint selection.")
    parser.add_argument("--raw-model-path", required=True)
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--grpo-run-dir", required=True)
    parser.add_argument("--data-dir", default="data/game24")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-n", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_files = discover_data_files(Path(args.data_dir))
    datasets = {name: load_problems(path) for name, path in data_files.items()}
    checkpoints = discover_checkpoints(Path(args.grpo_run_dir))

    selection_rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        row = evaluate_checkpoint_on_val(
            checkpoint=checkpoint,
            sft_model_path=args.sft_model_path,
            val_problems=datasets["val"],
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            torch_dtype=args.torch_dtype,
        )
        selection_rows.append(row)
        print(json.dumps(row, sort_keys=True))

    write_checkpoint_selection(output_dir / "checkpoint_selection.csv", selection_rows)
    best_row = sorted(selection_rows, key=lambda row: (-float(row["exact_accuracy"]), int(row["step"])))[0]
    best_adapter = str(Path(best_row["adapter_dir"]))

    final_datasets = {
        "test": datasets["test"],
        "hard100": datasets["hard100"],
        "unsolvable": datasets["unsolvable"],
    }
    model_specs = [
        ModelSpec("raw_qwen", args.raw_model_path),
        ModelSpec("sft", args.sft_model_path),
        ModelSpec(f"grpo_step_{best_row['step']}", args.sft_model_path, best_adapter),
    ]

    final_summaries: list[dict[str, Any]] = []
    for spec in model_specs:
        final_summaries.extend(
            run_model_eval(
                spec=spec,
                datasets=final_datasets,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                seed=args.seed,
                sample_n=args.sample_n,
                sample_temperature=args.sample_temperature,
                sample_top_p=args.sample_top_p,
                torch_dtype=args.torch_dtype,
            )
        )

    final_results = {
        "data_files": {name: str(path) for name, path in data_files.items()},
        "grpo_run_dir": str(args.grpo_run_dir),
        "raw_model_path": args.raw_model_path,
        "sft_model_path": args.sft_model_path,
        "best_checkpoint": best_row,
        "selection_rule": "max val exact_accuracy; ties choose lower global_step",
        "strict_exact_accuracy_definition": (
            "verification.is_correct: exactly one answer tag, expression parses under the AST whitelist, "
            "input number multiset matches exactly, and Fraction value equals 24"
        ),
        "sampling": {
            "sample_n": args.sample_n,
            "temperature": args.sample_temperature,
            "top_p": args.sample_top_p,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
        "summaries": final_summaries,
    }
    (output_dir / "final_results.json").write_text(
        json.dumps(final_results, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_results, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
