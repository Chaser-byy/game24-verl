#!/usr/bin/env python3
"""Strict final evaluation for raw, SFT, and GRPO-LoRA Game24 models."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import time
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


@dataclass
class MetricTotals:
    total: int = 0
    sample_n: int = 0
    exact_correct: int = 0
    format_valid: int = 0
    parse_valid: int = 0
    number_usage_valid: int = 0
    reward_sum: float = 0.0
    pass_at_n_correct: int = 0
    correct_sample_sum: int = 0

    def update_problem(self, responses: Sequence[str], first_result: Any, first_reward: float, correct_samples: int) -> None:
        self.total += 1
        self.sample_n = max(self.sample_n, len(responses))
        self.exact_correct += int(first_result.is_correct)
        self.format_valid += int(first_result.format_valid)
        self.parse_valid += int(first_result.parse_valid)
        self.number_usage_valid += int(first_result.numbers_valid)
        self.reward_sum += first_reward
        self.pass_at_n_correct += int(correct_samples > 0)
        self.correct_sample_sum += correct_samples

    def summary(self, *, model_name: str, dataset_name: str, mode: str) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "mode": mode,
            "total": self.total,
            "sample_n": self.sample_n,
            "exact_correct": self.exact_correct,
            "exact_accuracy": self.exact_correct / self.total if self.total else 0.0,
            "format_rate": self.format_valid / self.total if self.total else 0.0,
            "parse_rate": self.parse_valid / self.total if self.total else 0.0,
            "number_usage_rate": self.number_usage_valid / self.total if self.total else 0.0,
            "reward_mean": self.reward_sum / self.total if self.total else 0.0,
        }
        if self.sample_n > 1:
            summary.update(
                {
                    "pass_at_n_correct": self.pass_at_n_correct,
                    "pass_at_n_rate": self.pass_at_n_correct / self.total if self.total else 0.0,
                    "average_correct_samples_per_problem": (
                        self.correct_sample_sum / self.total if self.total else 0.0
                    ),
                }
            )
        if dataset_name == "unsolvable":
            summary["hallucinated_exact_correct"] = summary.get("pass_at_n_correct", self.exact_correct)
            summary["hallucinated_exact_rate"] = summary.get("pass_at_n_rate", summary["exact_accuracy"])
        return summary


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
    if normalized == "train" or normalized.endswith("_train"):
        return "train"
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
            if split == "train":
                return "train"
    return None


def discover_data_files(data_dir: Path, required: set[str]) -> dict[str, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"evaluation data directory does not exist: {data_dir}")

    discovered: dict[str, Path] = {}
    candidates = sorted(data_dir.rglob("*.parquet"))
    for path in candidates:
        split = _split_from_file(path) or _split_from_rows(path)
        if split is not None:
            discovered.setdefault(split, path)

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


def _as_token_list(token_ids: Any) -> list[int]:
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def render_prompt_and_ids(messages: Sequence[Mapping[str, str]], tokenizer: Any) -> tuple[str, list[int]]:
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return str(rendered), _as_token_list(token_ids)

    rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
    tokenized = tokenizer(rendered, add_special_tokens=False)
    return rendered, _as_token_list(tokenized["input_ids"])


def _prompt_text(problem: Problem, tokenizer: Any) -> str:
    rendered, _ = render_prompt_and_ids(problem.prompt_messages, tokenizer)
    return rendered


def _token_difference(training_ids: Sequence[int], eval_ids: Sequence[int]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(training_ids, eval_ids)):
        if left != right:
            return {"first_difference_index": index, "training_token_id": left, "eval_token_id": right}
    if len(training_ids) != len(eval_ids):
        return {
            "first_difference_index": min(len(training_ids), len(eval_ids)),
            "training_length": len(training_ids),
            "eval_length": len(eval_ids),
        }
    return None


def write_prompt_audit(train_path: Path, tokenizer: Any, output_dir: Path, max_new_tokens: int) -> dict[str, Any]:
    frame = pd.read_parquet(train_path)
    if frame.empty:
        raise ValueError(f"training parquet is empty: {train_path}")

    row = frame.iloc[0]
    training_messages = _load_messages(row.get("prompt"))
    if training_messages is None:
        raise ValueError(f"training parquet row does not contain verl prompt messages: {train_path}")

    problem = _problem_from_row(row)
    eval_messages = problem.prompt_messages
    training_rendered, training_ids = render_prompt_and_ids(training_messages, tokenizer)
    eval_rendered, eval_ids = render_prompt_and_ids(eval_messages, tokenizer)

    messages_equal = training_messages == eval_messages
    rendered_equal = training_rendered == eval_rendered
    token_ids_equal = training_ids == eval_ids
    differences: list[str] = []
    if not messages_equal:
        differences.append("messages differ")
    if not rendered_equal:
        differences.append("rendered prompts differ")
    if not token_ids_equal:
        differences.append("token ids differ")

    audit = {
        "train_path": str(train_path),
        "problem_id": problem.problem_id,
        "numbers": problem.numbers,
        "target": problem.target,
        "training_sample_original_messages": training_messages,
        "eval_messages": eval_messages,
        "system_prompt": [item["content"] for item in training_messages if item["role"] == "system"],
        "user_prompt": [item["content"] for item in training_messages if item["role"] == "user"],
        "training_prompt_rendered": training_rendered,
        "eval_prompt_rendered": eval_rendered,
        "training_token_ids": training_ids,
        "eval_token_ids": eval_ids,
        "token_ids_equal": token_ids_equal,
        "messages_equal": messages_equal,
        "rendered_prompts_equal": rendered_equal,
        "differences": differences,
        "first_token_difference": _token_difference(training_ids, eval_ids),
        "tokenizer_padding_side": getattr(tokenizer, "padding_side", None),
        "tokenizer_pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "tokenizer_eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "tokenizer_eos_token": getattr(tokenizer, "eos_token", None),
        "tokenizer_chat_template": getattr(tokenizer, "chat_template", None),
        "generation_max_new_tokens": max_new_tokens,
        "format_requirement": "<think>...</think><answer>...</answer>",
    }

    path = output_dir / "prompt_audit.json"
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if differences:
        raise RuntimeError(f"prompt audit failed; see {path}: {differences}")
    return audit


def _torch_dtype(torch_dtype: str) -> Any:
    import torch

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return dtype_map[torch_dtype]


def load_model_and_tokenizer(spec: ModelSpec, torch_dtype: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.base_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.base_path,
        torch_dtype=_torch_dtype(torch_dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    if spec.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, spec.adapter_path, adapter_name="default")
        model.set_adapter("default")
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


def activate_lora_adapter(model: Any, checkpoint: Checkpoint, adapter_name: str) -> Any:
    if hasattr(model, "peft_config"):
        if adapter_name not in model.peft_config:
            model.load_adapter(str(checkpoint.adapter_dir), adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    else:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(checkpoint.adapter_dir), adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    model.eval()
    return model


def generate_response_batch(
    model: Any,
    tokenizer: Any,
    problems: Sequence[Problem],
    *,
    n: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[list[str]]:
    import torch

    prompts = [_prompt_text(problem, tokenizer) for problem in problems]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": n,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        output_ids = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(output_ids[:, prompt_length:], skip_special_tokens=True)
    grouped: list[list[str]] = []
    for batch_index in range(len(problems)):
        item_start = batch_index * n
        grouped.append(decoded[item_start : item_start + n])
    return grouped


def _ground_truth(problem: Problem) -> dict[str, Any]:
    return {"numbers": problem.numbers, "target": problem.target, "solvable": problem.solvable}


def records_for_batch(
    *,
    model_name: str,
    dataset_name: str,
    mode: str,
    problems: Sequence[Problem],
    responses: Sequence[Sequence[str]],
    totals: MetricTotals,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for problem, problem_responses in zip(problems, responses, strict=True):
        truth_json = json.dumps(_ground_truth(problem), separators=(",", ":"))
        correct_samples = 0
        first_result = None
        first_reward = 0.0

        for sample_index, response in enumerate(problem_responses):
            verification = verify_solution(response, problem.numbers, target=problem.target)
            reward = float(compute_score("game24", response, truth_json))
            correct_samples += int(verification.is_correct)
            if sample_index == 0:
                first_result = verification
                first_reward = reward

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
                    "is_correct": verification.is_correct,
                    "verification": verification.to_dict(),
                    "failure_reason": verification.error_reason,
                    "reward": reward,
                }
            )

        if first_result is None:
            raise ValueError("generation returned zero responses for a problem")
        totals.update_problem(problem_responses, first_result, first_reward, correct_samples)
    return records


def evaluate_model_dataset(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    dataset_name: str,
    problems: Sequence[Problem],
    output_jsonl: Path,
    mode: str,
    n: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    record_limit: int | None,
) -> dict[str, Any]:
    from transformers import set_seed

    set_seed(seed)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    totals = MetricTotals()
    written = 0
    start_time = time.monotonic()
    total_batches = math.ceil(len(problems) / batch_size) if problems else 0

    with output_jsonl.open("w", encoding="utf-8") as handle:
        for batch_index, start in enumerate(range(0, len(problems), batch_size), start=1):
            batch_problems = problems[start : start + batch_size]
            responses = generate_response_batch(
                model,
                tokenizer,
                batch_problems,
                n=n,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            records = records_for_batch(
                model_name=model_name,
                dataset_name=dataset_name,
                mode=mode,
                problems=batch_problems,
                responses=responses,
                totals=totals,
            )
            for record in records:
                if record_limit is None or written < record_limit:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            handle.flush()

            elapsed = time.monotonic() - start_time
            print(
                f"[{model_name}][{dataset_name}][{mode}] "
                f"batch {batch_index}/{total_batches} completed "
                f"({min(start + len(batch_problems), len(problems))}/{len(problems)} problems), "
                f"elapsed={elapsed:.1f}s"
            )

    summary = totals.summary(model_name=model_name, dataset_name=dataset_name, mode=mode)
    summary["prediction_jsonl"] = str(output_jsonl)
    summary["diagnostic_records_written"] = written
    summary["elapsed_seconds"] = time.monotonic() - start_time
    return summary


def write_checkpoint_selection(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "checkpoint_dir",
        "adapter_dir",
        "active_adapter",
        "exact_correct",
        "total",
        "exact_accuracy",
        "format_rate",
        "number_usage_rate",
        "parse_rate",
        "reward_mean",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def select_grpo_checkpoint(
    *,
    sft_model_path: str,
    checkpoints: Sequence[Checkpoint],
    val_problems: Sequence[Problem],
    output_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    torch_dtype: str,
    record_limit: int | None,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    model, tokenizer = load_model_and_tokenizer(ModelSpec("grpo_selection_base", sft_model_path), torch_dtype)
    rows: list[dict[str, Any]] = []

    for checkpoint in checkpoints:
        adapter_name = f"global_step_{checkpoint.step}"
        model = activate_lora_adapter(model, checkpoint, adapter_name)
        start_time = time.monotonic()
        summary = evaluate_model_dataset(
            model=model,
            tokenizer=tokenizer,
            model_name=f"grpo_step_{checkpoint.step}",
            dataset_name="val",
            problems=val_problems,
            output_jsonl=output_dir / "predictions" / f"checkpoint_selection__global_step_{checkpoint.step}__val.jsonl",
            mode="checkpoint_selection_greedy_pass1",
            n=1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            seed=seed,
            record_limit=record_limit,
        )
        elapsed = time.monotonic() - start_time
        row = {
            "step": checkpoint.step,
            "checkpoint_dir": str(checkpoint.checkpoint_dir),
            "adapter_dir": str(checkpoint.adapter_dir),
            "active_adapter": adapter_name,
            "exact_correct": summary["exact_correct"],
            "total": summary["total"],
            "exact_accuracy": summary["exact_accuracy"],
            "format_rate": summary["format_rate"],
            "number_usage_rate": summary["number_usage_rate"],
            "parse_rate": summary["parse_rate"],
            "reward_mean": summary["reward_mean"],
            "elapsed_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"[checkpoint_selection] step={checkpoint.step} completed={row['total']} "
            f"elapsed={elapsed:.1f}s exact_accuracy={row['exact_accuracy']:.6f} "
            f"exact_correct={row['exact_correct']}/{row['total']}"
        )

    write_checkpoint_selection(output_dir / "checkpoint_selection.csv", rows)
    best_row = sorted(rows, key=lambda row: (-float(row["exact_accuracy"]), int(row["step"])))[0]
    model.set_adapter(str(best_row["active_adapter"]))
    return model, tokenizer, best_row, rows


def run_model_eval(
    *,
    model: Any,
    tokenizer: Any,
    spec: ModelSpec,
    datasets: Mapping[str, Sequence[Problem]],
    output_dir: Path,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    sample_n: int,
    sample_temperature: float,
    sample_top_p: float,
    record_limit: int | None,
    run_sampling: bool,
    compact_names: bool,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset_name, problems in datasets.items():
        prefix = f"{sanitize_name(spec.name)}_{dataset_name}" if compact_names else f"{sanitize_name(spec.name)}__{dataset_name}"
        greedy_path = output_dir / "predictions" / f"{prefix}.jsonl" if compact_names else output_dir / "predictions" / f"{prefix}__greedy_pass1.jsonl"
        summaries.append(
            evaluate_model_dataset(
                model=model,
                tokenizer=tokenizer,
                model_name=spec.name,
                dataset_name=dataset_name,
                problems=problems,
                output_jsonl=greedy_path,
                mode="greedy_pass1",
                n=1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                batch_size=batch_size,
                seed=seed,
                record_limit=record_limit,
            )
        )

        if run_sampling:
            summaries.append(
                evaluate_model_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=spec.name,
                    dataset_name=dataset_name,
                    problems=problems,
                    output_jsonl=output_dir
                    / "predictions"
                    / f"{sanitize_name(spec.name)}__{dataset_name}__sampling_pass{sample_n}.jsonl",
                    mode=f"sampling_pass{sample_n}",
                    n=sample_n,
                    do_sample=True,
                    temperature=sample_temperature,
                    top_p=sample_top_p,
                    max_new_tokens=max_new_tokens,
                    batch_size=batch_size,
                    seed=seed,
                    record_limit=record_limit,
                )
            )
    return summaries


def required_splits(mode: str) -> set[str]:
    if mode == "quick":
        return {"train", "val", "test"}
    return {"train", "val", "test", "hard100", "unsolvable"}


def load_required_datasets(data_files: Mapping[str, Path], split_names: Sequence[str]) -> dict[str, list[Problem]]:
    return {name: load_problems(data_files[name]) for name in split_names}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def pass1_brief(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exact_correct": summary["exact_correct"],
        "total": summary["total"],
        "exact_accuracy": summary["exact_accuracy"],
        "format_rate": summary["format_rate"],
        "parse_rate": summary["parse_rate"],
        "number_usage_rate": summary["number_usage_rate"],
        "reward_mean": summary["reward_mean"],
    }


def run_quick(args: argparse.Namespace, data_files: Mapping[str, Path], checkpoints: Sequence[Checkpoint]) -> None:
    output_dir = Path(args.output_dir)
    datasets = load_required_datasets(data_files, ["val", "test"])
    record_limit = args.diagnostic_limit

    grpo_model, grpo_tokenizer, best_row, selection_rows = select_grpo_checkpoint(
        sft_model_path=args.sft_model_path,
        checkpoints=checkpoints,
        val_problems=datasets["val"],
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        record_limit=record_limit,
    )
    prompt_audit = write_prompt_audit(data_files["train"], grpo_tokenizer, output_dir, args.max_new_tokens)

    grpo_summaries = run_model_eval(
        model=grpo_model,
        tokenizer=grpo_tokenizer,
        spec=ModelSpec("grpo", args.sft_model_path, str(best_row["adapter_dir"])),
        datasets=datasets,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        sample_n=args.sample_n,
        sample_temperature=args.sample_temperature,
        sample_top_p=args.sample_top_p,
        record_limit=record_limit,
        run_sampling=False,
        compact_names=True,
    )
    unload_model(grpo_model)

    sft_model, sft_tokenizer = load_model_and_tokenizer(ModelSpec("sft", args.sft_model_path), args.torch_dtype)
    try:
        sft_summaries = run_model_eval(
            model=sft_model,
            tokenizer=sft_tokenizer,
            spec=ModelSpec("sft", args.sft_model_path),
            datasets=datasets,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            seed=args.seed,
            sample_n=args.sample_n,
            sample_temperature=args.sample_temperature,
            sample_top_p=args.sample_top_p,
            record_limit=record_limit,
            run_sampling=False,
            compact_names=True,
        )
    finally:
        unload_model(sft_model)

    summaries = sft_summaries + grpo_summaries
    by_key = {f"{item['model_name']}_{item['dataset_name']}": item for item in summaries}
    quick_results = {
        "mode": "quick",
        "data_files": {name: str(path) for name, path in data_files.items()},
        "grpo_run_dir": str(args.grpo_run_dir),
        "sft_model_path": args.sft_model_path,
        "best_checkpoint": best_row,
        "selection_rule": "max val exact_accuracy; ties choose lower global_step",
        "checkpoint_selection": selection_rows,
        "prompt_audit_path": str(output_dir / "prompt_audit.json"),
        "prompt_audit_token_ids_equal": prompt_audit["token_ids_equal"],
        "comparison": {
            "sft_val_strict_pass1": pass1_brief(by_key["sft_val"]),
            "grpo_val_strict_pass1": pass1_brief(by_key["grpo_val"]),
            "sft_test_strict_pass1": pass1_brief(by_key["sft_test"]),
            "grpo_test_strict_pass1": pass1_brief(by_key["grpo_test"]),
        },
        "summaries": summaries,
    }
    write_json(output_dir / "quick_results.json", quick_results)
    print(json.dumps(quick_results, indent=2, sort_keys=True, ensure_ascii=False))


def run_full(args: argparse.Namespace, data_files: Mapping[str, Path], checkpoints: Sequence[Checkpoint]) -> None:
    output_dir = Path(args.output_dir)
    all_datasets = load_required_datasets(data_files, ["val", "test", "hard100", "unsolvable"])
    final_datasets = {name: all_datasets[name] for name in ["test", "hard100", "unsolvable"]}
    record_limit = None

    grpo_model, grpo_tokenizer, best_row, selection_rows = select_grpo_checkpoint(
        sft_model_path=args.sft_model_path,
        checkpoints=checkpoints,
        val_problems=all_datasets["val"],
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        torch_dtype=args.torch_dtype,
        record_limit=record_limit,
    )
    prompt_audit = write_prompt_audit(data_files["train"], grpo_tokenizer, output_dir, args.max_new_tokens)

    final_summaries: list[dict[str, Any]] = []
    try:
        final_summaries.extend(
            run_model_eval(
                model=grpo_model,
                tokenizer=grpo_tokenizer,
                spec=ModelSpec(f"grpo_step_{best_row['step']}", args.sft_model_path, str(best_row["adapter_dir"])),
                datasets=final_datasets,
                output_dir=output_dir,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                seed=args.seed,
                sample_n=args.sample_n,
                sample_temperature=args.sample_temperature,
                sample_top_p=args.sample_top_p,
                record_limit=record_limit,
                run_sampling=True,
                compact_names=False,
            )
        )
    finally:
        unload_model(grpo_model)

    for spec in [ModelSpec("raw_qwen", args.raw_model_path), ModelSpec("sft", args.sft_model_path)]:
        model, tokenizer = load_model_and_tokenizer(spec, args.torch_dtype)
        try:
            final_summaries.extend(
                run_model_eval(
                    model=model,
                    tokenizer=tokenizer,
                    spec=spec,
                    datasets=final_datasets,
                    output_dir=output_dir,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    sample_n=args.sample_n,
                    sample_temperature=args.sample_temperature,
                    sample_top_p=args.sample_top_p,
                    record_limit=record_limit,
                    run_sampling=True,
                    compact_names=False,
                )
            )
        finally:
            unload_model(model)

    final_results = {
        "mode": "full",
        "data_files": {name: str(path) for name, path in data_files.items()},
        "grpo_run_dir": str(args.grpo_run_dir),
        "raw_model_path": args.raw_model_path,
        "sft_model_path": args.sft_model_path,
        "best_checkpoint": best_row,
        "selection_rule": "max val exact_accuracy; ties choose lower global_step",
        "checkpoint_selection": selection_rows,
        "prompt_audit_path": str(output_dir / "prompt_audit.json"),
        "prompt_audit_token_ids_equal": prompt_audit["token_ids_equal"],
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
    write_json(output_dir / "final_results.json", final_results)
    print(json.dumps(final_results, indent=2, sort_keys=True, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict final Game24 evaluation with val-only checkpoint selection.")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--raw-model-path", required=True)
    parser.add_argument("--sft-model-path", required=True)
    parser.add_argument("--grpo-run-dir", required=True)
    parser.add_argument("--data-dir", default="data/game24")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sample-n", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--diagnostic-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.sample_n <= 0:
        raise ValueError("--sample-n must be positive")
    if args.diagnostic_limit < 0:
        raise ValueError("--diagnostic-limit must be non-negative")

    data_files = discover_data_files(Path(args.data_dir), required_splits(args.mode))
    checkpoints = discover_checkpoints(Path(args.grpo_run_dir))

    if args.mode == "quick":
        run_quick(args, data_files, checkpoints)
    else:
        run_full(args, data_files, checkpoints)


if __name__ == "__main__":
    main()
