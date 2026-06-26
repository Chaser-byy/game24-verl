#!/usr/bin/env python3
"""Evaluate a Game of 24 model or LoRA adapter with vLLM."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from game24.metrics import GenerationRecord, summarize_generation_groups
from game24.prompt import build_chat_prompt
from game24.verifier import verify_solution


def _load_messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        loaded = json.loads(value)
        if isinstance(loaded, list):
            return loaded
    raise ValueError(f"unsupported prompt value: {type(value).__name__}")


def _decode_nested(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


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


def _jsonable_truth(truth: dict[str, Any]) -> dict[str, Any]:
    numbers = _decode_nested(truth["numbers"])
    if hasattr(numbers, "tolist"):
        numbers = numbers.tolist()
    return {
        "numbers": [int(number) for number in numbers],
        "target": int(truth.get("target", 24)),
        "solvable": bool(truth.get("solvable", True)),
    }


def _prompt_text(row: pd.Series, tokenizer: Any) -> str:
    if "prompt" in row and row["prompt"] is not None:
        messages = _load_messages(row["prompt"])
    else:
        truth = _load_ground_truth(row)
        messages = build_chat_prompt(truth["numbers"], target=int(truth.get("target", 24)))

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def _finish_reason_is_truncated(output: Any) -> bool:
    finish_reason = getattr(output, "finish_reason", None)
    return str(finish_reason).lower() in {"length", "max_tokens"}


def evaluate_file(args: argparse.Namespace, data_file: Path, llm: Any, sampling_params: Any) -> dict[str, Any]:
    tokenizer = llm.get_tokenizer()
    frame = pd.read_parquet(data_file)
    prompts = [_prompt_text(row, tokenizer) for _, row in frame.iterrows()]
    truths = [_jsonable_truth(_load_ground_truth(row)) for _, row in frame.iterrows()]

    generate_kwargs: dict[str, Any] = {}
    if args.adapter:
        from vllm.lora.request import LoRARequest

        generate_kwargs["lora_request"] = LoRARequest("game24_lora", 1, args.adapter)

    request_outputs = llm.generate(prompts, sampling_params, **generate_kwargs)
    groups: list[list[GenerationRecord]] = []
    jsonl_records: list[dict[str, Any]] = []

    for row_index, (request_output, truth) in enumerate(zip(request_outputs, truths, strict=True)):
        numbers = truth["numbers"]
        target = int(truth.get("target", 24))
        group: list[GenerationRecord] = []

        for sample_index, output in enumerate(request_output.outputs):
            response = output.text
            verification = verify_solution(response, numbers, target=target)
            record = GenerationRecord(
                response=response,
                verification=verification,
                truncated=_finish_reason_is_truncated(output),
            )
            group.append(record)
            jsonl_records.append(
                {
                    "data_file": str(data_file),
                    "row_index": row_index,
                    "sample_index": sample_index,
                    "response": response,
                    "truncated": record.truncated,
                    "verification": verification.to_dict(),
                    "ground_truth": truth,
                }
            )
        groups.append(group)

    extra_info_text = frame.get("extra_info", pd.Series(dtype=str)).astype(str)
    is_unsolvable = data_file.stem == "unsolvable" or bool(extra_info_text.str.contains("unsolvable").any())
    summary = summarize_generation_groups(groups, unsolvable=is_unsolvable)
    summary["data_file"] = str(data_file)

    return {"summary": summary, "records": jsonl_records}


def _resolve_data_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    names = ["test.parquet", "tot_hard100.parquet", "unsolvable.parquet"]
    files = [path / name for name in names if (path / name).exists()]
    if not files:
        raise FileNotFoundError(f"no evaluation parquet files found under {path}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Game of 24 outputs with vLLM and the strict verifier.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--data-file", required=True, help="A parquet file or a directory containing test/tot/unsolvable parquet files.")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, enable_lora=bool(args.adapter), trust_remote_code=True)
    sampling_params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    all_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for data_file in _resolve_data_files(Path(args.data_file)):
        result = evaluate_file(args, data_file, llm, sampling_params)
        summaries.append(result["summary"])
        all_records.extend(result["records"])
        print(json.dumps(result["summary"], indent=2, sort_keys=True))

    if args.output_jsonl:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
