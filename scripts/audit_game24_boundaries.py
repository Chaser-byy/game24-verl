#!/usr/bin/env python3
"""Audit Game24 split boundaries before improved GRPO training."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


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


def _numbers_from_row(row: pd.Series) -> tuple[int, int, int, int]:
    reward_model = _decode_nested(row.get("reward_model"))
    if isinstance(reward_model, Mapping) and "ground_truth" in reward_model:
        truth = _decode_nested(reward_model["ground_truth"])
        if isinstance(truth, Mapping) and "numbers" in truth:
            return _canonical_id(truth["numbers"])

    extra_info = _decode_nested(row.get("extra_info"))
    if isinstance(extra_info, Mapping) and "numbers" in extra_info:
        return _canonical_id(extra_info["numbers"])

    if "numbers" in row:
        return _canonical_id(row["numbers"])

    raise ValueError(f"could not extract numbers from row fields: {list(row.index)}")


def _canonical_id(numbers: Sequence[Any]) -> tuple[int, int, int, int]:
    if len(numbers) != 4:
        raise ValueError(f"expected four numbers, got {numbers!r}")
    return tuple(sorted(int(number) for number in numbers))  # type: ignore[return-value]


def load_ids(path: Path) -> set[tuple[int, int, int, int]]:
    if not path.exists():
        raise FileNotFoundError(f"required split file does not exist: {path}")
    frame = pd.read_parquet(path)
    return {_numbers_from_row(row) for _, row in frame.iterrows()}


def check_empty(name: str, left: set[tuple[int, int, int, int]], right: set[tuple[int, int, int, int]]) -> dict[str, Any]:
    overlap = left & right
    result = {
        "name": name,
        "count": len(overlap),
        "sample": [list(item) for item in sorted(overlap)[:20]],
    }
    if overlap:
        raise ValueError(f"{name} overlap is not empty: {json.dumps(result, sort_keys=True)}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical Game24 split IDs.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--hard100-file", required=True)
    parser.add_argument("--sft-train-file", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_ids = load_ids(Path(args.train_file))
    val_ids = load_ids(Path(args.val_file))
    test_ids = load_ids(Path(args.test_file))
    hard100_ids = load_ids(Path(args.hard100_file))
    sft_train_ids = load_ids(Path(args.sft_train_file))

    checks = [
        check_empty("train_vs_val", train_ids, val_ids),
        check_empty("train_vs_test", train_ids, test_ids),
        check_empty("train_vs_hard100", train_ids, hard100_ids),
        check_empty("sft_train_vs_project_val", sft_train_ids, val_ids),
        check_empty("sft_train_vs_project_test", sft_train_ids, test_ids),
        check_empty("sft_train_vs_hard100", sft_train_ids, hard100_ids),
    ]
    report = {
        "canonical_id": "tuple(sorted(numbers))",
        "counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "hard100": len(hard100_ids),
            "sft_train": len(sft_train_ids),
        },
        "checks": checks,
    }
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
