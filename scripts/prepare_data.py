#!/usr/bin/env python3
"""Prepare Game of 24 parquet data for verl."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from game24.prompt import build_chat_prompt

NLILE_DATASET = "nlile/24-game"
TOT_DATASET = "test-time-compute/game-of-24"
TOT_HARD100_SLICE = slice(900, 1000)

NUMBER_FIELDS = (
    "numbers",
    "nums",
    "digits",
    "cards",
    "values",
    "input",
    "inputs",
    "question",
    "problem",
    "puzzle",
)
SOLVABLE_FIELDS = ("solvable", "is_solvable", "can_solve", "has_solution", "label")
TARGET_FIELDS = ("target", "answer", "result")


@dataclass(frozen=True)
class Problem:
    numbers: tuple[int, int, int, int]
    target: int
    solvable: bool
    source: str
    source_index: int

    @property
    def problem_id(self) -> tuple[int, int, int, int]:
        return tuple(sorted(self.numbers))  # type: ignore[return-value]

    @property
    def problem_id_text(self) -> str:
        return "_".join(str(number) for number in self.problem_id)


def _available_fields(row: Mapping[str, Any]) -> str:
    return ", ".join(sorted(row.keys()))


def _numbers_from_value(value: Any) -> tuple[int, int, int, int] | None:
    if isinstance(value, str):
        parsed = [int(match) for match in re.findall(r"-?\d+", value)]
        if len(parsed) == 5 and 24 in parsed:
            parsed.remove(24)
        values = parsed
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        return None

    if len(values) != 4:
        return None

    normalized: list[int] = []
    for number in values:
        if isinstance(number, bool):
            return None
        try:
            normalized.append(int(number))
        except (TypeError, ValueError):
            return None
    return tuple(normalized)  # type: ignore[return-value]


def extract_numbers(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    for field in NUMBER_FIELDS:
        if field in row:
            numbers = _numbers_from_value(row[field])
            if numbers is not None:
                return numbers
    raise ValueError(f"could not extract four numbers; available fields: {_available_fields(row)}")


def _bool_from_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "solvable", "solved"}:
            return True
        if normalized in {"false", "no", "n", "0", "unsolvable", "not_solvable"}:
            return False
    return None


def extract_solvable(row: Mapping[str, Any], *, default: bool | None = None) -> bool:
    for field in SOLVABLE_FIELDS:
        if field in row:
            solvable = _bool_from_value(row[field])
            if solvable is not None:
                return solvable
    if default is not None:
        return default
    raise ValueError(f"could not extract solvable flag; available fields: {_available_fields(row)}")


def extract_target(row: Mapping[str, Any], *, default: int = 24) -> int:
    for field in TARGET_FIELDS:
        if field in row:
            value = row[field]
            if isinstance(value, bool):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return default


def _iter_dataset_rows(dataset_name: str) -> list[tuple[str, int, Mapping[str, Any]]]:
    dataset = load_dataset(dataset_name)
    rows: list[tuple[str, int, Mapping[str, Any]]] = []

    if isinstance(dataset, DatasetDict):
        for split_name, split in dataset.items():
            for index, row in enumerate(split):
                rows.append((f"{dataset_name}:{split_name}", len(rows), row))
    elif isinstance(dataset, Dataset):
        for index, row in enumerate(dataset):
            rows.append((dataset_name, index, row))
    else:
        raise TypeError(f"unsupported dataset object for {dataset_name}: {type(dataset).__name__}")

    return rows


def _problem_from_row(
    row: Mapping[str, Any],
    *,
    source: str,
    source_index: int,
    default_solvable: bool | None,
) -> Problem:
    return Problem(
        numbers=extract_numbers(row),
        target=extract_target(row, default=24),
        solvable=extract_solvable(row, default=default_solvable),
        source=source,
        source_index=source_index,
    )


def _dedupe_by_problem(problems: Iterable[Problem]) -> list[Problem]:
    seen: dict[tuple[int, int, int, int], Problem] = {}
    for problem in problems:
        seen.setdefault(problem.problem_id, problem)
    return list(seen.values())


def _resolve_size(value: str, total: int, name: str) -> int:
    if "." in value:
        fraction = float(value)
        if not 0 < fraction < 1:
            raise ValueError(f"{name} as a fraction must be in (0, 1), got {value}")
        return int(round(total * fraction))
    size = int(value)
    if size < 0:
        raise ValueError(f"{name} must be non-negative, got {size}")
    return size


def _ids(problems: Sequence[Problem]) -> set[tuple[int, int, int, int]]:
    return {problem.problem_id for problem in problems}


def _pairwise_intersections(named_sets: Mapping[str, set[tuple[int, int, int, int]]]) -> dict[str, int]:
    intersections: dict[str, int] = {}
    for left, right in combinations(named_sets.keys(), 2):
        count = len(named_sets[left] & named_sets[right])
        intersections[f"{left}__{right}"] = count
    return intersections


def _assert_disjoint(named_sets: Mapping[str, set[tuple[int, int, int, int]]]) -> None:
    overlaps = {name: count for name, count in _pairwise_intersections(named_sets).items() if count}
    if overlaps:
        raise ValueError(f"problem ID overlap between output splits: {overlaps}")


def _to_verl_record(problem: Problem, split: str) -> dict[str, Any]:
    ground_truth = {
        "numbers": list(problem.numbers),
        "target": problem.target,
        "solvable": problem.solvable,
    }
    return {
        "data_source": "game24",
        "prompt": build_chat_prompt(problem.numbers, target=problem.target),
        "ability": "game24",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, separators=(",", ":")),
        },
        "extra_info": {
            "problem_id": problem.problem_id_text,
            "numbers": list(problem.numbers),
            "target": problem.target,
            "solvable": problem.solvable,
            "source": problem.source,
            "source_index": problem.source_index,
            "split": split,
        },
    }


def _write_parquet(path: Path, problems: Sequence[Problem], split: str) -> None:
    records = [_to_verl_record(problem, split) for problem in problems]
    pd.DataFrame(records).to_parquet(path, index=False)


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nlile_rows = _iter_dataset_rows(NLILE_DATASET)
    tot_rows = _iter_dataset_rows(TOT_DATASET)

    nlile_problems = [
        _problem_from_row(row, source=source, source_index=index, default_solvable=None)
        for source, index, row in nlile_rows
    ]
    tot_problems = [
        _problem_from_row(row, source=source, source_index=index, default_solvable=True)
        for source, index, row in tot_rows
    ]

    tot_hard100_raw = tot_problems[TOT_HARD100_SLICE]
    if len(tot_hard100_raw) != 100:
        raise ValueError(
            f"{TOT_DATASET} must provide at least 1000 rows for ToT-100; got {len(tot_problems)} rows"
        )

    solvable_nlile = [problem for problem in nlile_problems if problem.solvable]
    unsolvable = _dedupe_by_problem(problem for problem in nlile_problems if not problem.solvable)
    tot_hard100 = _dedupe_by_problem(tot_hard100_raw)
    tot_hard100_ids = _ids(tot_hard100)

    solvable_deduped = _dedupe_by_problem(solvable_nlile)
    train_candidates = [problem for problem in solvable_deduped if problem.problem_id not in tot_hard100_ids]

    rng = random.Random(args.seed)
    rng.shuffle(train_candidates)

    val_size = min(_resolve_size(args.val_size, len(train_candidates), "--val-size"), len(train_candidates))
    val_problems = train_candidates[:val_size]
    train_problems = train_candidates[val_size:]
    if args.train_limit is not None:
        train_problems = train_problems[: args.train_limit]

    train_val_ids = _ids(train_problems) | _ids(val_problems)
    tot_deduped = _dedupe_by_problem(tot_problems)
    ordinary_test_candidates = [
        problem
        for problem in tot_deduped
        if problem.problem_id not in train_val_ids and problem.problem_id not in tot_hard100_ids
    ]
    rng.shuffle(ordinary_test_candidates)
    test_size = min(_resolve_size(args.test_size, len(ordinary_test_candidates), "--test-size"), len(ordinary_test_candidates))
    test_problems = ordinary_test_candidates[:test_size]

    required = {
        "train": train_problems,
        "val": val_problems,
        "test": test_problems,
        "tot_hard100": tot_hard100,
    }
    empty = [name for name, problems in required.items() if not problems]
    if empty:
        raise ValueError(f"required output split is empty: {', '.join(empty)}")

    split_sets = {
        "train": _ids(train_problems),
        "val": _ids(val_problems),
        "test": _ids(test_problems),
        "tot_hard100": _ids(tot_hard100),
        "unsolvable": _ids(unsolvable),
    }
    intersections = _pairwise_intersections(split_sets)
    _assert_disjoint(split_sets)

    _write_parquet(output_dir / "train.parquet", train_problems, "train")
    _write_parquet(output_dir / "val.parquet", val_problems, "val")
    _write_parquet(output_dir / "test.parquet", test_problems, "test")
    _write_parquet(output_dir / "tot_hard100.parquet", tot_hard100, "tot_hard100")
    _write_parquet(output_dir / "unsolvable.parquet", unsolvable, "unsolvable")

    stats = {
        "datasets": {
            NLILE_DATASET: {
                "raw_count": len(nlile_rows),
                "solvable_raw_count": len(solvable_nlile),
                "solvable_dedup_count": len(solvable_deduped),
                "unsolvable_dedup_count": len(unsolvable),
            },
            TOT_DATASET: {
                "raw_count": len(tot_rows),
                "dedup_count": len(tot_deduped),
                "tot_hard100_raw_count": len(tot_hard100_raw),
                "tot_hard100_dedup_count": len(tot_hard100),
            },
        },
        "excluded": {
            "train_candidates_removed_by_tot_hard100": len(solvable_deduped) - len(train_candidates),
            "ordinary_test_candidates_removed_by_train_val_or_tot_hard100": len(tot_deduped)
            - len(ordinary_test_candidates),
        },
        "intersections": intersections,
        "outputs": {
            "train": len(train_problems),
            "val": len(val_problems),
            "test": len(test_problems),
            "tot_hard100": len(tot_hard100),
            "unsolvable": len(unsolvable),
        },
        "seed": args.seed,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "train_limit": args.train_limit,
    }

    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Game of 24 parquet data for verl.")
    parser.add_argument("--output-dir", default="data/game24")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-size", default="512")
    parser.add_argument("--test-size", default="1000")
    parser.add_argument("--train-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    prepare_data(parse_args())


if __name__ == "__main__":
    main()
