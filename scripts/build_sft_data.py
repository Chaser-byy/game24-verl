#!/usr/bin/env python3
"""Build high-quality SFT trajectories from the prepared train split only."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import pandas as pd

from game24.prompt import build_user_prompt
from game24.solver import Solution, solve_game24
from game24.trajectory import solution_to_response


@dataclass(frozen=True)
class Problem:
    problem_id: str
    numbers: tuple[int, int, int, int]
    target: int


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


def _normalize_numbers(value: Any) -> tuple[int, int, int, int]:
    value = _decode_nested(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"numbers must be a sequence, got {type(value).__name__}")
    if len(value) != 4:
        raise ValueError(f"expected exactly 4 numbers, got {len(value)}")
    if any(isinstance(number, bool) for number in value):
        raise ValueError(f"numbers must be non-boolean integers, got {value!r}")
    return tuple(int(number) for number in value)  # type: ignore[return-value]


def _problem_id(numbers: Sequence[int]) -> str:
    return "_".join(str(number) for number in sorted(int(item) for item in numbers))


def _ground_truth_from_row(row: pd.Series) -> dict[str, Any]:
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
        }

    raise ValueError("row does not contain reward_model.ground_truth or extra_info.numbers")


def _problem_from_row(row: pd.Series) -> Problem:
    extra_info = _decode_nested(row.get("extra_info"))
    truth = _ground_truth_from_row(row)
    numbers = _normalize_numbers(truth["numbers"])
    target = int(truth.get("target", 24))

    problem_id = None
    if isinstance(extra_info, Mapping):
        raw_problem_id = extra_info.get("problem_id")
        if raw_problem_id is not None:
            problem_id = str(raw_problem_id)

    return Problem(
        problem_id=problem_id or _problem_id(numbers),
        numbers=numbers,
        target=target,
    )


def _load_unique_problems(path: Path) -> dict[str, Problem]:
    frame = pd.read_parquet(path)
    problems: dict[str, Problem] = {}
    for _, row in frame.iterrows():
        problem = _problem_from_row(row)
        problems.setdefault(problem.problem_id, problem)
    return problems


def _read_required_split(processed_data_dir: Path, filename: str) -> dict[str, Problem]:
    path = processed_data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"required prepared split does not exist: {path}")
    return _load_unique_problems(path)


def _unique_permutations(numbers: Sequence[int], *, limit: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    unique = sorted(set(permutations(tuple(numbers), 4)))
    rng.shuffle(unique)
    return [tuple(item) for item in unique[:limit]]  # type: ignore[list-item]


def _prompt_from_template(numbers: Sequence[int], target: int, template_index: int) -> str:
    if template_index == 0:
        return build_user_prompt(numbers, target=target)

    numbers_text = ", ".join(str(number) for number in numbers)
    if template_index == 1:
        return (
            f"Make exactly {target} using these four numbers: {numbers_text}.\n\n"
            "Rules:\n"
            "- Use every given number exactly once.\n"
            "- Use only +, -, *, /, and parentheses.\n"
            "- Fractions during intermediate steps are allowed.\n"
            "- Do not concatenate digits or introduce new numbers.\n\n"
            "Respond only with:\n"
            "<think>\n"
            "brief derivation\n"
            "</think>\n"
            "<answer>\n"
            "final expression\n"
            "</answer>"
        )

    return (
        "Solve this 24-point arithmetic puzzle.\n\n"
        f"Input numbers: {numbers_text}\n"
        f"Goal: {target}\n\n"
        "Your expression must use each input number once and only once. "
        "Allowed operators are +, -, *, / with parentheses. "
        "Intermediate fractions are valid, but digit concatenation is not.\n\n"
        "Use this XML format exactly:\n"
        "<think>\n"
        "brief derivation\n"
        "</think>\n"
        "<answer>\n"
        "final expression\n"
        "</answer>"
    )


def _select_diverse_solutions(solutions: Sequence[Solution], limit: int) -> list[Solution]:
    remaining = list(solutions)
    selected: list[Solution] = []
    seen_structures: set[str] = set()
    seen_operator_sequences: set[tuple[str, ...]] = set()
    seen_top_operators: set[str] = set()
    seen_fraction_flags: set[bool] = set()

    while remaining and len(selected) < limit:
        best_index = 0
        best_score: tuple[int, int, int] | None = None

        for index, solution in enumerate(remaining):
            novelty = 0
            novelty += int(solution.structure_key not in seen_structures)
            novelty += int(solution.operator_sequence not in seen_operator_sequences)
            novelty += int(solution.top_operator not in seen_top_operators)
            novelty += int(solution.requires_fraction not in seen_fraction_flags)
            score = (novelty, -len(solution.expression), -index)
            if best_score is None or score > best_score:
                best_index = index
                best_score = score

        chosen = remaining.pop(best_index)
        selected.append(chosen)
        seen_structures.add(chosen.structure_key)
        seen_operator_sequences.add(chosen.operator_sequence)
        seen_top_operators.add(chosen.top_operator)
        seen_fraction_flags.add(chosen.requires_fraction)

    return selected


def _operation_matches_step(solution: Solution) -> bool:
    for step in solution.steps:
        if step.operator == "+":
            expected = step.left_value + step.right_value
        elif step.operator == "-":
            expected = step.left_value - step.right_value
        elif step.operator == "*":
            expected = step.left_value * step.right_value
        elif step.operator == "/":
            if step.right_value == 0:
                return False
            expected = step.left_value / step.right_value
        else:
            return False

        if expected != step.result:
            return False

    return True


def _make_records_for_problem(
    problem: Problem,
    *,
    split: str,
    rng: random.Random,
    solutions_per_problem: int,
    permutations_per_problem: int,
    prompt_templates: int,
    max_search_nodes: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    solver_limit = max(solutions_per_problem * 4, solutions_per_problem)
    solutions = solve_game24(
        problem.numbers,
        target=problem.target,
        max_solutions=solver_limit,
        max_search_nodes=max_search_nodes,
    )
    selected = _select_diverse_solutions(solutions, solutions_per_problem)

    if not selected:
        return [], {
            "solutions_found": len(solutions),
            "solutions_selected": 0,
            "skipped_inconsistent_records": 0,
            "records": 0,
        }

    permutations_for_problem = _unique_permutations(
        problem.numbers,
        limit=max(1, permutations_per_problem),
        rng=rng,
    )
    template_count = max(1, min(prompt_templates, 3))
    variants_per_solution = max(1, min(2, template_count, len(permutations_for_problem)))

    records: list[dict[str, Any]] = []
    seen_records: set[tuple[str, str, str]] = set()
    skipped_inconsistent_records = 0

    for solution_index, solution in enumerate(selected):
        for variant_index in range(variants_per_solution):
            if not _operation_matches_step(solution):
                skipped_inconsistent_records += 1
                continue

            permutation_index = (solution_index + variant_index) % len(permutations_for_problem)
            template_index = (solution_index + variant_index) % template_count
            numbers = permutations_for_problem[permutation_index]
            prompt = _prompt_from_template(numbers, problem.target, template_index)
            assistant_response = solution_to_response(solution, numbers, target=problem.target)

            record_key = (problem.problem_id, solution.expression, prompt)
            if record_key in seen_records:
                continue
            seen_records.add(record_key)

            records.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": assistant_response},
                    ],
                    "problem_id": problem.problem_id,
                    "numbers": list(numbers),
                    "target": problem.target,
                    "expression": solution.expression,
                    "requires_fraction": solution.requires_fraction,
                    "solution_index": solution_index,
                    "prompt_template": template_index,
                    "input_permutation_index": permutation_index,
                    "split": split,
                    "operator_sequence": " ".join(solution.operator_sequence),
                    "top_operator": solution.top_operator,
                }
            )

    return records, {
        "solutions_found": len(solutions),
        "solutions_selected": len(selected),
        "skipped_inconsistent_records": skipped_inconsistent_records,
        "records": len(records),
    }


def _check_no_overlap(name: str, left: set[str], right: set[str]) -> None:
    overlap = left & right
    if overlap:
        sample = sorted(overlap)[:10]
        raise ValueError(f"{name} overlap is not empty: count={len(overlap)}, sample={sample}")


def build_sft_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.solutions_per_problem <= 0:
        raise ValueError("--solutions-per-problem must be positive")
    if args.permutations_per_problem <= 0:
        raise ValueError("--permutations-per-problem must be positive")
    if args.prompt_templates <= 0:
        raise ValueError("--prompt-templates must be positive")
    if not 0 <= args.sft_val_ratio < 1:
        raise ValueError("--sft-val-ratio must be in [0, 1)")
    if args.max_search_nodes <= 0:
        raise ValueError("--max-search-nodes must be positive")

    processed_data_dir = Path(args.processed_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_problems = _read_required_split(processed_data_dir, "train.parquet")
    val_problems = _read_required_split(processed_data_dir, "val.parquet")
    test_problems = _read_required_split(processed_data_dir, "test.parquet")
    tot_problems = _read_required_split(processed_data_dir, "tot_hard100.parquet")

    train_ids = set(train_problems)
    project_val_ids = set(val_problems)
    project_test_ids = set(test_problems)
    tot_ids = set(tot_problems)

    if not train_ids:
        raise ValueError("train.parquet does not contain any unique problem IDs")

    rng = random.Random(args.seed)
    shuffled_train_ids = sorted(train_ids)
    rng.shuffle(shuffled_train_ids)

    if len(shuffled_train_ids) > 1 and args.sft_val_ratio > 0:
        sft_val_count = max(1, round(len(shuffled_train_ids) * args.sft_val_ratio))
        sft_val_count = min(sft_val_count, len(shuffled_train_ids) - 1)
    else:
        sft_val_count = 0

    sft_val_ids = set(shuffled_train_ids[:sft_val_count])
    sft_train_ids = train_ids - sft_val_ids

    _check_no_overlap("SFT train ID vs project val ID", sft_train_ids, project_val_ids)
    _check_no_overlap("SFT train ID vs project test ID", sft_train_ids, project_test_ids)
    _check_no_overlap("SFT train ID vs ToT-100 ID", sft_train_ids, tot_ids)
    _check_no_overlap("SFT train ID vs SFT val ID", sft_train_ids, sft_val_ids)

    split_inputs = {
        "sft_train": [train_problems[problem_id] for problem_id in sorted(sft_train_ids)],
        "sft_val": [train_problems[problem_id] for problem_id in sorted(sft_val_ids)],
    }

    all_records: dict[str, list[dict[str, Any]]] = {"sft_train": [], "sft_val": []}
    per_split_stats: dict[str, dict[str, int]] = {}

    for split, problems in split_inputs.items():
        solved_problems = 0
        unsolved_problems = 0
        solutions_found = 0
        solutions_selected = 0
        skipped_inconsistent_records = 0

        for problem in problems:
            records, problem_stats = _make_records_for_problem(
                problem,
                split=split,
                rng=rng,
                solutions_per_problem=args.solutions_per_problem,
                permutations_per_problem=args.permutations_per_problem,
                prompt_templates=args.prompt_templates,
                max_search_nodes=args.max_search_nodes,
            )
            if records:
                solved_problems += 1
            else:
                unsolved_problems += 1

            solutions_found += problem_stats["solutions_found"]
            solutions_selected += problem_stats["solutions_selected"]
            skipped_inconsistent_records += problem_stats["skipped_inconsistent_records"]
            all_records[split].extend(records)

        per_split_stats[split] = {
            "input_problem_ids": len(problems),
            "solved_problem_ids": solved_problems,
            "unsolved_problem_ids": unsolved_problems,
            "solutions_found": solutions_found,
            "solutions_selected": solutions_selected,
            "skipped_inconsistent_records": skipped_inconsistent_records,
            "records": len(all_records[split]),
        }

    if not all_records["sft_train"]:
        raise ValueError("no SFT training records were generated")
    if sft_val_ids and not all_records["sft_val"]:
        raise ValueError("SFT validation split was requested but no validation records were generated")

    random.Random(args.seed).shuffle(all_records["sft_train"])
    random.Random(args.seed).shuffle(all_records["sft_val"])

    pd.DataFrame(all_records["sft_train"]).to_parquet(output_dir / "sft_train.parquet", index=False)
    pd.DataFrame(all_records["sft_val"]).to_parquet(output_dir / "sft_val.parquet", index=False)

    stats = {
        "inputs": {
            "train_problem_ids": len(train_ids),
            "project_val_problem_ids": len(project_val_ids),
            "project_test_problem_ids": len(project_test_ids),
            "tot_hard100_problem_ids": len(tot_ids),
        },
        "overlap_checks": {
            "sft_train_vs_project_val": 0,
            "sft_train_vs_project_test": 0,
            "sft_train_vs_tot_hard100": 0,
            "sft_train_vs_sft_val": 0,
        },
        "settings": {
            "seed": args.seed,
            "solutions_per_problem": args.solutions_per_problem,
            "permutations_per_problem": args.permutations_per_problem,
            "prompt_templates": args.prompt_templates,
            "sft_val_ratio": args.sft_val_ratio,
            "max_search_nodes": args.max_search_nodes,
            "rows_shuffled": True,
        },
        "splits": per_split_stats,
        "expected_scale_note": (
            "Default generation keeps up to two prompt/permutation variants per selected solution, "
            "so record count is roughly solved_problem_ids * solutions_per_problem * 2."
        ),
    }

    (output_dir / "sft_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact-solver SFT data from prepared Game24 train IDs.")
    parser.add_argument("--processed-data-dir", default="data/game24")
    parser.add_argument("--output-dir", default="data/game24-sft")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--solutions-per-problem", type=int, default=8)
    parser.add_argument("--permutations-per-problem", type=int, default=4)
    parser.add_argument("--prompt-templates", type=int, default=3)
    parser.add_argument("--sft-val-ratio", type=float, default=0.05)
    parser.add_argument("--max-search-nodes", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    build_sft_data(parse_args())


if __name__ == "__main__":
    main()
