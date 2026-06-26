"""Metric helpers shared by evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from game24.verifier import VerificationResult


@dataclass(frozen=True)
class GenerationRecord:
    response: str
    verification: VerificationResult
    truncated: bool = False


def _mean(values: Sequence[bool | int | float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / len(values)


def summarize_generation_groups(
    groups: Sequence[Sequence[GenerationRecord]],
    *,
    unsolvable: bool = False,
) -> dict[str, float | int | bool]:
    """Summarize generations grouped by problem."""

    flat = [record for group in groups for record in group]
    first_records = [group[0] for group in groups if group]
    n = max((len(group) for group in groups), default=0)

    summary: dict[str, float | int | bool] = {
        "num_problems": len(groups),
        "num_generations": len(flat),
        "generations_per_problem": n,
        "unsolvable": unsolvable,
        "greedy_pass_at_1": _mean([record.verification.is_correct for record in first_records]),
        "sampling_pass_at_n": _mean([any(record.verification.is_correct for record in group) for group in groups]),
        "strict_exact_accuracy": _mean([record.verification.is_correct for record in first_records]),
        "format_valid_rate": _mean([record.verification.format_valid for record in flat]),
        "parse_valid_rate": _mean([record.verification.parse_valid for record in flat]),
        "correct_number_multiset_rate": _mean([record.verification.numbers_valid for record in flat]),
        "average_response_length": _mean([len(record.response) for record in flat]),
        "response_truncation_rate": _mean([record.truncated for record in flat]),
    }

    if unsolvable:
        looks_like_answer = [
            record.verification.format_valid
            and record.verification.parse_valid
            and not record.verification.is_correct
            for record in flat
        ]
        summary["failed_answer_hallucination_rate"] = _mean(looks_like_answer)
        summary["verifier_correct_rate"] = _mean([record.verification.is_correct for record in flat])

    return summary
