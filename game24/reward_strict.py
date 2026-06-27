"""Strict 0/1 reward entry point for improved GRPO experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from game24.verifier import verify_solution


def _load_ground_truth(ground_truth: str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(ground_truth, str):
        loaded = json.loads(ground_truth)
    else:
        loaded = ground_truth

    if not isinstance(loaded, Mapping):
        raise ValueError("ground_truth must be a JSON object or mapping")
    return loaded


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | Mapping[str, Any],
    extra_info: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | int | str | None]:
    """Return strict training reward plus diagnostic fields for verl.

    The optimization reward is only ``score`` and is exactly 1.0 for strict
    correctness, otherwise 0.0. Other fields are diagnostics consumed by the
    DAPO reward manager and trainer metrics.
    """

    del data_source, extra_info, kwargs

    try:
        truth = _load_ground_truth(ground_truth)
        numbers = truth["numbers"]
        target = int(truth.get("target", 24))
        result = verify_solution(solution_str, numbers, target=target)
    except Exception as exc:
        return {
            "score": 0.0,
            "acc": 0.0,
            "format_valid": 0.0,
            "parse_valid": 0.0,
            "number_usage_valid": 0.0,
            "error_reason": f"reward_exception:{type(exc).__name__}",
        }

    score = 1.0 if result.is_correct else 0.0
    return {
        "score": score,
        "acc": score,
        "format_valid": float(result.format_valid),
        "parse_valid": float(result.parse_valid),
        "number_usage_valid": float(result.numbers_valid),
        "error_reason": result.error_reason,
    }
