"""verl custom reward entry point for Game of 24."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from game24.verifier import verify_solution

CORRECT_REWARD = 1.0
WRONG_VALUE_REWARD = 0.05
TAG_ONLY_REWARD = 0.01
NO_REWARD = 0.0


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
) -> float:
    """Return the scalar reward expected by verl custom reward functions."""

    del data_source, extra_info, kwargs

    try:
        truth = _load_ground_truth(ground_truth)
        numbers = truth["numbers"]
        target = int(truth.get("target", 24))
        result = verify_solution(solution_str, numbers, target=target)
    except Exception:
        return NO_REWARD

    if result.is_correct:
        return CORRECT_REWARD

    if result.has_answer_tag and result.parse_valid and result.operators_valid and result.numbers_valid:
        return WRONG_VALUE_REWARD

    if result.has_answer_tag:
        return TAG_ONLY_REWARD

    return NO_REWARD
