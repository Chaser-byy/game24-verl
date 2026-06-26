"""Prompt construction for the Game of 24 task."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral


def _normalize_numbers(numbers: Sequence[int]) -> tuple[int, int, int, int]:
    if len(numbers) != 4:
        raise ValueError(f"expected exactly 4 numbers, got {len(numbers)}")

    normalized: list[int] = []
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, Integral):
            raise ValueError(f"numbers must be non-boolean integers, got {number!r}")
        normalized.append(int(number))
    return tuple(normalized)  # type: ignore[return-value]


def build_user_prompt(numbers: Sequence[int], target: int = 24) -> str:
    """Build the single user message used by training and evaluation."""

    nums = _normalize_numbers(numbers)
    numbers_text = ", ".join(str(number) for number in nums)
    return (
        "Solve this Game of 24 puzzle.\n\n"
        f"Numbers: {numbers_text}\n"
        f"Target: {target}\n\n"
        "Rules:\n"
        "- Use each input number exactly once.\n"
        "- Only use +, -, *, /, and parentheses.\n"
        "- Intermediate fractions are allowed.\n"
        "- Do not concatenate digits or create new numbers.\n"
        f"- The final expression must equal exactly {target}.\n\n"
        "Return your response in exactly this XML format:\n"
        "<think>\n"
        "Your reasoning may go here.\n"
        "</think>\n"
        "<answer>\n"
        "final expression only\n"
        "</answer>\n\n"
        "The verifier will only check the expression inside <answer>."
    )


def build_chat_prompt(numbers: Sequence[int], target: int = 24) -> list[dict[str, str]]:
    """Return a verl-compatible chat prompt with one user turn."""

    return [{"role": "user", "content": build_user_prompt(numbers, target=target)}]
