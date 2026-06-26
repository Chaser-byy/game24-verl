"""Convert exact solver paths into compact SFT answers."""

from __future__ import annotations

from fractions import Fraction

from game24.solver import OperationStep, Solution
from game24.verifier import verify_solution


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_operand(value: Fraction) -> str:
    text = format_fraction(value)
    if value.denominator == 1:
        return text
    return f"({text})"


def _format_remaining(values: tuple[Fraction, ...]) -> str:
    return ", ".join(format_fraction(value) for value in values)


def step_to_sentence(step: OperationStep) -> str:
    left = format_operand(step.left_value)
    right = format_operand(step.right_value)
    result = format_fraction(step.result)
    remaining = _format_remaining(step.remaining)
    return (
        f"Combine {left} and {right}: {left} {step.operator} {right} = {result}. "
        f"Remaining: {remaining}."
    )


def solution_to_response(solution: Solution, numbers: list[int] | tuple[int, ...], target: int = 24) -> str:
    """Build the assistant XML response and verify its final answer."""

    think = "\n".join(step_to_sentence(step) for step in solution.steps)
    response = f"<think>\n{think}\n</think>\n<answer>\n{solution.expression}\n</answer>"

    verification = verify_solution(response, numbers, target=target)
    if not verification.is_correct:
        raise ValueError(
            f"solver trajectory failed verifier for numbers={numbers}, expression={solution.expression!r}"
        )

    return response
