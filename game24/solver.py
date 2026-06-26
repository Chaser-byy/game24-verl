"""Exact Game of 24 solver used for SFT trajectory generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

from game24.verifier import verify_solution


@dataclass(frozen=True)
class OperationStep:
    left_value: Fraction
    right_value: Fraction
    operator: str
    result: Fraction
    remaining: tuple[Fraction, ...]
    left_expression: str
    right_expression: str
    result_expression: str


@dataclass(frozen=True)
class Solution:
    expression: str
    value: Fraction
    steps: tuple[OperationStep, ...]
    structure_key: str
    operator_sequence: tuple[str, ...]
    top_operator: str
    requires_fraction: bool


@dataclass(frozen=True)
class _Term:
    value: Fraction
    expression: str
    structure_key: str
    root_operator: str | None = None


def _normalize_numbers(numbers: Sequence[int]) -> tuple[int, int, int, int]:
    if len(numbers) != 4:
        raise ValueError(f"expected exactly 4 numbers, got {len(numbers)}")

    normalized: list[int] = []
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, Integral):
            raise ValueError(f"numbers must be non-boolean integers, got {number!r}")
        normalized.append(int(number))
    return tuple(normalized)  # type: ignore[return-value]


def _fraction_key(value: Fraction) -> tuple[int, int]:
    return value.numerator, value.denominator


def _term_sort_key(term: _Term) -> tuple[int, int, str]:
    return (*_fraction_key(term.value), term.structure_key)


def _state_key(terms: Sequence[_Term]) -> tuple[tuple[int, int, str], ...]:
    return tuple(sorted(_term_sort_key(term) for term in terms))


def _wraps_entire_expression(expression: str) -> bool:
    if not (expression.startswith("(") and expression.endswith(")")):
        return False

    depth = 0
    for index, char in enumerate(expression):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(expression) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _strip_outer_parentheses(expression: str) -> str:
    while _wraps_entire_expression(expression):
        expression = expression[1:-1]
    return expression


def _operation_result(left: Fraction, right: Fraction, operator: str) -> Fraction | None:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        if right == 0:
            return None
        return left / right
    raise ValueError(f"unsupported operator: {operator}")


def _combine_terms(left: _Term, right: _Term, rest: Sequence[_Term]) -> list[tuple[_Term, OperationStep]]:
    combinations: list[tuple[_Term, OperationStep]] = []
    operation_inputs: list[tuple[_Term, _Term, str]] = []

    ordered_left, ordered_right = sorted((left, right), key=_term_sort_key)
    operation_inputs.append((ordered_left, ordered_right, "+"))
    operation_inputs.append((ordered_left, ordered_right, "*"))
    operation_inputs.append((left, right, "-"))
    operation_inputs.append((right, left, "-"))
    operation_inputs.append((left, right, "/"))
    operation_inputs.append((right, left, "/"))

    seen: set[tuple[int, int, str]] = set()
    for op_left, op_right, operator in operation_inputs:
        value = _operation_result(op_left.value, op_right.value, operator)
        if value is None:
            continue

        if operator in {"+", "*"}:
            child_keys = sorted((op_left.structure_key, op_right.structure_key))
            structure_key = f"({child_keys[0]}{operator}{child_keys[1]})"
        else:
            structure_key = f"({op_left.structure_key}{operator}{op_right.structure_key})"

        dedupe_key = (*_fraction_key(value), structure_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        expression = f"({op_left.expression}{operator}{op_right.expression})"
        term = _Term(
            value=value,
            expression=expression,
            structure_key=structure_key,
            root_operator=operator,
        )
        remaining = tuple(item.value for item in (*rest, term))
        step = OperationStep(
            left_value=op_left.value,
            right_value=op_right.value,
            operator=operator,
            result=value,
            remaining=remaining,
            left_expression=op_left.expression,
            right_expression=op_right.expression,
            result_expression=expression,
        )
        combinations.append((term, step))

    return combinations


def _requires_fraction(steps: Sequence[OperationStep]) -> bool:
    for step in steps:
        if (
            step.left_value.denominator != 1
            or step.right_value.denominator != 1
            or step.result.denominator != 1
        ):
            return True
    return False


def solve_game24(
    numbers: Sequence[int],
    target: int = 24,
    *,
    max_solutions: int = 64,
    max_search_nodes: int = 20_000,
) -> list[Solution]:
    """Return exact solutions and derivation steps for one Game of 24 puzzle."""

    normalized = _normalize_numbers(numbers)
    if max_solutions <= 0 or max_search_nodes <= 0:
        return []

    target_value = Fraction(target, 1)
    initial_terms = tuple(
        _Term(value=Fraction(number, 1), expression=str(number), structure_key=f"N{number}")
        for number in normalized
    )

    solutions: list[Solution] = []
    seen_states: set[tuple[tuple[int, int, str], ...]] = set()
    seen_final_keys: set[str] = set()
    search_nodes = 0

    def search(terms: tuple[_Term, ...], history: tuple[OperationStep, ...]) -> None:
        nonlocal search_nodes

        if len(solutions) >= max_solutions or search_nodes >= max_search_nodes:
            return

        key = _state_key(terms)
        if key in seen_states:
            return
        seen_states.add(key)
        search_nodes += 1

        if len(terms) == 1:
            term = terms[0]
            if term.value != target_value or term.structure_key in seen_final_keys:
                return

            expression = _strip_outer_parentheses(term.expression)
            verification = verify_solution(f"<answer>{expression}</answer>", normalized, target=target)
            if not verification.is_correct:
                return

            seen_final_keys.add(term.structure_key)
            solutions.append(
                Solution(
                    expression=expression,
                    value=term.value,
                    steps=history,
                    structure_key=term.structure_key,
                    operator_sequence=tuple(step.operator for step in history),
                    top_operator=term.root_operator or "",
                    requires_fraction=_requires_fraction(history),
                )
            )
            return

        for left_index in range(len(terms)):
            for right_index in range(left_index + 1, len(terms)):
                left = terms[left_index]
                right = terms[right_index]
                rest = tuple(
                    term for index, term in enumerate(terms) if index not in {left_index, right_index}
                )
                for new_term, step in _combine_terms(left, right, rest):
                    search((*rest, new_term), (*history, step))

    search(initial_terms, ())
    return solutions
