"""Strict Game of 24 verifier without eval or symbolic dependencies."""

from __future__ import annotations

import ast
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Integral

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
EXPRESSION_CHARS_RE = re.compile(r"^[0-9+\-*/()\s]+$")
MAX_EXPRESSION_CHARS = 128
MAX_AST_NODES = 64
TARGET_VALUE = 24

ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Constant,
)


@dataclass(frozen=True)
class VerificationResult:
    has_answer_tag: bool
    format_valid: bool
    parse_valid: bool
    operators_valid: bool
    numbers_valid: bool
    value: Fraction | None
    equals_target: bool
    is_correct: bool
    error_reason: str | None
    expression: str | None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["value"] = str(self.value) if self.value is not None else None
        return data


class VerificationError(ValueError):
    """Raised internally for invalid expressions."""


def _failure(
    *,
    has_answer_tag: bool,
    format_valid: bool = False,
    parse_valid: bool = False,
    operators_valid: bool = False,
    numbers_valid: bool = False,
    value: Fraction | None = None,
    equals_target: bool = False,
    error_reason: str,
    expression: str | None,
) -> VerificationResult:
    return VerificationResult(
        has_answer_tag=has_answer_tag,
        format_valid=format_valid,
        parse_valid=parse_valid,
        operators_valid=operators_valid,
        numbers_valid=numbers_valid,
        value=value,
        equals_target=equals_target,
        is_correct=False,
        error_reason=error_reason,
        expression=expression,
    )


def _normalize_numbers(numbers: Sequence[int]) -> tuple[int, int, int, int]:
    if len(numbers) != 4:
        raise ValueError(f"expected exactly 4 input numbers, got {len(numbers)}")

    normalized: list[int] = []
    for number in numbers:
        if isinstance(number, bool) or not isinstance(number, Integral):
            raise ValueError(f"input numbers must be non-boolean integers, got {number!r}")
        normalized.append(int(number))
    return tuple(normalized)  # type: ignore[return-value]


def _eval_node(node: ast.AST) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise VerificationError("invalid_constant")
        return Fraction(node.value, 1), [node.value]

    if isinstance(node, ast.BinOp):
        left_value, left_numbers = _eval_node(node.left)
        right_value, right_numbers = _eval_node(node.right)

        if isinstance(node.op, ast.Add):
            value = left_value + right_value
        elif isinstance(node.op, ast.Sub):
            value = left_value - right_value
        elif isinstance(node.op, ast.Mult):
            value = left_value * right_value
        elif isinstance(node.op, ast.Div):
            if right_value == 0:
                raise VerificationError("division_by_zero")
            value = left_value / right_value
        else:
            raise VerificationError(f"disallowed_operator:{type(node.op).__name__}")

        return value, left_numbers + right_numbers

    raise VerificationError(f"disallowed_ast_node:{type(node).__name__}")


def _extract_single_answer(solution_str: str) -> tuple[bool, str | None, str | None]:
    open_count = solution_str.count("<answer>")
    close_count = solution_str.count("</answer>")
    has_answer_tag = bool(open_count or close_count)
    if open_count == 0 and close_count == 0:
        return has_answer_tag, None, "missing_answer_tag"
    if open_count != 1 or close_count != 1:
        return has_answer_tag, None, "multiple_answer_tags"

    matches = ANSWER_RE.findall(solution_str)
    if len(matches) != 1:
        return has_answer_tag, None, "malformed_answer_tag"

    expression = matches[0].strip()
    if not expression:
        return has_answer_tag, expression, "empty_answer"
    if len(expression) > MAX_EXPRESSION_CHARS:
        return has_answer_tag, expression, "expression_too_long"
    if not EXPRESSION_CHARS_RE.fullmatch(expression):
        return has_answer_tag, expression, "invalid_expression_characters"
    return has_answer_tag, expression, None


def verify_solution(
    solution_str: str,
    numbers: Sequence[int],
    target: int = TARGET_VALUE,
    *,
    max_ast_nodes: int = MAX_AST_NODES,
) -> VerificationResult:
    """Verify a model response for one Game of 24 puzzle."""

    expected_numbers = _normalize_numbers(numbers)
    has_answer_tag, expression, format_error = _extract_single_answer(solution_str)
    if format_error is not None:
        return _failure(
            has_answer_tag=has_answer_tag,
            error_reason=format_error,
            expression=expression,
        )

    assert expression is not None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return _failure(
            has_answer_tag=True,
            format_valid=True,
            error_reason="syntax_error",
            expression=expression,
        )

    nodes = list(ast.walk(tree))
    if len(nodes) > max_ast_nodes:
        return _failure(
            has_answer_tag=True,
            format_valid=True,
            parse_valid=True,
            error_reason="too_many_ast_nodes",
            expression=expression,
        )

    for node in nodes:
        if not isinstance(node, ALLOWED_NODE_TYPES):
            return _failure(
                has_answer_tag=True,
                format_valid=True,
                parse_valid=True,
                error_reason=f"disallowed_ast_node:{type(node).__name__}",
                expression=expression,
            )

    try:
        value, used_numbers = _eval_node(tree.body)
    except VerificationError as exc:
        reason = str(exc)
        return _failure(
            has_answer_tag=True,
            format_valid=True,
            parse_valid=True,
            operators_valid=reason == "division_by_zero",
            error_reason=reason,
            expression=expression,
        )

    numbers_valid = Counter(used_numbers) == Counter(expected_numbers)
    equals_target = value == Fraction(target, 1)
    is_correct = numbers_valid and equals_target

    return VerificationResult(
        has_answer_tag=True,
        format_valid=True,
        parse_valid=True,
        operators_valid=True,
        numbers_valid=numbers_valid,
        value=value,
        equals_target=equals_target,
        is_correct=is_correct,
        error_reason=None if is_correct else "incorrect",
        expression=expression,
    )
