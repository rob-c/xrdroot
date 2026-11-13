"""The integer arithmetic of ``#if``: what the condition of a conditional section is worth.

By the time an expression gets here ``defined`` has been answered, macros
expanded and every identifier left over made ``0``, as the C standard says;
what remains is numbers, character literals and C's operators, evaluated
with C's precedence.
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import Refusal, Where
from .literals import number
from .tokens import Token, unescape

__all__ = ["evaluate"]

#: The binary operators by precedence, loosest first, and what each computes.
LEVELS: list[dict[str, Callable[[int, int], int]]] = [
    {"||": lambda a, b: int(bool(a) or bool(b))},
    {"&&": lambda a, b: int(bool(a) and bool(b))},
    {"|": lambda a, b: a | b},
    {"^": lambda a, b: a ^ b},
    {"&": lambda a, b: a & b},
    {"==": lambda a, b: int(a == b), "!=": lambda a, b: int(a != b)},
    {
        "<": lambda a, b: int(a < b),
        ">": lambda a, b: int(a > b),
        "<=": lambda a, b: int(a <= b),
        ">=": lambda a, b: int(a >= b),
    },
    {"<<": lambda a, b: a << b, ">>": lambda a, b: a >> b},
    {"+": lambda a, b: a + b, "-": lambda a, b: a - b},
    {"*": lambda a, b: a * b, "/": lambda a, b: _divide(a, b), "%": lambda a, b: _rest(a, b)},
]

#: The prefix operators.
UNARY: dict[str, Callable[[int], int]] = {
    "!": lambda a: int(not a),
    "~": lambda a: ~a,
    "-": lambda a: -a,
    "+": lambda a: a,
}


def _divide(a: int, b: int) -> int:
    # A compiler rejects a division by zero in #if; nothing a macro means depends on it.
    if b == 0:
        return 0
    quotient = abs(a) // abs(b)
    return quotient if (a < 0) == (b < 0) else -quotient


def _rest(a: int, b: int) -> int:
    return a - b * _divide(a, b)


class _Reader:
    def __init__(self, tokens: list[Token], where: Where) -> None:
        self.tokens = tokens
        self.at = 0
        self.where = where

    def peek(self) -> Token | None:
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self) -> Token:
        token = self.peek()
        if token is None:
            raise Refusal("an #if condition ends before its expression does", self.where)
        self.at += 1
        return token

    def next_is(self, *texts: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == "op" and token.text in texts


def _conditional(reader: _Reader) -> int:
    condition = _binary(reader, 0)
    if not reader.next_is("?"):
        return condition
    reader.take()
    yes = _conditional(reader)
    if not reader.next_is(":"):
        raise Refusal("an #if condition has a ? without its :", reader.where)
    reader.take()
    no = _conditional(reader)
    return yes if condition else no


def _binary(reader: _Reader, level: int) -> int:
    if level == len(LEVELS):
        return _unary(reader)
    value = _binary(reader, level + 1)
    operators = LEVELS[level]
    while reader.next_is(*operators):
        operator = operators[reader.take().text]
        value = operator(value, _binary(reader, level + 1))
    return value


def _unary(reader: _Reader) -> int:
    token = reader.take()
    if token.kind == "op" and token.text in UNARY:
        return UNARY[token.text](_unary(reader))
    if token.is_("("):
        value = _conditional(reader)
        if not reader.next_is(")"):
            raise Refusal("an #if condition opens a bracket it does not close", reader.where)
        reader.take()
        return value
    return _atom(token, reader.where)


def _atom(token: Token, where: Where) -> int:
    if token.kind == "num":
        value, _ = number(token.text, where)
        return int(value)
    if token.kind == "chr":
        return ord(unescape(token.text)[:1] or "\0")
    raise Refusal(f"{token.text!r} cannot be part of an #if condition", where)


def evaluate(tokens: list[Token], where: Where) -> int:
    """The value of an ``#if`` condition whose names have all been replaced by numbers."""
    if not tokens:
        raise Refusal("an #if has no condition", where)
    reader = _Reader(tokens, where)
    value = _conditional(reader)
    if reader.peek() is not None:
        raise Refusal(f"an #if condition has {reader.take().text!r} left over", where)
    return value
