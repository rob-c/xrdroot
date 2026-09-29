"""A CFITSIO row filter - ``[DATAMAX > 2e-15]`` - as the rows of a table it keeps.

The filter is an expression of the table's columns, named without regard to
case, and of numbers and quoted strings: arithmetic, comparisons (``==``,
``!=``, ``<`` ... or FORTRAN's ``.eq.`` ...), ``&&``, ``||`` and ``!``. It
is parsed here by precedence, never evaluated as Python.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError

__all__ = ["keep"]

#: The tokens of a filter: a number, a quoted string, a name, or an operator.
TOKEN = re.compile(r"""\s*(?:(\d+\.?\d*(?:[eEdD][+-]?\d+)?|\.\d+(?:[eEdD][+-]?\d+)?)"""
                   r"""|('[^']*'|"[^"]*")|([A-Za-z_][\w]*)"""
                   r"""|(\.\w+\.|&&|\|\||==|!=|<=|>=|[-+*/<>!()]))""")  # fmt: skip

#: FORTRAN's spelling of each operator, as C spells it.
FORTRAN = {".eq.": "==", ".ne.": "!=", ".lt.": "<", ".le.": "<=", ".gt.": ">", ".ge.": ">=",
           ".and.": "&&", ".or.": "||", ".not.": "!"}  # fmt: skip

#: Each binary operator's precedence and what it does; higher binds tighter.
BINARY: dict[str, tuple[int, Callable[[Any, Any], Any]]] = {
    "||": (1, np.logical_or), "&&": (2, np.logical_and),
    "==": (3, operator.eq), "!=": (3, operator.ne), "<": (3, operator.lt),
    "<=": (3, operator.le), ">": (3, operator.gt), ">=": (3, operator.ge),
    "+": (4, operator.add), "-": (4, operator.sub), "*": (5, operator.mul),
    "/": (5, operator.truediv),
}  # fmt: skip


def _tokens(text: str) -> list[tuple[str, str]]:
    """The filter as ``(kind, text)`` pairs: ``number``, ``string``, ``name`` or ``op``."""
    found, at = [], 0
    while text[at:].strip():
        match = TOKEN.match(text, at)
        if match is None:
            raise UnsupportedFeatureError(f"xrdroot cannot read the row filter {text!r} "
                                          f"from {text[at:].strip()!r} on.")  # fmt: skip
        kind = ("number", "string", "name", "op")[match.lastindex - 1]  # type: ignore[operator]
        word = match.group(match.lastindex or 0)
        found.append((kind, FORTRAN.get(word.lower(), word)))
        at = match.end()
    return found


class _Parser:
    """Precedence climbing over the tokens, a column being its array of values."""

    def __init__(self, tokens: list[tuple[str, str]], values: dict[str, Any]) -> None:
        self.tokens, self.values, self.at = tokens, values, 0

    def peek(self) -> str:
        return self.tokens[self.at][1] if self.at < len(self.tokens) else ""

    def take(self) -> tuple[str, str]:
        if self.at >= len(self.tokens):
            raise UnsupportedFeatureError("This row filter ends where a value was expected.")
        self.at += 1
        return self.tokens[self.at - 1]

    def expression(self, lowest: int = 1) -> Any:
        left = self.unary()
        while self.peek() in BINARY and BINARY[self.peek()][0] >= lowest:
            level, apply = BINARY[self.take()[1]]
            left = apply(left, self.expression(level + 1))
        return left

    def unary(self) -> Any:
        kind, word = self.take()
        if word in ("!", "-"):
            inner = self.unary()
            return np.logical_not(inner) if word == "!" else -inner
        if word == "(":
            inner = self.expression()
            self.take()
            return inner
        return self.atom(kind, word)

    def atom(self, kind: str, word: str) -> Any:
        if kind == "number":
            return float(word.replace("d", "e").replace("D", "E"))
        if kind == "string":
            return word[1:-1]
        if word.upper() not in self.values:
            raise UnsupportedFeatureError(f"This row filter names {word!r}, which is not a "
                                          "column of the table.")  # fmt: skip
        return self.values[word.upper()]


def keep(text: str, values: dict[str, Any], rows: int) -> np.ndarray[Any, Any]:
    """Which of ``rows`` rows the filter keeps, given every column by its name in capitals."""
    parser = _Parser(_tokens(text), values)
    kept = parser.expression()
    if parser.at != len(parser.tokens):
        raise UnsupportedFeatureError(f"xrdroot cannot read the row filter {text!r} past "
                                      f"{parser.tokens[parser.at][1]!r}.")  # fmt: skip
    return np.broadcast_to(np.asarray(kept, dtype=bool), (rows,)).copy()
