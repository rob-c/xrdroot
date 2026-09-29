"""``TGeoCompositeShape``'s Boolean expressions: ``"(A:t1+B:t2)-C"`` read into a tree.

A composite shape names other shapes, each optionally placed by a named
matrix after a colon, joined by ``+`` (union), ``-`` (subtraction) and
``*`` (intersection), with parentheses. Intersection binds tighter than the
other two, which read left to right. Painted in a pad, a composite is its
components each painted where its matrix puts it, as
``TGeoBoolNode::Paint`` does; its box is the union of theirs, the left's
for a subtraction, and their overlap for an intersection.
"""

from __future__ import annotations

import re
from typing import Any, Union

__all__ = ["parse", "leaves", "Leaf", "Node"]

#: A shape's name, or a matrix's, as ``TGeoCompositeShape`` reads one.
_NAME = r"[A-Za-z_][\w.]*"
_TOKEN = re.compile(rf"\s*(?:(?P<leaf>{_NAME}(?::{_NAME})?)|(?P<op>[-+*()]))")


class Leaf(tuple):  # type: ignore[type-arg]
    """A component: the shape's name and its matrix's (empty for none)."""

    __slots__ = ()

    def __new__(cls, shape: str, matrix: str = "") -> Leaf:
        return super().__new__(cls, (shape, matrix))


#: An operation: its operator, and the trees on each side of it.
Node = tuple[str, Any, Any]
Tree = Union[Leaf, Node]


def _tokens(expression: str) -> list[str]:
    found, at, text = [], 0, expression.strip()
    while at < len(text):
        match = _TOKEN.match(text, at)
        if match is None:
            raise ValueError(f"the composite shape expression {expression!r} cannot be read "
                             f"at {text[at:]!r}")  # fmt: skip
        found.append(match.group("leaf") or match.group("op"))
        at = match.end()
    return found


class _Reader:
    def __init__(self, expression: str) -> None:
        self.tokens = _tokens(expression)
        self.expression = expression

    def _next(self) -> str:
        if not self.tokens:
            raise ValueError(f"the composite shape expression {self.expression!r} ends too soon")
        return self.tokens.pop(0)

    def term(self) -> Tree:
        token = self._next()
        if token == "(":
            inside = self.sum()
            if self._next() != ")":
                raise ValueError(f"the composite shape expression {self.expression!r} "
                                 f"has an unclosed parenthesis")  # fmt: skip
            return inside
        if token in "+-*)":
            raise ValueError(f"the composite shape expression {self.expression!r} has "
                             f"{token!r} where a shape should be")  # fmt: skip
        return Leaf(*token.split(":", 1))

    def product(self) -> Tree:
        tree = self.term()
        while self.tokens and self.tokens[0] == "*":
            self.tokens.pop(0)
            tree = ("*", tree, self.term())
        return tree

    def sum(self) -> Tree:
        tree = self.product()
        while self.tokens and self.tokens[0] in "+-":
            tree = (self.tokens.pop(0), tree, self.product())
        return tree


def parse(expression: str) -> Tree:
    """The expression as a tree of :class:`Leaf` components and ``(op, left, right)``."""
    reader = _Reader(expression)
    tree = reader.sum()
    if reader.tokens:
        raise ValueError(f"the composite shape expression {expression!r} has "
                         f"{reader.tokens[0]!r} left over")  # fmt: skip
    return tree


def leaves(tree: Tree) -> list[Leaf]:
    """Every component, left to right: what painting the composite paints."""
    if isinstance(tree, Leaf):
        return [tree]
    return leaves(tree[1]) + leaves(tree[2])
