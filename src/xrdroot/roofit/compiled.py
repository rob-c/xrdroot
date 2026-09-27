"""A formula compiled to nested Python closures, for evaluating it a million times cheaply.

:mod:`xrdroot.formula` evaluates a tree's expression over whole columns,
and its per-call bookkeeping - which columns, what shapes, which entries
are valid - is nothing against a column of a million entries, but it is
everything against the one point a sampler asks for at a time. RooFit
evaluates its formulas at single points constantly - drawing events,
sampling curves, integrating - so a :class:`Compiled` formula is the same
parsed tree (:func:`xrdroot.formula.parser.parse`, so the language is the
same), each node made once into a closure over NumPy, evaluated with the
operators of :mod:`xrdroot.formula.ops`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..formula import ops
from ..formula.functions import CASTS, FUNCTIONS
from ..formula.names import Names
from ..formula.nodes import Binary, Call, Cast, Node, Number, Ref, Ternary, Unary
from ..formula.parser import parse

__all__ = ["Compiled"]

Values = dict[str, Any]
Closure = Callable[[Values], Any]


def _number(node: Number) -> Closure:
    value = np.float64(node.value)
    return lambda values: value


def _ref(node: Ref) -> Closure:
    column = node.column
    return lambda values: values[column]


def _unary(node: Unary) -> Closure:
    operand, op = _compile(node.operand), node.op
    return lambda values: ops.unary(op, np.asarray(operand(values)))


def _binary(node: Binary) -> Closure:
    left, right, op = _compile(node.left), _compile(node.right), node.op
    return lambda values: ops.binary(op, np.asarray(left(values)), np.asarray(right(values)))


def _ternary(node: Ternary) -> Closure:
    test, then, otherwise = _compile(node.condition), _compile(node.then), _compile(node.otherwise)
    return lambda values: np.where(
        ops.truth(np.asarray(test(values))), then(values), otherwise(values)
    )


def _call(node: Call) -> Closure:
    function = FUNCTIONS[node.name].apply
    args = [_compile(arg) for arg in node.args]
    return lambda values: function(*(np.asarray(a(values)) for a in args))


def _cast(node: Cast) -> Closure:
    dtype, operand = CASTS[node.type], _compile(node.operand)
    return lambda values: ops.cast(np.asarray(operand(values)), dtype, "a cast")[0]


#: How each kind of node is compiled.
COMPILERS: dict[type, Callable[[Any], Closure]] = {
    Number: _number,
    Ref: _ref,
    Unary: _unary,
    Binary: _binary,
    Ternary: _ternary,
    Call: _call,
    Cast: _cast,
}


def _compile(node: Node) -> Closure:
    make = COMPILERS.get(type(node))
    if make is None:
        raise UnsupportedFeatureError(
            f"a RooFit formula cannot use a {type(node).__name__}: that is TTree::Draw's language, "
            "not a function of RooFit's variables"
        )
    return make(node)


class Compiled:
    """An expression over named values, made once into closures."""

    def __init__(self, text: str, names: list[str]) -> None:
        self._closure = _compile(parse(text, Names(names)))

    def __call__(self, values: Values) -> Any:
        return self._closure(values)
