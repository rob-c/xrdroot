"""Nullary random expressions share one stream, drawn in entry order."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...cint import nodes as cxx

#: These methods return doubles and only advance their generator's state.
_RANDOM_METHODS = frozenset({"Rndm", "Uniform", "Gaus", "Exp", "Landau", "PoissonD", "BreitWigner"})


class _RandomDraw:
    """A translated random call stays nullary so neighboring defines run in entry order."""

    _rdf_nullary = True
    _rdf_return_type = "double"

    def __init__(self, function: Any, method: str, arguments: list[float]) -> None:
        self.function, self.method, self.arguments = function, method, arguments

    @property
    def columns(self) -> list[str]:
        return []

    def __call__(self) -> Any:
        return self.function()

    @staticmethod
    def _rdf_draw_group(functions: list[Any], count: int) -> list[Any] | None:
        if not _uniform_calls(functions):
            return None
        if not count:
            return [np.empty(0, dtype=np.float64) for _ in functions]
        generator = _generator(functions)
        bounds = _group_bounds(functions)
        if generator is None or bounds is None:
            return None
        draws = generator._xrd.rndm(n=count * len(functions)).reshape(count, len(functions))
        return [low + (high - low) * draws[:, at] for at, (low, high) in enumerate(bounds)]


def _uniform_calls(functions: list[Any]) -> bool:
    return bool(functions) and all(
        type(f) is _RandomDraw and f.method == "Uniform" for f in functions
    )


def _generator(functions: list[Any]) -> Any:
    from ..core.randoms import TRandom, TRandom1, TRandom2, TRandom3

    namespaces = [f.function.__globals__["ROOT"] for f in functions]
    if any(root is not namespaces[0] for root in namespaces):
        return None
    generator = namespaces[0].gRandom
    return generator if type(generator) in (TRandom, TRandom1, TRandom2, TRandom3) else None


def _group_bounds(functions: list[Any]) -> list[tuple[float, float]] | None:
    pairs = [_bounds(f.arguments) for f in functions]
    found = [pair for pair in pairs if pair is not None]
    return found if len(found) == len(pairs) else None


def _bounds(arguments: list[float]) -> tuple[float, float] | None:
    if len(arguments) > 2 or not all(math.isfinite(arg) for arg in arguments):
        return None
    if not arguments:
        return 0., 1.
    return (0., arguments[0]) if len(arguments) == 1 else (arguments[0], arguments[1])


def _constant(node: Any) -> float | None:
    if isinstance(node, cxx.Literal) and node.kind in ("int", "float"):
        return float(node.value)
    if isinstance(node, cxx.Unary) and node.op in ("+", "-") and not node.postfix:
        value = _constant(node.operand)
        if value is not None:
            return value if node.op == "+" else -value
    return None


def _random_call(body: list[Any]) -> cxx.Call | None:
    if len(body) != 1 or not isinstance(body[0], cxx.Return):
        return None
    call = body[0].value
    if not isinstance(call, cxx.Call) or not isinstance(call.func, cxx.Member):
        return None
    member = call.func
    if not isinstance(member.obj, cxx.Name) or member.obj.text != "gRandom":
        return None
    return call if member.name in _RANDOM_METHODS else None


def random_draw(function: Any, body: list[Any]) -> _RandomDraw | None:
    call = _random_call(body)
    if call is None:
        return None
    arguments = [_constant(arg) for arg in call.args]
    found = [arg for arg in arguments if arg is not None]
    if len(found) != len(arguments):
        return None
    assert isinstance(call.func, cxx.Member)
    return _RandomDraw(function, call.func.name, found)
