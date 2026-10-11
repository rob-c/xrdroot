"""``ROOT::RDF::Not`` and ``PassAsVec``: callables made from callables, for a frame's methods.

``Not(f)`` takes what ``f`` takes and is true where ``f`` is false, for a
``Filter`` the other way round; ``PassAsVec<N, T>(f)`` takes ``N`` values
and hands ``f`` one ``std::vector<T>`` of them. Each made from a macro's
function stays a macro's function, so the frame still calls it once per
entry (:mod:`.entrywise`); made from one of whole batches, it is given the
batches.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["Not", "PassAsVec"]


def _like(made: Callable[..., Any], function: Any, parameters: list[str]) -> Callable[..., Any]:
    """``made`` dressed as ``function``'s kind - a macro's or not - taking ``parameters``."""
    made.__module__ = getattr(function, "__module__", __name__)
    plain = inspect.Parameter.POSITIONAL_OR_KEYWORD
    made.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(name, plain) for name in parameters]
    )
    return made


def Not(function: Callable[..., Any]) -> Callable[..., Any]:
    """``RDF::Not(f)``: true where ``f`` is false - entry by entry, or element by element of
    a batch's array."""

    def negated(*args: Any) -> Any:
        found = function(*args)
        return np.logical_not(found) if np.ndim(found) else not found

    names = list(inspect.signature(function).parameters)
    return _like(negated, function, names)


class _PassAsVec:
    """``RDF::PassAsVec<N, T>``: ``PassAsVec[N, "int"](f)``, the template's arguments first."""

    def __getitem__(self, arguments: tuple[int, Any]) -> Callable[[Callable[..., Any]], Any]:
        count, kind = arguments

        def wrapping(function: Callable[..., Any]) -> Callable[..., Any]:
            from ..stl import std

            def passed(*args: Any) -> Any:
                return function(std.vector[kind](list(args)))

            return _like(passed, function, [f"arg{at}" for at in range(int(count))])

        return wrapping


#: ``ROOT.RDF.PassAsVec``.
PassAsVec = _PassAsVec()
