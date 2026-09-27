"""Which components of a sum are drawn: ``RooAbsReal::selectComp`` while a curve is sampled.

``plotOn(frame, Components("bkg"))`` evaluates the whole model with only
some terms of each sum switched on. The switch is this module's one piece
of state - the names selected, or ``None`` for all - held for as long as
the curve is being sampled, as RooFit holds its ``_selectComp`` flags.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["active", "selecting"]

_SELECTED: list[set[str] | None] = [None]


def active(obj: Any) -> bool:
    """Whether ``obj`` contributes: everything does unless a selection leaves it out."""
    chosen = _SELECTED[0]
    return chosen is None or obj.GetName() in chosen


@contextmanager
def selecting(names: set[str] | None) -> Iterator[None]:
    before = _SELECTED[0]
    _SELECTED[0] = names
    try:
        yield
    finally:
        _SELECTED[0] = before
