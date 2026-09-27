"""The copies of a model RooFit fits, plots and generates with - for nodes whose state shows it.

RooFit does not fit, plot or sample the model it is handed but a clone of
it, and a clone starts with empty caches. For most nodes that is invisible;
for one that announces each cache it fills - ``RooFFTConvPdf``'s
"creating new cache" - it decides what is printed, and when. xrdroot does
not clone; instead a node that cares defines ``copy_for(purpose, nset)``,
returning the state its clone would have - announcing what the clone
announces on being made - and :func:`within` makes those states the ones
:func:`state_of` finds while the fit, plot or generation runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["copies_of", "state_of", "within"]

#: The copies in use, innermost last: each maps a node's ``id`` to its copy's state.
_ACTIVE: list[dict[int, Any]] = []


def copies_of(top: Any, purpose: str, nset: frozenset[str]) -> dict[int, Any]:
    """The state of each node under ``top`` that keeps one, as a clone made to ``purpose`` has."""
    found: dict[int, Any] = {}
    for node in top._walk():
        make = getattr(node, "copy_for", None)
        if make is not None:
            found[id(node)] = make(purpose, nset)
    return found


@contextmanager
def within(states: dict[int, Any]) -> Iterator[None]:
    """Use the copies' states in ``states`` until the block ends."""
    _ACTIVE.append(states)
    try:
        yield
    finally:
        _ACTIVE.pop()


def state_of(node: Any) -> Any:
    """The state of the copy of ``node`` in use now, or ``None`` for the node itself."""
    for states in reversed(_ACTIVE):
        if id(node) in states:
            return states[id(node)]
    return None
