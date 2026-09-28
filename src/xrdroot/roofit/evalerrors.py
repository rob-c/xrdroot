"""RooFit's evaluation-error log: what went wrong, where, while Minuit was minimising.

When a likelihood cannot be computed at the parameters Minuit tried, RooFit
prints why - for each object that failed, its description and one line per
failure with the values of what it is made of - before handing Minuit a
worse value (``RooAbsReal::printEvalErrors``). The failures are collected
here while a likelihood is evaluated, keyed by the object that failed, in
the order they first failed, and cleared after each evaluation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["collecting", "clear", "count", "record", "text"]

#: The failures of the current evaluation: each object's description and its messages.
_LOG: dict[Any, tuple[str, list[tuple[str, str]]]] = {}
#: Whether failures are being collected: only while a likelihood is evaluated for Minuit.
_ACTIVE = [False]


#: Whether a failure is being described: what that evaluates is not to report errors at all.
_DESCRIBING = [False]


def collecting(flag: bool) -> None:
    _ACTIVE[0] = bool(flag)


def active() -> bool:
    return _ACTIVE[0]


def quiet() -> bool:
    """Whether errors are neither collected nor printed: while a failure is being described."""
    return _DESCRIBING[0]


def record(
    key: Any,
    origin: Callable[[], str],
    message: str,
    servers: Callable[[], str],
    times: int = 1,
    top: bool = False,
) -> None:
    """Note ``times`` failures of ``key`` with ``message``; the descriptions are made once.

    The descriptions evaluate what they describe, which is not to fail twice,
    so nothing is collected while they are made. The likelihood's own
    density - ``top`` - is listed first, as RooFit's log happens to list it.
    """
    if not _ACTIVE[0] or times <= 0:
        return
    _ACTIVE[0], _DESCRIBING[0] = False, True
    try:
        if key not in _LOG:
            entry: tuple[str, list[tuple[str, str]]] = (origin(), [])
            rest = dict(_LOG)
            _LOG.clear()
            if top:
                _LOG[key] = entry
            _LOG.update(rest)
            _LOG.setdefault(key, entry)
        line = (message, servers())
    finally:
        _ACTIVE[0], _DESCRIBING[0] = True, False
    _LOG[key][1].extend([line] * times)


def count() -> int:
    return sum(len(entries) for _, entries in _LOG.values())


def clear() -> None:
    _LOG.clear()


def text(most: int) -> str:
    """``RooAbsReal::printEvalErrors(os, most)``: every object's failures, ``most`` lines each."""
    found = ""
    for origin, entries in _LOG.values():
        if most == 0:
            found += f"{origin} has {len(entries)} errors\n"
            continue
        found += origin + "\n"
        for index, (message, servers) in enumerate(entries):
            found += f"     {message} @ {servers}\n"
            if index > most:
                found += f"    ... (remaining {len(entries) - most} messages suppressed)\n"
                break
    return found
