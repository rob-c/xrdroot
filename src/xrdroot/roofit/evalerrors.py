"""RooFit's evaluation-error log: what went wrong, where, while Minuit was minimising.

When a likelihood cannot be computed at the parameters Minuit tried, RooFit
prints why - for each object that failed, its description and one line per
failure with the values of what it is made of - before handing Minuit a
worse value (``RooAbsReal::printEvalErrors``). The failures are collected
here while a likelihood is evaluated, keyed by the object that failed, in
the order they first failed, and cleared after each evaluation.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

__all__ = ["collecting", "clear", "count", "record", "text"]

#: The failures of the current evaluation: each object's description and its messages.
_LOG: dict[Any, tuple[str, list[tuple[str, str]]]] = {}
#: Whether failures are being collected: only while a likelihood is evaluated for Minuit.
_ACTIVE = [False]
#: Public logging policy; a minimizer temporarily collects errors independently of it.
_MODE = [0]
#: CountErrors keeps a count without retaining object descriptions or messages.
_COUNT = [0]
#: The ROOT enum names accepted by PyROOT as strings.
MODES = {"PrintErrors": 0, "CollectErrors": 1, "CountErrors": 2, "Ignore": 3}


#: Whether a failure is being described: what that evaluates is not to report errors at all.
_DESCRIBING = [False]


def collecting(flag: bool) -> None:
    _ACTIVE[0] = bool(flag)


def active() -> bool:
    return not _DESCRIBING[0] and (_ACTIVE[0] or _MODE[0] in (1, 2))


def quiet() -> bool:
    """Whether errors are neither collected nor printed: while a failure is being described."""
    return _DESCRIBING[0] or (_MODE[0] == 3 and not _ACTIVE[0])


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
    if not active() or times <= 0:
        return
    if _MODE[0] == 2 and not _ACTIVE[0]:
        _COUNT[0] += times
        return
    was_active = _ACTIVE[0]
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
        _ACTIVE[0], _DESCRIBING[0] = was_active, False
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


def mode() -> int:
    return _MODE[0]


def set_mode(value: Any) -> None:
    if isinstance(value, str):
        value = MODES[value] if value in MODES else -1
    value = operator.index(value)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("The evaluation-error logging mode must be a ROOT enum value.")
    _MODE[0] = value


def number() -> int:
    return _COUNT[0] if _MODE[0] == 2 else count()


def items() -> int:
    return len(_LOG)


def clear_mode() -> None:
    """ROOT clears only the storage belonging to the current collecting mode."""
    if _MODE[0] == 1:
        _LOG.clear()
    elif _MODE[0] == 2:
        _COUNT[0] = 0


def formatted(most: int) -> str:
    return f"{_COUNT[0]} errors counted\n" if _MODE[0] == 2 else text(most)


def report(node: Any, message: str, server_values: str | None = None) -> None:
    from .messages import ERROR, log
    from .pdf import _proxy_values
    from .printing import kArgs, kClassName, kInline, kName

    if quiet():
        return

    def origin() -> str:
        return str(node.printStream(kClassName | kName | kArgs, kInline))

    def servers() -> str:
        return (
            ", ".join(_proxy_values(p) for p in node._proxies)
            if server_values is None
            else str(server_values)
        )

    if active():
        record(id(node), origin, str(message), servers)
    elif mode() == 0:
        found = f"RooAbsReal::logEvalError({node.GetName()}) evaluation error, \n origin       : "
        found += f"{origin()}\n message      : {message}\n server values: {servers()}"
        log(node, ERROR, "Eval", found)
