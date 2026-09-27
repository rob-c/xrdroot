"""``RooAbsData::reduce``: a smaller dataset - fewer variables, fewer events.

``reduce(Cut="y>5")``, ``reduce(SelectVars={x})``, ``reduce(CutRange="sb")``,
``reduce(EventRange=(0, 100))`` - or a cut string alone, or a set of
variables alone - each keeps what it names, the dataset's name and title
kept too, as ``RooDataSet::reduceEng`` keeps them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cmdargs import RooCmdArg, commands
from ..collections import as_list

__all__ = ["reduced"]


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The options, a bare cut string or set of variables read as ``Cut`` or ``SelectVars``."""
    given: list[Any] = []
    for arg in args:
        if isinstance(arg, RooCmdArg):
            given.append(arg)
        elif isinstance(arg, str):
            given.append(RooCmdArg("Cut", arg))
        else:
            given.append(RooCmdArg("SelectVars", arg))
    return commands(given, kwargs)


def _kept(data: Any, options: Any) -> Any:
    """The events the ``Cut``, ``CutRange`` and ``EventRange`` options keep."""
    keep = data.mask(options.get("Cut"), options.get("CutRange"))
    if "EventRange" in options:
        first, last = (
            options.get("EventRange", 0, 0),
            options.get("EventRange", 1, data.numEntries()),
        )
        window = np.zeros_like(keep)
        window[int(first) : int(last)] = True
        keep &= window
    return keep


def _chosen(data: Any, options: Any) -> list[Any]:
    """The variables ``SelectVars`` keeps - all of them, without it."""
    chosen = options.get("SelectVars")
    if chosen is None:
        return list(data.get())
    names = {one.GetName() for one in as_list(chosen)}
    return [one for one in data.get() if one.GetName() in names]


def _part(values: Any, keep: Any) -> Any:
    return None if values is None else values[keep]


def reduced(data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    from .store import copies_of

    options = _options(args, kwargs)
    keep = _kept(data, options)
    variables = _chosen(data, options)
    kind: Any = type(data)
    made = kind.__new__(kind)
    made.__dict__.update(data.__dict__)
    made._vars = copies_of(variables)
    made._columns = {one.GetName(): data.column(one.GetName())[keep] for one in variables}
    made._weights = _part(data._weights, keep)
    made._sumw2 = _part(data._sumw2, keep)
    if "Name" in options:
        made.SetName(str(options.get("Name")))
    return made
