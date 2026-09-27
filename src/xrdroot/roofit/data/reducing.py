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


def reduced(data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    options = _options(args, kwargs)
    keep = data.mask(options.get("Cut"), options.get("CutRange"))
    first, last = options.get("EventRange", 0, 0), options.get("EventRange", 1, data.numEntries())
    if "EventRange" in options:
        window = np.zeros_like(keep)
        window[int(first):int(last)] = True
        keep &= window
    chosen = options.get("SelectVars")
    names = [one.GetName() for one in as_list(chosen)] if chosen is not None else None
    variables = [one for one in data.get() if names is None or one.GetName() in names]
    made = type(data).__new__(type(data))
    made.__dict__.update(data.__dict__)
    from .store import copies_of

    made._vars = copies_of(variables)
    made._columns = {one.GetName(): data.column(one.GetName())[keep] for one in variables}
    made._weights = None if data._weights is None else data._weights[keep]
    made._sumw2 = None if data._sumw2 is None else data._sumw2[keep]
    if "Name" in options:
        made.SetName(str(options.get("Name")))
    return made
