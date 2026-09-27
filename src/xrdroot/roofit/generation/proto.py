"""``generate(vars, ProtoData(data))``: events drawn given the values a prototype dataset has.

A conditional density - a decay whose resolution is a per-event error -
is generated event by event at the prototype's values: the prototype's
``i``-th event is loaded, then the observables are drawn given it, as
``RooAbsGenContext::generate`` does, and the dataset made has the
prototype's variables as well as the generated ones. With no number of
events asked for, there are as many as the prototype has.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ..collections import as_list
from .contexts import Context, context_for

__all__ = ["ProtoFeed"]


class ProtoFeed:
    """A prototype dataset's events, handed to the density one at a time."""

    def __init__(self, pdf: Any, data: Any) -> None:
        """The feed of ``data``'s events to ``pdf`` - one that feeds nothing if ``data`` is ``None``."""
        self.variables = list(as_list(data.get())) if data is not None else []
        self.names = frozenset(one.GetName() for one in self.variables)
        self.columns = {name: np.asarray(data.column(name)) for name in self.names}
        self.size = int(data.numEntries()) if data is not None else 0
        self.targets = [pdf.variable(name) for name in sorted(self.names & pdf.dependents())]

    def context(self, pdf: Any, names: frozenset[str]) -> Context:
        """The density's context, told which variables come from the prototype if it wants to know."""
        make = getattr(pdf, "gen_context", None)
        if self.names and make is not None and "proto" in inspect.signature(make).parameters:
            return make(names, proto=self.names)  # type: ignore[no-any-return]
        return context_for(pdf, names)

    def count(self, asked: Any) -> Any:
        """The events asked for - or, with a prototype and no number asked, as many as it has."""
        return asked if asked is not None or not self.names else self.size

    def load(self, index: int) -> dict[str, float]:
        """The prototype's event ``index`` - from the start again past its end - set on the density."""
        row = {name: float(column[index % self.size]) for name, column in self.columns.items()}
        for one in self.targets:
            one.load_value(row[one.GetName()])
        return row

    def extra(self, variables: list[Any]) -> list[Any]:
        """The prototype's variables the dataset needs besides those generated."""
        have = {one.GetName() for one in variables}
        return [one for one in self.variables if one.GetName() not in have]
