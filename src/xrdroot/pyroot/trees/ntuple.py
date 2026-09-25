"""``TNtuple`` and ``TNtupleD``: a tree of named numbers, filled with them directly.

    >>> ntuple = TNtuple("ntuple", "data", "px:py:pz")
    >>> ntuple.Fill(1.0, 2.0, 3.0)

Each variable is a branch of one ``Float_t`` (a ``Double_t`` for
``TNtupleD``), whose title is its name, as ROOT makes them; ``Fill`` takes
the values in the order the variables were named, or one sequence of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .addresses import View
from .store import Slot
from .tree import TTree

__all__ = ["TNtuple", "TNtupleD"]


class TNtuple(TTree):
    """``TNtuple``: see the module's docstring."""

    _classname = "TNtuple"
    #: The type every variable is.
    _code = "f"

    def __init__(self, name: str = "", title: str = "", varlist: str = "", bufsize: int = 32000):
        super().__init__(name, title)
        names = [part.strip() for part in str(varlist).split(":") if part.strip()]
        self._values = np.zeros(len(names), dtype=self._code)
        assert self._store is not None
        for at, variable in enumerate(names):
            self._store.add(Slot(variable, variable, self._code, View(self._values, at)))

    def GetNvar(self) -> int:
        return len(self._layout())

    def Fill(self, *values: Any) -> int:  # type: ignore[override]
        """One entry: a value for every variable, in order, or one sequence of them."""
        self._writable("Fill")
        given = values[0] if len(values) == 1 and np.ndim(values[0]) == 1 else values
        if len(given) != len(self._values):
            raise ValueError(
                f"{self._name!r} has {len(self._values)} variables, and Fill was given "
                f"{len(given)} values"
            )
        self._values[:] = given
        return super().Fill()

    def GetArgs(self) -> np.ndarray[Any, Any]:
        """The values of the entry last read, one per variable, in order."""
        return np.array(
            [0 if self._current(column) is None else self._current(column) for column in
             self._columns()],
            dtype=self._code,
        )


class TNtupleD(TNtuple):
    """``TNtupleD``: a ``TNtuple`` of ``Double_t``."""

    _classname = "TNtupleD"
    _code = "d"
