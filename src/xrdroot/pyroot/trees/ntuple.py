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

    def Fill(self, *values: Any) -> int:
        """One entry: a value for every variable, in order, or one sequence of them."""
        self._writable("Fill")
        if not values:  # TTree::Fill: the values already at the variables' addresses
            return super().Fill()
        if len(values) == len(self._values) and all(isinstance(v, np.ndarray) for v in values):
            return self._fill_columns(values)
        given = values[0] if len(values) == 1 and np.ndim(values[0]) == 1 else values
        if len(given) != len(self._values):
            raise ValueError(
                f"{self._name!r} has {len(self._values)} variables, and Fill was given "
                f"{len(given)} values"
            )
        self._values[:] = given
        return super().Fill()

    def _fill_columns(self, columns: tuple[Any, ...]) -> int:
        """Many entries at once, an array for each variable - what a macro's fill loop of
        every turn at once hands over: the entries ``Fill`` of each in turn would make."""
        assert self._store is not None
        arrays = [np.asarray(column, dtype=self._code).reshape(-1) for column in columns]
        count = len(arrays[0])
        if any(len(array) != count for array in arrays):
            raise ValueError(f"{self._name!r} was given columns of different lengths to fill")
        names = list(self._store.slots)
        self._store.extend(dict(zip(names, arrays, strict=True)), count)
        self._changed()
        self._values[:] = [array[-1] for array in arrays] if count else self._values
        return int(count * len(arrays) * self._values.itemsize)

    def GetArgs(self) -> np.ndarray[Any, Any]:
        """The values of the entry last read, one per variable, in order."""
        return np.array(
            [
                0 if self._current(column) is None else self._current(column)
                for column in self._columns()
            ],
            dtype=self._code,
        )


class TNtupleD(TNtuple):
    """``TNtupleD``: a ``TNtuple`` of ``Double_t``."""

    _classname = "TNtupleD"
    _code = "d"
