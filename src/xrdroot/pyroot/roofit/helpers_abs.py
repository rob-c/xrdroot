"""``RooDataSetHelper`` and ``RooDataHistHelper``: what a frame's ``Book`` fills.

ROOT declares these in ``RooAbsDataHelper.h`` as the sinks of an
``RDataFrame``'s ``Book``: each holds a RooFit data object and a set of
variables, and the frame's event loop fills it, the columns matched to the
variables by position rather than by name. The helper gives back the data
object once the loop is done, which is the booked result's value.

A ``Fill`` of one entry is here too, since a macro may call one itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...roofit.data.datahist import RooDataHist
from ...roofit.data.dataset import RooDataSet, as_set

__all__ = ["RooDataSetHelper", "RooDataHistHelper"]


class _Helper:
    """What both helpers do the same way: an empty data object, and the columns filling it."""

    #: The class of the data object the result is.
    _data_class: Any = RooDataSet

    def __init__(self, name: Any, title: Any, variables: Any) -> None:
        chosen = as_set(variables)
        #: The variables, in the order the frame's columns are matched to them.
        self.variables = chosen
        #: The names the variables go by, in that order.
        self.names = [one.GetName() for one in chosen]
        #: The data object, empty until the loop fills it.
        self.data = self._data_class(name, title, list(chosen))

    def _values(self, columns: Iterable[Any]) -> dict[str, Any]:
        """A batch's columns, by the names of the variables they are matched to."""
        return dict(zip(self.names, columns, strict=True))

    def fill_columns(self, columns: dict[str, Any]) -> None:
        """A batch of entries, one array per column, matched to the variables by position."""
        self._fill(self._values(columns.values()))

    def _fill(self, columns: dict[str, Any]) -> None:
        self.data.add_columns(columns)

    def Fill(self, *values: Any) -> None:
        """One more event, the values in the order of the variables."""
        self._fill({name: [value] for name, value in self._values(values).items()})

    def result(self) -> Any:
        """What the loop filled."""
        return self.data


class RooDataSetHelper(_Helper):
    """``RooDataSetHelper``: ``Book``'s sink, whose result is the unbinned dataset it filled."""


class RooDataHistHelper(_Helper):
    """``RooDataHistHelper``: ``Book``'s sink, whose result is the histogram it filled."""

    _data_class = RooDataHist

    def _fill(self, columns: dict[str, Any]) -> None:
        self.data.add_arrays(columns)
