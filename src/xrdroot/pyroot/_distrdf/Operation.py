"""The operations distributed ``RDataFrame`` can record, and what kind each is.

A transformation (``Define``, ``Filter``...) makes a node the graph goes on
from; an action (``Count``, ``Histo1D``...) books a result, computed when
first asked for; an instant action (``Snapshot``, ``AsNumpy``) runs the graph
at once, as it does in ROOT. :func:`create_op` refuses any other name, as
ROOT's distributed mode supports only these.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Operation",
    "Action",
    "Histo",
    "VariationsFor",
    "InstantAction",
    "AsNumpy",
    "Snapshot",
    "Transformation",
    "SUPPORTED_OPERATIONS",
    "create_op",
]


class Operation:
    """An operation recorded on a node of the graph: its name and its arguments."""

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.name = name
        self.args = list(args)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


class Action(Operation):
    """An action: a result, booked."""


class Histo(Action):
    """A ``Histo*D``, which needs its model: histograms of tasks are added up bin by bin, so
    every task has to book the same bins, which only a model says beforehand."""

    def __init__(self, name: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(name, *args, **kwargs)
        if "model" not in self.kwargs and not (self.args and isinstance(self.args[0], tuple)):
            raise ValueError(
                "Creating a histogram without a model is not supported in distributed mode. "
                "Please make sure to specify the histogram model when rerunning the "
                "distributed RDataFrame application. For example:\n\n"
                "\tHisto1D('mycolumn') --> Histo1D(('myhist', 'myhist', 100, 0, 10), "
                "'mycolumn')\n\nSee the RDataFrame documentation for more details."
            )


class VariationsFor(Action):
    """``VariationsFor``: the varied results of an action."""


class InstantAction(Operation):
    """An action that runs the graph as soon as it is called."""


class AsNumpy(InstantAction):
    """``AsNumpy``: instant, unless called with ``lazy=True``."""


class Snapshot(InstantAction):
    """``Snapshot``: instant, unless its options say ``fLazy``."""


class Transformation(Operation):
    """A transformation: a node to go on from."""


#: Every operation distributed ``RDataFrame`` supports, and its kind.
SUPPORTED_OPERATIONS: dict[str, type[Operation]] = {
    "Alias": Transformation,
    "AsNumpy": AsNumpy,
    "Count": Action,
    "DefaultValueFor": Transformation,
    "Define": Transformation,
    "DefinePerSample": Transformation,
    "Filter": Transformation,
    "FilterAvailable": Transformation,
    "FilterMissing": Transformation,
    "Graph": Action,
    "GraphAsymmErrors": Action,
    "Histo1D": Histo,
    "Histo2D": Histo,
    "Histo3D": Histo,
    "HistoND": Histo,
    "HistoNSparseD": Histo,
    "Max": Action,
    "Mean": Action,
    "Min": Action,
    "Profile1D": Action,
    "Profile2D": Action,
    "Profile3D": Action,
    "Redefine": Transformation,
    "Snapshot": Snapshot,
    "Stats": Action,
    "StdDev": Action,
    "Sum": Action,
    "Report": Action,
    "VariationsFor": VariationsFor,
    "Vary": Transformation,
}


def create_op(name: str, *args: Any, **kwargs: Any) -> Operation:
    """The operation of that name, of its kind; a ``ValueError`` for one not supported."""
    kind = SUPPORTED_OPERATIONS.get(name)
    if kind is None:
        raise ValueError(
            f"Operation '{name}' is either invalid or not supported in distributed mode. "
            "See the documentation for a list of supported operations."
        )
    return kind(name, *args, **kwargs)
