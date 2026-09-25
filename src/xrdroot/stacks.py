"""The two containers ROOT draws several things at once with.

A ``TMultiGraph`` is a list of graphs, and a ``THStack`` a list of
histograms, each with the frame they are drawn in beside it. Neither holds
anything the things in it do not, so each comes back as the sequence it is:
the graphs as :class:`~.graph.Graph` and the histograms as
:class:`~.hist.Histogram`, in the order they were added.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .display import Displayed
from .errors import FormatError
from .function.attached import listed
from .graph import Graph
from .hist import Histogram

__all__ = ["COLLECTIONS", "MultiGraph", "Stack"]


class _Held(Displayed, Sequence[Any]):
    """What the two have in common: a name, a title, and a list of things."""

    __slots__ = ("classname", "members", "_items")

    #: The member the things are kept in, and the class each of them has to be.
    held = ""
    kind: type = object

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        items = members.get(self.held)
        items = [] if items is None else list(items)
        strangers = [item for item in items if not isinstance(item, self.kind)]
        if strangers:
            raise FormatError(
                f"a {classname} holds {len(strangers)} things that are not a "
                f"{self.kind.__name__}, the first of them {strangers[0]!r}"
            )
        #: The class the file says this is.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        self._items = items

    @property
    def name(self) -> str:
        return str(self.members["TNamed"]["fName"])

    @property
    def title(self) -> str:
        return str(self.members["TNamed"]["fTitle"])

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: Any) -> Any:
        return self._items[index]

    def __repr__(self) -> str:
        return f"<{self.classname} {self.name!r} of {len(self)}>"


class MultiGraph(_Held):
    """A ``TMultiGraph``: graphs drawn on one frame, as a sequence of :class:`~.graph.Graph`.

        >>> [graph.classname for graph in f["mg"]]      # doctest: +SKIP
        ['TGraph', 'TGraphErrors']
    """

    __slots__ = ()
    held = "fGraphs"
    kind = Graph

    @property
    def functions(self) -> list[Any]:
        """``GetListOfFunctions``: the fits made to all the graphs together."""
        return listed(self.members)

    def fit(
        self,
        model: Any,
        option: str = "",
        range: Any = None,
        *,
        parameters: Any = None,
        limits: Any = None,
        fixed: Any = None,
        npar: int | None = None,
    ) -> Any:
        """``TMultiGraph::Fit``: one fit to the points of every graph, as :meth:`Graph.fit`.

        The data are of the most elaborate kind any graph's bars are - so in
        a multigraph with errors, a graph without them has no points to give.
        """
        from .fit import fit_object

        return fit_object(
            self, model, option, range, parameters=parameters, limits=limits, fixed=fixed, npar=npar
        )


class Stack(_Held):
    """A ``THStack``: histograms drawn one on top of another, as a sequence of them.

        >>> sum(h.values() for h in f["stack"])        # doctest: +SKIP
    """

    __slots__ = ()
    held = "fHists"
    kind = Histogram


#: The containers this reads, against the class each comes back as.
COLLECTIONS: dict[str, type[_Held]] = {"TMultiGraph": MultiGraph, "THStack": Stack}
