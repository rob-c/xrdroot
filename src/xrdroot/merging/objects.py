"""The objects ``hadd`` adds up, added up as ROOT's own ``Merge`` methods do.

A histogram or profile is ``TH1::Merge``: the bins, the squares of the
weights, the running sums and the entries all add, which needs the same bins
in each - two histograms binned differently are refused by name, since
adding them bin by bin would put one histogram's counts in the other's bins.
An efficiency is ``TEfficiency::Merge``: what passed added to what passed,
and what was tried to what was tried. A graph is ``TGraph::Merge``: the
points of each after the points of the one before, with their error bars -
bars of the kind the first graph keeps, zero for a graph that kept none -
and a multigraph is ``TMultiGraph::Merge``, each one's graphs after the last's.

Anything else is not added up here, and :mod:`.files` says what becomes of
it: what ROOT does with an object it has no ``Merge`` for, which is to carry
each file's copy over as it is.
"""

from __future__ import annotations

import copy as _copy
from typing import Any

import numpy as np

from ..efficiency import Efficiency
from ..graph import Graph, _core
from ..hist import Histogram
from ..stacks import MultiGraph
from ..winfo import INFOS

__all__ = ["MERGERS", "mergeable"]


def merge_histograms(first: Histogram, other: Histogram) -> Histogram:
    """``TH1::Merge`` of two, which ``Profile`` does as ``TProfile::Merge``."""
    return type(first).merge([first, other])


def merge_efficiencies(first: Efficiency, other: Efficiency) -> Efficiency:
    """``TEfficiency::Merge``: the two histograms of each added to the other's."""
    members = dict(first.members)
    members["fPassedHistogram"] = Histogram.merge([first.passed, other.passed])
    members["fTotalHistogram"] = Histogram.merge([first.total, other.total])
    return Efficiency(first.classname, members)


def _bars(pair: Any, count: int) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """A graph's bars along one axis, low and high, or zero where it kept none."""
    if pair is None:
        return np.zeros(count), np.zeros(count)
    return pair[0], pair[1]


def _even(low: np.ndarray[Any, Any], high: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """One bar for both sides, as ``TGraphErrors::GetErrorX`` makes of two."""
    return np.asarray(np.sqrt((low * low + high * high) / 2))


def _joined_bars(members: dict[str, Any], classname: str, graphs: tuple[Graph, ...]) -> None:
    """The error bars of every graph, one after another, kept as ``classname`` keeps them."""
    across = [_bars(graph.xerr, len(graph)) for graph in graphs]
    upward = [_bars(graph.yerr, len(graph)) for graph in graphs]
    if classname == "TGraphErrors":
        members["fEX"] = np.concatenate([_even(*pair) for pair in across])
        members["fEY"] = np.concatenate([_even(*pair) for pair in upward])
    elif classname == "TGraphAsymmErrors":
        _uneven_bars(members, "fEXlow", "fEXhigh", across)
        _uneven_bars(members, "fEYlow", "fEYhigh", upward)


def _uneven_bars(members: dict[str, Any], low: str, high: str, pairs: list[Any]) -> None:
    """The bars below and above, each side of every graph one after another."""
    members[low] = np.concatenate([pair[0] for pair in pairs])
    members[high] = np.concatenate([pair[1] for pair in pairs])


def merge_graphs(first: Graph, other: Graph) -> Graph:
    """``TGraph::Merge``: the other's points after this one's, bars and all.

    The class is the first graph's, and so is everything else about it -
    its title, its drawing attributes, the functions fitted to it - except
    the histogram ROOT keeps to draw its axes, which the new points would
    not fit and ROOT makes again.
    """
    members = _copy.deepcopy(first.members)
    core = _core(members)
    assert core is not None  # a Graph cannot be made without one
    graphs = (first, other)
    core["fNpoints"] = len(first) + len(other)
    core["fX"] = np.concatenate([graph.x for graph in graphs])
    core["fY"] = np.concatenate([graph.y for graph in graphs])
    core["fHistogram"] = None
    _joined_bars(members, first.classname, graphs)
    return Graph(first.classname, members)


def merge_multigraphs(first: MultiGraph, other: MultiGraph) -> MultiGraph:
    """``TMultiGraph::Merge``: the other's graphs after this one's, each as it was."""
    members = dict(first.members)
    members["fGraphs"] = [*first, *other]
    return MultiGraph(first.classname, members)


#: How each kind of object is merged with the next, by the class it comes back as.
MERGERS: dict[type, Any] = {
    Histogram: merge_histograms,
    Efficiency: merge_efficiencies,
    Graph: merge_graphs,
    MultiGraph: merge_multigraphs,
}


def mergeable(value: Any) -> Any:
    """How ``value`` is merged with another like it, or ``None`` if it is not merged here.

    It has to be one of the kinds above, and of a class this library can
    write again: a ``TGraphMultiErrors`` is a graph, but one the writer has
    no layout for, and so is carried over rather than merged.
    """
    if getattr(value, "classname", None) not in INFOS:
        return None
    for kind, merger in MERGERS.items():
        if isinstance(value, kind):
            return merger
    return None
