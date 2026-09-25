"""The two containers ROOT draws several things at once with: ``THStack`` and ``TMultiGraph``.

A stack is drawn as ``THStack::Paint`` draws it: each histogram sitting on
the sum of the ones added before it, the first at the bottom, and every one
drawn with the stack's option - ``HIST`` unless told otherwise. ``NOSTACK``
draws them over one another instead, and ``NOSTACKB`` as bars side by side
in each bin. A multigraph draws each of its graphs with its option, on the
one frame; ``PLC``, ``PMC`` and ``PFC`` colour them from the palette.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..hist import Histogram
from ..stacks import MultiGraph, Stack
from .binned import Binned, binned_layers
from .drawers import attached, default_graph, graph, graph_titles, titles
from .model import Bars, Frame
from .request import Request, styled

__all__ = ["histograms", "multigraph", "stack"]


def _flat_only(items: list[Any], what: str) -> list[Histogram]:
    for item in items:
        if not isinstance(item, Histogram) or len(item.axes) != 1:
            raise ValueError(
                f"{what} is drawn from histograms of one axis, and "
                f"{getattr(item, 'name', item)!r} is not one; ROOT piles 2-D histograms up "
                f"as LEGO plots, which is a sum of them drawn with option LEGO here"
            )
    return list(items)


def _each(request: Request, index: int, count: int) -> Request:
    chosen = request.chosen.without("NOSTACK", "NOSTACKB")
    if not chosen.drawing:
        chosen = chosen._replace(words=chosen.words | {"HIST"})
    return request.again(chosen, index, count)


def _stacked(items: list[Histogram], request: Request) -> list[Any]:
    below = np.zeros(len(items[0].values()))
    layers: list[Any] = []
    for index, item in enumerate(items):
        each = _each(request, index, len(items))
        top = below + item.values()
        data = Binned(item.edges(), top, item.errors(), styled(item.members, each), below)
        layers += binned_layers(data, each)
        below = top
    return layers


def _overlaid(items: list[Histogram], request: Request) -> list[Any]:
    layers: list[Any] = []
    for index, item in enumerate(items):
        each = _each(request, index, len(items))
        data = Binned(item.edges(), item.values(), item.errors(), styled(item.members, each))
        layers += binned_layers(data, each)
    return layers


def _side_by_side(items: list[Histogram], request: Request) -> list[Any]:
    """``NOSTACKB``: each bin split into as many bars as there are histograms."""
    layers: list[Any] = []
    share = 1.0 / len(items)
    for index, item in enumerate(items):
        each = _each(request, index, len(items))
        look = styled(item.members, each)
        edges = item.edges()
        widths = np.diff(edges)
        left = edges[:-1] + widths * share * index
        layers.append(Bars(left, left + widths * share, item.values(), look))
    return layers


def histograms(items: list[Any], request: Request, title: str) -> tuple[list[Any], Frame]:
    """Histograms drawn as a stack is: piled, overlaid with ``NOSTACK``, or ``NOSTACKB`` bars."""
    flat = _flat_only(items, "a stack")
    if not flat:
        return [], Frame(title=title)
    chosen = request.chosen
    if chosen.has("NOSTACKB"):
        layers = _side_by_side(flat, request)
    elif chosen.has("NOSTACK"):
        layers = _overlaid(flat, request)
    else:
        layers = _stacked(flat, request)
    return layers, titles(flat[0], title or flat[0].title)


def stack(obj: Stack, request: Request) -> tuple[list[Any], Frame]:
    """A ``THStack``, with its own title over the histograms'."""
    return histograms(list(obj), request, obj.title)


def multigraph(obj: MultiGraph, request: Request) -> tuple[list[Any], Frame]:
    """A ``TMultiGraph``: every graph with the option, on one frame titled by the multigraph.

    A fit made to the graphs together hangs on the multigraph rather than on
    any one of them, and ROOT draws it over them all, as it does a graph's own.
    """
    layers: list[Any] = []
    graphs = list(obj)
    for index, item in enumerate(graphs):
        chosen = default_graph(item, request.chosen)
        layers += graph(item, request.again(chosen, index, len(graphs)))[0]
    layers += attached(obj.functions, request)
    return layers, graph_titles(obj)
