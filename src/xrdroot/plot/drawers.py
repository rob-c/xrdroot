"""Each kind of object this library reads, turned into the layers and frame of its picture.

This is where ROOT's defaults live: a histogram with no option is drawn as
``HIST``, or ``E`` once it keeps the squares of its weights, with its fits
drawn over it; a profile as ``E``; a 2-D histogram as ``COLZ``; a graph as
``ALP``, or what its ``fOption`` was set to; an efficiency as ``AP`` or
``COLZ``; a function as a line through ``fNpx`` points over its range, or
a ``TF2`` as its contour lines. The numbers come from the object; how they
are drawn, from :mod:`~.binned`, :mod:`~.grid` and :mod:`~.scatter`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..efficiency import Efficiency
from ..errors import FormatError, UnsupportedFeatureError
from ..formula import FormulaError
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from .attributes import found
from .binned import Binned, binned_layers
from .grid import Grid, grid_layers
from .model import Cloud, Curve, Frame, Points
from .options import Chosen, choose
from .request import Request, styled
from .scatter import Scatter, scatter_layers

__all__ = [
    "NOT_DRAW",
    "attached",
    "default_graph",
    "graph_titles",
    "titles",
    "efficiency",
    "function",
    "graph",
    "held",
    "histogram",
    "kind_of",
]

#: ``TF1::kNotDraw``: a fit made with option ``0``, kept but not drawn.
NOT_DRAW = 1 << 9

#: The dimensions of a histogram, against the kind of thing its option is read for.
HISTOGRAM_KINDS = {
    1: "histogram",
    2: "two-dimensional histogram",
    3: "three-dimensional histogram",
}


def held(members: Any, key: str) -> Any:
    """Whatever is under ``key`` nearest the top of ``members``: a number, an object, a dict."""
    queue: deque[Any] = deque([members])
    while queue:
        here = queue.popleft()
        if key in here and here[key] is not None:
            return here[key]
        queue.extend(value for value in here.values() if isinstance(value, Mapping))
    return None


#: The members holding each axis of a ``TH1``.
_AXES = ("fXaxis", "fYaxis", "fZaxis")


def _axis_title(members: Any, axis: str) -> str:
    """The title of ``fXaxis``, ``fYaxis`` or ``fZaxis``, which ROOT keeps on every ``TH1``."""
    named = (found(members, axis) or {}).get("TNamed", {})
    return str(named.get("fTitle", ""))


def titles(histogram: Histogram, title: str) -> Frame:
    """The frame of a histogram's picture: its title, and the title of each axis.

    The axes it is binned along say theirs; the one its values go up -
    ``fYaxis`` of a 1-D histogram, ``fZaxis`` of a 2-D one - is still kept.
    """
    labels = [axis.title for axis in histogram.axes]
    labels += [_axis_title(histogram.members, axis) for axis in _AXES[len(labels) :]]
    return Frame(title=title, xlabel=labels[0], ylabel=labels[1], zlabel=labels[2])


# -- the functions a histogram or graph carries ------------------------------------------


def _drawable(candidate: Any) -> bool:
    if not isinstance(candidate, Function) or candidate.dimensions != 1:
        return False
    bits = int((found(candidate.members, "TNamed") or {}).get("fBits", 0))
    return not bits & NOT_DRAW


def _sampled(function: Function, request: Request) -> list[Any]:
    """The function as a line through ``fNpx + 1`` points spanning its range."""
    low, high = function.range
    xs = np.linspace(low, high, int(held(function.members, "fNpx") or 100) + 1)
    ys = np.asarray(function(xs), dtype=np.float64)
    look = styled(function.members, request, markers=request.chosen.has("P", "*"))
    if request.chosen.has("P", "*"):
        zeros = np.zeros_like(xs)
        return [Points(xs, ys, zeros, zeros, zeros, zeros, look)]
    return [Curve(xs, ys, look, smooth=request.chosen.has("C"))]


def attached(functions: list[Any], request: Request) -> list[Any]:
    """ROOT's ``FUNC`` layer: every fit hung on the object, unless option ``0`` hid it.

    A function that cannot be worked out here - its formula calls C++ this
    library does not have - is left out, as the thing it was fitted to is
    still worth drawing.
    """
    lines: list[Any] = []
    plain = request.again(Chosen(frozenset()))._replace(marks={}, each=())
    for candidate in functions:
        if not _drawable(candidate):
            continue
        try:
            lines.extend(_sampled(candidate, plain))
        except (FormulaError, UnsupportedFeatureError, FormatError):
            continue
    return lines


# -- histograms and profiles ---------------------------------------------------------------


def _default_flat(histogram: Histogram, chosen: Chosen) -> Chosen:
    """``HIST``, or ``E`` for a profile or a histogram keeping the squares of its weights."""
    if chosen.drawing or chosen.has("FUNC"):
        return chosen
    errors = histogram.kind == "MEAN" or histogram.weighted
    return chosen._replace(words=chosen.words | {"E" if errors else "HIST"})


def _normalised(histogram: Histogram, chosen: Chosen) -> float:
    """``NORM``: what to scale by so the bins sum to one - refused for a profile's means."""
    if not chosen.has("NORM"):
        return 1.0
    if histogram.kind == "MEAN":
        raise ValueError(
            f"{histogram.name!r} is a profile, whose bins are means; they do not add up to "
            f"anything that NORM could scale to one"
        )
    total = histogram.sum()
    return 1.0 / total if total else 1.0


def _flat(histogram: Histogram, request: Request) -> tuple[list[Any], Frame]:
    """The histogram as its option says, then its fits - which ``HIST`` itself, and only
    that, leaves out: ROOT's default outline is drawn with them."""
    hidden = request.chosen.has("HIST")
    chosen = _default_flat(histogram, request.chosen)
    request = request._replace(chosen=chosen)
    scale = _normalised(histogram, chosen)
    data = Binned(
        histogram.edges(),
        histogram.values() * scale,
        histogram.errors() * scale,
        styled(histogram.members, request),
    )
    layers = binned_layers(data, request)
    if not hidden:
        layers += attached(histogram.functions, request)
    return layers, titles(histogram, histogram.title)


def _default_grid(chosen: Chosen) -> Chosen:
    """``COLZ``: ROOT's ``COL``, with the scale that says what the colours mean."""
    if chosen.drawing:
        return chosen
    return chosen._replace(words=chosen.words | {"COL", "Z"})


def _grid(histogram: Histogram, request: Request) -> tuple[list[Any], Frame]:
    chosen = _default_grid(request.chosen)
    scale = _normalised(histogram, chosen)
    grid = Grid(
        histogram.edges(0),
        histogram.edges(1),
        histogram.values() * scale,
        styled(histogram.members, request),
    )
    layers = grid_layers(grid, request.again(chosen))
    return layers, titles(histogram, histogram.title)


def _cloud(histogram: Histogram, request: Request) -> tuple[list[Any], Frame]:
    """A 3-D histogram: a marker at each bin with anything in it, or ``ISO`` its surfaces."""
    values = histogram.values() * _normalised(histogram, request.chosen)
    centres = [axis.centers() for axis in histogram.axes]
    xs, ys, zs = np.meshgrid(*centres, indexing="ij")
    iso = request.chosen.has("ISO")
    keep = np.ones(values.shape, dtype=bool) if iso else values != 0
    layer = Cloud(xs[keep], ys[keep], zs[keep], values[keep], request.palette, iso)
    return [layer], titles(histogram, histogram.title)


#: How each number of dimensions is drawn.
_BY_DIMENSION = {1: _flat, 2: _grid, 3: _cloud}


def histogram(obj: Histogram, request: Request) -> tuple[list[Any], Frame]:
    """A histogram or profile of one, two or three dimensions."""
    return _BY_DIMENSION[len(obj.axes)](obj, request)


# -- graphs --------------------------------------------------------------------------------


def graph_titles(obj: Any) -> Frame:
    """A graph's axis titles, which ROOT keeps on the histogram it draws the frame with."""
    frame = held(obj.members, "fHistogram")
    if isinstance(frame, Histogram):
        return titles(frame, obj.title)
    return Frame(title=obj.title)


def _graph_scatter(obj: Graph, request: Request) -> Scatter:
    zeros = np.zeros_like(obj.x)
    xerr = obj.xerr or (zeros, zeros)
    return Scatter(obj.x, obj.y, xerr[0], xerr[1], obj.layers, styled(obj.members, request))


def default_graph(obj: Any, chosen: Chosen) -> Chosen:
    """``ALP``, or the ``fOption`` the graph was saved with, as ``TGraph::Draw`` takes it."""
    if chosen.drawing:
        return chosen
    saved = str(held(obj.members, "fOption") or "") or "ALP"
    words = choose(saved, "graph").words
    return chosen._replace(words=chosen.words | words)


def graph(obj: Graph, request: Request) -> tuple[list[Any], Frame]:
    """A graph of any kind: its points, their errors in every layer, and its fits."""
    request = request._replace(chosen=default_graph(obj, request.chosen))
    layers = scatter_layers(_graph_scatter(obj, request), request)
    layers += attached(obj.functions, request)
    return layers, graph_titles(obj)


# -- efficiencies and functions ------------------------------------------------------------


def _efficiency_flat(obj: Efficiency, request: Request) -> tuple[list[Any], Frame]:
    """``TEfficiency::CreateGraph``: the bins anything was tried in, with their intervals."""
    chosen = request.chosen
    if not chosen.drawing:
        chosen = chosen._replace(words=chosen.words | {"A", "P"})
    axis = obj.axes[0]
    tried = obj.total.values() != 0
    low, high = obj.errors()
    half = axis.widths()[tried] / 2
    request = request._replace(chosen=chosen)
    points = Scatter(
        axis.centers()[tried],
        obj.values()[tried],
        half,
        half,
        ((low[tried], high[tried]),),
        styled(obj.members, request),
    )
    frame = titles(obj.total, obj.title)
    return scatter_layers(points, request), frame


def _efficiency_grid(obj: Efficiency, request: Request) -> tuple[list[Any], Frame]:
    """``TEfficiency::CreateHistogram``: the efficiency of each cell, shaded."""
    chosen = _default_grid(request.chosen)
    values = np.where(obj.total.values() != 0, obj.values(), 0.0)
    grid = Grid(obj.total.edges(0), obj.total.edges(1), values, styled(obj.members, request))
    return grid_layers(grid, request.again(chosen)), titles(obj.total, obj.title)


def efficiency(obj: Efficiency, request: Request) -> tuple[list[Any], Frame]:
    """An efficiency of one axis as points with their intervals, of two as a shaded grid."""
    if len(obj.axes) > 2:
        raise UnsupportedFeatureError(
            f"{obj.name!r} is an efficiency of {len(obj.axes)} axes, which has no picture "
            f"here; slice its passed and total histograms down to two and divide those"
        )
    if len(obj.axes) == 1:
        return _efficiency_flat(obj, request)
    return _efficiency_grid(obj, request)


def _function_grid(obj: Function, request: Request) -> tuple[list[Any], Frame]:
    """``TF2::Paint``: the function at the centre of ``fNpx`` by ``fNpy`` cells, as ``cont3``."""
    chosen = request.chosen
    if not chosen.drawing:
        chosen = chosen._replace(words=chosen.words | {"CONTL"})
    (xlow, xhigh), (ylow, yhigh) = obj.range
    xedges = np.linspace(xlow, xhigh, int(held(obj.members, "fNpx") or 30) + 1)
    yedges = np.linspace(ylow, yhigh, int(held(obj.members, "fNpy") or 30) + 1)
    xs, ys = np.meshgrid((xedges[1:] + xedges[:-1]) / 2, (yedges[1:] + yedges[:-1]) / 2)
    values = np.asarray(obj(xs.T.ravel(), ys.T.ravel()), dtype=np.float64)
    values = values.reshape(len(xedges) - 1, len(yedges) - 1)
    grid = Grid(xedges, yedges, values, styled(obj.members, request))
    return grid_layers(grid, request.again(chosen)), Frame(title=obj.title)


def function(obj: Function, request: Request) -> tuple[list[Any], Frame]:
    """A ``TF1`` as its curve, a ``TF2`` as its contours; a ``TF3`` has no picture here."""
    if obj.dimensions == 2:
        return _function_grid(obj, request)
    if obj.dimensions != 1:
        raise UnsupportedFeatureError(
            f"{obj.name!r} is a function of {obj.dimensions} variables, which has no picture "
            f"here; evaluate it on a grid and slice that down to two"
        )
    return _sampled(obj, request), Frame(title=obj.title)


def kind_of(obj: Any) -> str:
    """The kind of thing a histogram's, efficiency's or function's option is read as.

    That depends on its dimensions: an efficiency of one axis is drawn as a
    graph is, and a ``TF2`` as a 2-D histogram.
    """
    if isinstance(obj, Histogram):
        return HISTOGRAM_KINDS[len(obj.axes)]
    if isinstance(obj, Efficiency):
        return {1: "graph", 2: HISTOGRAM_KINDS[2]}.get(len(obj.axes), HISTOGRAM_KINDS[3])
    return HISTOGRAM_KINDS[2] if obj.dimensions == 2 else "function"
