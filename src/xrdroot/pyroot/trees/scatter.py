"""ROOT's tree scatter keeps its points separate from the histogram behind them.

The plain Python draw API has no pad and defaults to filling a histogram.
Here a single formula loop feeds both the named histogram, when requested,
and the graph ROOT leaves in the pad; the frame itself carries no new fills.
"""

from __future__ import annotations

from typing import Any

from ... import drawspec, treedraw
from ._base import hooks


class _Scatter(treedraw.Scatter):
    """Collect points while the usual filler determines and fills their axes."""

    def __init__(self, histogram: treedraw.Filling) -> None:
        super().__init__()
        self.histogram = histogram

    def feed(self, columns: list[Any], weights: Any) -> None:
        super().feed(columns, weights)
        self.histogram.feed(columns, weights)


def wanted(text: str, option: str) -> bool:
    """Apply TSelectorDraw's graph test before the ordinary histogram draw."""
    if len(drawspec.split_names(text.split(">>")[0])) != 2:
        return False
    lowered = option.lower()
    if "prof" in lowered or any(style in lowered for style in drawspec.BINNED):
        return False
    remaining = lowered.replace("same", "")
    return not remaining or "same" in lowered or any(mark in remaining for mark in "pl*")


def _filling(
    source: Any, chosen: Any, found: Any, estimate: int, formulas: Any, option: str
) -> treedraw.Filling:
    """SAME borrows the pad's axes; ROOT leaves a new internal histogram at 0..1."""
    bins = None
    if "same" in option.lower() and chosen.existing is None:
        counts = treedraw.axes_for(found, None, "hist", 2).nbins
        bins = [(count, 0.0, 1.0) for count in counts]
        found = found._replace(binning=None)
    made = treedraw._filler(source, chosen, found, bins, estimate, formulas)
    assert isinstance(made, treedraw.Filling)
    return made


def _styled(tree: Any, made: Any) -> Any:
    wrapped = hooks.wrap(made)
    tree.CopyTAttLine(wrapped)
    tree.CopyTAttFill(wrapped)
    tree.CopyTAttMarker(wrapped)
    return wrapped


def _graph(tree: Any, graph: Any, option: str) -> Any:
    """The tree's attributes reach the points; a star requests ROOT's marker 3."""
    points = _styled(tree, graph)
    points.SetEditable(False)
    if "*" in option:
        points.SetMarkerStyle(3)
    return points


def _empty(kept: Any, option: str) -> Any:
    """An empty selection replaces the previous picture unless SAME or goff preserves it."""
    lowered = option.lower()
    if "goff" not in lowered and "same" not in lowered:
        hooks.draw(kept, "axis")
    return kept


def _paint(
    tree: Any, histogram: Any, frame: Any, graph: Any, option: str, named: bool
) -> Any:
    """Named histograms keep their fills while their drawn copy hides statistics."""
    kept = _styled(tree, histogram)
    if not named:
        kept.SetStats(False)
    if not len(graph.x):
        return _empty(kept, option)
    points = _graph(tree, graph, option)
    if "goff" in option.lower():
        return kept
    return _visible(tree, kept, frame, points, option, named)


def _visible(tree: Any, kept: Any, frame: Any, points: Any, option: str, named: bool) -> Any:
    """The pad keeps axes separately unless the graph explicitly draws its own."""
    if "same" not in option.lower():
        background = _styled(tree, frame)
        background.SetStats(False)
        hooks.draw(background, "axis")
    drawing = option or "p"
    if drawing.lower() == "same":
        drawing = "p"
    hooks.draw(points, drawing.replace("*", "p"))
    own_axes = "a" in option.lower().replace("same", "") and not named
    return points.GetHistogram() if own_axes else kept


def _frame(histogram: Any, old: Any, named: bool) -> Any:
    """Only a kept named histogram needs a separate pad copy."""
    if not named:
        histogram.reset()
        return histogram
    if old is not None:
        return old
    frame = histogram.copy()
    frame.reset()
    return frame


def draw(
    tree: Any, text: str, selection: str, option: str, count: int | None, first: int
) -> int | None:
    """Leave ROOT's Graph and frame without evaluating the tree a second time."""
    treedraw._checked_range(first, count, treedraw.STEP, tree._estimate)
    source, registry = tree._view(), hooks.registry()
    chosen, found = treedraw.plan(text, selection, option, None, None, registry)
    if chosen.kind == "prof":
        return None
    chosen = chosen._replace(kind="hist", labels=chosen.parts if found.name is None else None)
    formulas, weighing = treedraw._compiled(
        source, chosen, selection, tree._weight, tree._aliases or None
    )
    filling = _filling(source, chosen, found, tree._estimate, formulas, option)
    old = filling.histogram.copy() if filling.histogram is not None else None
    scatter = _Scatter(filling)
    selected = treedraw._run(
        source, formulas, scatter, 2, weighing, (first, count, treedraw.STEP)
    )
    histogram = filling.finish()
    named = found.name is not None
    frame = _frame(histogram, old, named)
    if named or "goff" in option.lower():
        registry[chosen.name] = histogram
    graph = scatter.finish("Graph", "Graph")
    tree._draw_histogram = _paint(tree, histogram, frame, graph, option, named)
    tree._drawn = [graph.y, graph.x]
    return selected
