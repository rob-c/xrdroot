"""The things a pad draws that are data: histograms, graphs, stacks and functions.

Each is drawn by its draw option, the way ROOT's painters read it. A
histogram's outline is drawn by its own :meth:`~xrdroot.Histogram.plot`,
and a graph's points and bars by :meth:`~xrdroot.Graph.plot`, each given
the colours, widths and markers the object's attributes say; the rest of
ROOT's pictures - error bars and bands, bars, markers alone, the numbers
in each bin, a colour scale - are drawn here. What ROOT hangs on the object
is drawn with it: the functions fitted to it, its stats box, and the
palette of a ``COLZ``.

Two approximations are made, and said here rather than hidden: a
two-dimensional histogram drawn as ``LEGO``, ``SURF`` or with no option at
all is drawn as ``COL``, having no flat picture of its own in matplotlib,
and a curve (``C``) is drawn through its points with straight lines.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import ROOTError
from ..function import Function
from ..graph import Graph
from ..hist import Histogram
from ..profile import Profile
from ..stacks import MultiGraph, Stack
from .model import lookup
from .options import ERRORS, PICTURES_2D, SHAPES, graph_option, histogram_option, strip_same
from .paves import stats_box
from .scene import Scene
from .shapes import patch_style
from .statbox import NOT_DRAW, default_stats, shows_stats

__all__ = ["DATA", "paint_data"]

#: The half-width of a bin's horizontal error bar, as ``gStyle->GetErrorX()``.
ERROR_X = 0.5
#: How long the end of an error bar is, in pixels, as ``gStyle->GetEndErrorSize()``.
END_ERROR = 2.0
#: How many points a function is drawn with when it says nothing: ``TF1``'s ``fNpx``.
NPX = 100
#: Where a histogram's colour scale goes, without a saved ``TPaletteAxis``: how
#: far right of the frame it starts, and how wide it is, as fractions of the pad.
PALETTE_GAP, PALETTE_WIDTH = 0.005, 0.05
#: The keywords of an error bar, which a graph drawn without its bars leaves out.
BAR_KEYWORDS = frozenset({"fmt", "ecolor", "elinewidth", "capsize"})
#: The error options drawn as boxes or a band rather than bars.
BANDS = frozenset({"E2", "E3", "E4", "E5", "E6"})


# -- histograms ------------------------------------------------------------------


def _stairs_style(scene: Scene, h: Any) -> dict[str, Any]:
    """The keywords :meth:`Histogram.plot` draws an outline with, from ``h``'s attributes.

    A fill of colour 0 is no fill at all, as ROOT draws a histogram.
    """
    line = scene.line(h)
    style: dict[str, Any] = {
        "edgecolor": line["color"],
        "linewidth": line["linewidth"],
        "linestyle": line["linestyle"],
        "fill": False,
    }
    filled = scene.fill(h)
    if filled is not None and int(lookup(h, "fFillColor", 0)):
        style["fill"] = True
        style["facecolor"] = filled["facecolor"]
        if filled["hatch"]:
            style["hatch"] = filled["hatch"]
            style["facecolor"] = "none"
            style["edgecolor"] = filled["hatchcolor"]
    return style


def _outline(scene: Scene, h: Histogram) -> None:
    title = scene.ax.get_title()
    h.plot(ax=scene.ax, **_stairs_style(scene, h))
    scene.ax.set_title(title)  # the pad's title is its own primitive, not the axes'


def _visible(h: Histogram, words: frozenset[str]) -> np.ndarray[Any, Any]:
    """The bins drawn with bars or markers: every one for ``E0``, else those not empty."""
    values, errors = h.values(), h.errors()
    if "E0" in words or "P0" in words:
        return np.ones(len(values), dtype=bool)
    return (values != 0) | (errors != 0)


def _bars(scene: Scene, h: Histogram, words: frozenset[str], markers: bool) -> None:
    """Error bars, with a marker at each point, and ends on them for ``E1``."""
    axis = h.axes[0]
    keep = _visible(h, words)
    line = scene.line(h)
    style = scene.marker(h) if markers else {"marker": "none"}
    scene.ax.errorbar(
        axis.centers()[keep],
        h.values()[keep],
        yerr=h.errors()[keep],
        xerr=(axis.widths() * ERROR_X)[keep],
        linestyle="none",
        ecolor=line["color"],
        elinewidth=line["linewidth"],
        capsize=2 * END_ERROR * 0.72 if "E1" in words else 0.0,
        **style,
    )


def _band(scene: Scene, h: Histogram, words: frozenset[str]) -> None:
    """``E2``'s boxes round each point, or ``E3`` and ``E4``'s band through them."""
    axis = h.axes[0]
    values, errors = h.values(), h.errors()
    style = patch_style(scene, h, outline=False)
    if "E2" in words:
        scene.ax.bar(
            axis.centers(), 2 * errors, bottom=values - errors, width=axis.widths(), **style
        )
    else:
        scene.ax.fill_between(axis.centers(), values - errors, values + errors, **style)


def _points(scene: Scene, h: Histogram, words: frozenset[str]) -> None:
    """Markers alone (``P``), or a line through the points (``L``, ``C``)."""
    axis = h.axes[0]
    if words & {"P", "P0", "*H"}:
        keep = _visible(h, words)
        style = scene.marker(h)
        if "*H" in words:
            style["marker"] = "*"
        scene.ax.plot(axis.centers()[keep], h.values()[keep], linestyle="none", **style)
    if words & {"L", "C"}:
        scene.ax.plot(axis.centers(), h.values(), **scene.line(h))


def _bar_chart(scene: Scene, h: Histogram) -> None:
    """``B`` and ``BAR``: a bar per bin, of ``fBarWidth`` of it, ``fBarOffset`` along."""
    axis = h.axes[0]
    width = float(lookup(h, "fBarWidth", 1000)) / 1000.0
    offset = float(lookup(h, "fBarOffset", 0)) / 1000.0
    scene.ax.bar(
        axis.edges()[:-1] + axis.widths() * offset,
        h.values(),
        width=axis.widths() * width,
        align="edge",
        **patch_style(scene, h),
    )


def _numbers(scene: Scene, xs: Any, ys: Any, values: Any) -> None:
    """``TEXT``: each bin's content written at its middle."""
    style = scene.text(None, None, scene.text_points(0.02))
    style["ha"], style["va"] = "center", "bottom"
    for x, y, value in zip(xs, ys, values):
        if value:
            scene.ax.text(x, y, f"{value:g}", zorder=5, **style)


def _errors_by_default(h: Histogram, words: frozenset[str]) -> bool:
    """Whether a histogram is drawn with error bars without being asked.

    ROOT does for a profile, and for a histogram keeping the squares of its
    weights; ``HIST`` says not to, and a picture of its own replaces it.
    """
    if "HIST" in words or words & SHAPES:
        return False
    return isinstance(h, Profile) or h.weighted


def _error_picture(scene: Scene, h: Histogram, words: frozenset[str]) -> None:
    """The errors, as bars with markers or as boxes and bands, by which ``E`` it is."""
    if words & BANDS:
        _band(scene, h, words)
    else:
        _bars(scene, h, frozenset(words | {"E"}), markers=True)


def _histogram_1d(scene: Scene, h: Histogram, words: frozenset[str]) -> None:
    errors = words & ERRORS or _errors_by_default(h, words)
    if "HIST" in words or not (errors or words & SHAPES):
        _outline(scene, h)
    if errors:
        _error_picture(scene, h, words)
    _points(scene, h, words)
    if words & {"B", "BAR"}:
        _bar_chart(scene, h)
    if "TEXT" in words:
        _numbers(scene, h.axes[0].centers(), h.values(), h.values())


def _colormap(scene: Scene) -> Any:
    """The palette, with what is below its lowest value - an empty bin - not drawn."""
    cmap = scene.colors.colormap()
    cmap.set_under((0, 0, 0, 0))
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def _norm(scene: Scene, h: Histogram) -> Any:
    """The scale of the colours: ``fMinimum`` and ``fMaximum`` if set, else the bins'.

    As in ROOT, a bin at zero is not drawn when the lowest bin is not below
    it, and a pad drawn ``SetLogz`` scales the colours logarithmically.
    """
    from matplotlib.colors import LogNorm, Normalize

    values = h.values()
    positive = values[values > 0]
    low = _set(lookup(h, "fMinimum")) or (float(positive.min()) if positive.size else 1.0)
    high = _set(lookup(h, "fMaximum")) or float(values.max(initial=low))
    if values.min(initial=0.0) < 0 and not scene.pad.logz:
        low = float(values.min())
    if scene.pad.logz:
        return LogNorm(vmin=low, vmax=max(high, low * 10))
    return Normalize(vmin=low, vmax=max(high, low))


def _set(value: Any) -> float | None:
    """A ``fMinimum`` or ``fMaximum``, or ``None`` for ROOT's ``-1111``, which is unset."""
    if value is None or float(value) == -1111:
        return None
    return float(value)


def _palette(scene: Scene, mesh: Any, h: Histogram) -> None:
    """The colour scale of ``COLZ``, where its ``TPaletteAxis`` was, or in the right margin."""
    saved = [one for one in h.functions if getattr(one, "classname", "") == "TPaletteAxis"]
    _left, right, bottom, top = scene.pad.margins
    corners = (1 - right + PALETTE_GAP, bottom, 1 - right + PALETTE_GAP + PALETTE_WIDTH, 1 - top)
    if saved:
        corners = tuple(
            float(saved[0].get(name, 0.0)) for name in ("fX1NDC", "fY1NDC", "fX2NDC", "fY2NDC")
        )  # type: ignore[assignment]
    x, y, w, height = scene.box
    rect = (
        x + corners[0] * w,
        y + corners[1] * height,
        (corners[2] - corners[0]) * w,
        (corners[3] - corners[1]) * height,
    )
    cax = scene.figure.add_axes(rect, label=f"{scene.pad.name} palette")
    scene.figure.colorbar(mesh, cax=cax)
    axis = lookup(h, "fZaxis") or {}
    cax.tick_params(direction="in", labelsize=scene.text_points(lookup(axis, "fLabelSize", 0.035)))


def _histogram_2d(scene: Scene, h: Histogram, words: frozenset[str]) -> None:
    pictures = words & PICTURES_2D
    if not pictures or pictures & {"COL", "COLZ"}:
        title = scene.ax.get_title()
        h.plot(ax=scene.ax, cmap=_colormap(scene), norm=_norm(scene, h))
        scene.ax.set_title(title)
        if words & {"COLZ", "Z"}:
            _palette(scene, scene.ax.collections[-1], h)
    if "BOX" in words:
        _boxes(scene, h)
    if words & {"CONT", "CONTZ"}:
        xs, ys = h.axes[0].centers(), h.axes[1].centers()
        scene.ax.contour(xs, ys, h.values().T, cmap=scene.colors.colormap())
    if "TEXT" in words:
        xs, ys = np.meshgrid(h.axes[0].centers(), h.axes[1].centers(), indexing="ij")
        _numbers(scene, xs.ravel(), ys.ravel(), h.values().ravel())


def _boxes(scene: Scene, h: Histogram) -> None:
    """``BOX``: a box in each bin, as big across as its content is of the largest."""
    from matplotlib.patches import Rectangle

    values = np.abs(h.values())
    largest = float(values.max(initial=0.0)) or 1.0
    style = patch_style(scene, h)
    for (i, j), value in np.ndenumerate(values):
        if not value:
            continue
        (xlo, xhi), (ylo, yhi) = h.axes[0][i], h.axes[1][j]
        scale = value / largest
        dx, dy = (xhi - xlo) * scale / 2, (yhi - ylo) * scale / 2
        cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
        scene.ax.add_patch(Rectangle((cx - dx, cy - dy), 2 * dx, 2 * dy, **style))


def _hung(scene: Scene, obj: Any, words: frozenset[str], option: str) -> None:
    """What ROOT draws with an object: its fitted functions, and its stats box."""
    functions = obj.functions
    if "HIST" not in words:
        _fits(scene, functions)
    saved = [one for one in functions if getattr(one, "classname", "") == "TPaveStats"]
    for box in saved:
        stats_box(scene, box, "")
    if not saved and _stats_made(scene, obj, option):
        default_stats(scene, obj)


def _fits(scene: Scene, functions: list[Any]) -> None:
    """The functions hung on an object, but for one told ``kNotDraw``."""
    for function in functions:
        if isinstance(function, Function) and not int(lookup(function, "fBits", 0)) & NOT_DRAW:
            paint_function(scene, function, "SAME")


def _stats_made(scene: Scene, obj: Any, option: str) -> bool:
    """Whether drawing ``obj`` now would make a stats box, as ROOT does for a histogram.

    A pad drawn before it was saved kept the box it made, if any, so this
    is only for one saved without being drawn.
    """
    return isinstance(obj, Histogram) and not scene.pad.painted and shows_stats(obj, option)


def paint_histogram(scene: Scene, h: Histogram, option: str) -> None:
    """A histogram of one or two dimensions, by its draw option."""
    if len(h.axes) > 2:
        scene.skipped.append(f"{h.classname} {h.name!r} (three dimensions have no flat picture)")
        return
    words = histogram_option(option)
    if len(h.axes) == 1:
        _histogram_1d(scene, h, words)
    else:
        _histogram_2d(scene, h, words)
    _hung(scene, h, words, option)


# -- graphs ----------------------------------------------------------------------


def _graph_style(scene: Scene, g: Any, letters: frozenset[str]) -> dict[str, Any]:
    """The keywords :meth:`Graph.plot` draws with, from its attributes and letters."""
    line = scene.line(g)
    style: dict[str, Any] = {"fmt": "none" if not letters & set("LCP*") else ""}
    style.update(scene.marker(g) if letters & {"P", "*"} else {"marker": "none"})
    if "*" in letters:
        style["marker"] = "*"
    style["linestyle"] = line["linestyle"] if letters & {"L", "C"} else "none"
    style["color"] = line["color"]
    style["linewidth"] = line["linewidth"]
    style["ecolor"] = line["color"]
    style["elinewidth"] = line["linewidth"]
    style["capsize"] = 0.0 if "Z" in letters else END_ERROR * 0.72
    return style


def _graph_bands(scene: Scene, g: Graph, letters: frozenset[str]) -> None:
    """Errors as boxes (``2``) or a band (``3``, ``4``) rather than bars."""
    low, high = g.yerr if g.yerr is not None else (np.zeros(len(g)), np.zeros(len(g)))
    style = patch_style(scene, g, outline=False)
    if "2" in letters:
        xlow, xhigh = g.xerr if g.xerr is not None else (np.zeros(len(g)), np.zeros(len(g)))
        scene.ax.bar(
            g.x - xlow, low + high, bottom=g.y - low, width=xlow + xhigh, align="edge", **style
        )
    else:
        scene.ax.fill_between(g.x, g.y - low, g.y + high, **style)


def _graph_areas(scene: Scene, g: Graph, letters: frozenset[str]) -> None:
    """``F``, the area the points enclose, and ``B``, a bar at each of them."""
    style = patch_style(scene, g)
    if "F" in letters:
        scene.ax.fill(g.x, g.y, **style)
    if "B" in letters and len(g):
        spacing = float(np.ptp(g.x)) / max(len(g) - 1, 1) or 1.0
        scene.ax.bar(
            g.x, g.y, width=spacing * float(lookup(g, "fBarWidth", 1000) or 1000) / 2000, **style
        )


def paint_graph(scene: Scene, g: Graph, option: str) -> None:
    """A graph, by the letters of its draw option."""
    letters = graph_option(option)
    banded = bool(letters & set("2345"))
    title = scene.ax.get_title()
    style = _graph_style(scene, g, letters)
    if "X" in letters or banded:  # the points without their bars
        plain = {key: value for key, value in style.items() if key not in BAR_KEYWORDS}
        if letters & set("LCP*"):
            scene.ax.plot(g.x, g.y, **plain)
    else:
        g.plot(ax=scene.ax, **style)
    scene.ax.set_title(title)
    if banded and "X" not in letters:
        _graph_bands(scene, g, letters)
    _graph_areas(scene, g, letters)
    _hung(scene, g, frozenset(), option)


def paint_multigraph(scene: Scene, mg: MultiGraph, option: str) -> None:
    """A ``TMultiGraph``: each graph by its own option, or the multigraph's without its axes."""
    shared = strip_same(option).replace("A", "")
    for graph, own in zip(mg, _held_options(mg, "fGraphs")):
        paint_graph(scene, graph, own or shared)
    _hung(scene, mg, frozenset(), option)


def _held_options(held: Any, member: str) -> list[str]:
    """The option each thing in a multigraph or stack was added with, ``""`` for none."""
    listed = held.members.get(member) or []
    return list(getattr(listed, "options", [""] * len(held)))


def paint_stack(scene: Scene, stack: Stack, option: str) -> None:
    """A ``THStack``: stacked, the top drawn first so each fill shows, or ``NOSTACK``."""
    if "NOSTACK" in histogram_option(option) or any(len(h.axes) != 1 for h in stack):
        shared = option.upper().replace("NOSTACK", "")
        for h, own in zip(stack, _held_options(stack, "fHists")):
            paint_histogram(scene, h, (own or shared) + " SAME")
        return
    totals = np.cumsum([h.values() for h in stack], axis=0) if len(stack) else []
    for h, total in reversed(list(zip(stack, totals))):
        scene.ax.stairs(total, h.edges(), **_stairs_style(scene, h))


# -- functions -------------------------------------------------------------------


def paint_function(scene: Scene, f: Function, option: str) -> None:
    """A ``TF1``, drawn over its range with ``fNpx`` points, as a line."""
    if f.dimensions != 1:
        scene.skipped.append(f"{f.classname} {f.name!r} (a function of {f.dimensions} variables)")
        return
    low, high = (float(end) for end in f.range[:2])
    xs = np.linspace(low, high, int(lookup(f, "fNpx", NPX) or NPX) + 1)
    try:
        ys = np.asarray(f(xs), dtype=float)
    except ROOTError as why:
        scene.skipped.append(f"{f.classname} {f.name!r} ({why})")
        return
    scene.ax.plot(xs, ys, **scene.line(f))


#: How each kind of data draws, by the Python class it comes back as.
DATA: tuple[tuple[type, Any], ...] = (
    (Histogram, paint_histogram),
    (Graph, paint_graph),
    (MultiGraph, paint_multigraph),
    (Stack, paint_stack),
    (Function, paint_function),
)


def paint_data(scene: Scene, obj: Any, option: str) -> bool:
    """Draw ``obj`` if it is data, and say whether it was."""
    for kind, paint in DATA:
        if isinstance(obj, kind):
            paint(scene, obj, option)
            return True
    return False
