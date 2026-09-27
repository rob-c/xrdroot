"""``RooAbsPdf::plotOn`` and ``RooAbsReal::plotOn``: a function's curve on a frame.

A density is drawn normalised to the data already on the frame - its
events, times the frame's bin width, so curve and points share a scale -
or, with nothing there, to one event per bin width. ``Normalization(s)``
multiplies that, ``Components("bkg")`` draws only some of a sum's terms,
``Range("signal")`` draws only over a range and ``NormRange`` normalises to
the data in one. The observables of the data that are not the frame's are
integrated out: the curve is a projection, called ``<name>_Norm[x]``.
"""

from __future__ import annotations

import fnmatch
from typing import Any

import numpy as np

from .. import selection
from ..cmdargs import Commands, commands
from ..messages import INFO, log
from .curve import RooCurve, sample

__all__ = ["plot_pdf", "plot_function", "style"]

#: The colour words PyROOT's keywords take: ``LineColor="r"``.
COLOURS = {"r": 632, "b": 600, "g": 416, "k": 1, "m": 616, "c": 432, "y": 400, "w": 0}
#: The plotting options that set an attribute, and the member each sets.
STYLE = {
    "LineColor": ("TAttLine", "fLineColor"), "LineStyle": ("TAttLine", "fLineStyle"),
    "LineWidth": ("TAttLine", "fLineWidth"), "FillColor": ("TAttFill", "fFillColor"),
    "FillStyle": ("TAttFill", "fFillStyle"), "MarkerColor": ("TAttMarker", "fMarkerColor"),
    "MarkerStyle": ("TAttMarker", "fMarkerStyle"), "MarkerSize": ("TAttMarker", "fMarkerSize"),
}  # fmt: skip


def colour(value: Any) -> Any:
    return COLOURS.get(value, value) if isinstance(value, str) else value


def style(graph: Any, options: Commands) -> None:
    """Set the line, fill and marker attributes the options name."""
    for name, (group, member) in STYLE.items():
        if name in options:
            value = colour(options.get(name))
            graph._core[group][member] = float(value) if member == "fMarkerSize" else int(value)


def _selected(pdf: Any, options: Commands) -> tuple[set[str] | None, str]:
    """``plotOnCompSelect``: the components to draw - direct and indirect - and the name suffix."""
    spec = options.get("Components")
    if spec is None:
        return None, ""
    branches = [node for node in pdf._walk() if not node.isFundamental()]
    if isinstance(spec, str):
        patterns = [p for p in spec.split(",") if p]
        direct = [b for b in branches if any(fnmatch.fnmatchcase(b.GetName(), p) for p in patterns)]
        suffix = f"_Comp[{spec}]"
    else:
        wanted = {one.GetName() for one in _items(spec)}
        direct = [b for b in branches if b.GetName() in wanted]
        suffix = "_Comp[" + ",".join(one.GetName() for one in _items(spec)) + "]"
    log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) directly selected PDF components: "
        f"({','.join(b.GetName() for b in direct)})")  # fmt: skip
    indirect = _indirect(pdf, branches, direct)
    log(pdf, INFO, "Plotting", f"RooAbsPdf::plotOn({pdf.GetName()}) indirectly selected PDF components: "
        f"({','.join(b.GetName() for b in indirect)})")  # fmt: skip
    return {b.GetName() for b in direct + indirect}, suffix


def _indirect(pdf: Any, branches: list[Any], direct: list[Any]) -> list[Any]:
    names = {b.GetName() for b in direct}
    found = []
    for branch in branches:
        if branch.GetName() in names or branch is pdf:
            continue
        below = any(d.dependsOn(branch) for d in direct)
        above = branch.dependsOn(direct)
        if below or above:
            found.append(branch)
    return found


def _items(spec: Any) -> list[Any]:
    from ..collections import as_list

    return as_list(spec)


def _range(frame: Any, options: Commands) -> tuple[float, float, str | None]:
    """The part of the frame to draw: a named range, two numbers, or all of it."""
    found = options.args("Range")
    var = frame.getPlotVar()
    if not found:
        return frame.GetXmin(), frame.GetXmax(), None
    if isinstance(found[0], str):
        return var.getMin(found[0]), var.getMax(found[0]), found[0]
    return float(found[0]), float(found[1]), None


def _norm_vars(pdf: Any, frame: Any) -> frozenset[str]:
    """The observables the curve is normalised over: the frame's variable and the data's others."""
    frame.update_norm_vars([frame.getPlotVar()])
    names = {one.GetName() for one in frame.norm_vars or ()}
    return frozenset(names & pdf.dependents()) | {frame.getPlotVar().GetName()}


def plot_pdf(pdf: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.plotOn(frame, options...)``: add the density's curve - or curves, one per range - to ``frame``."""
    from .ranges import plan

    options = commands(args, kwargs)
    options.warn_duplicates(f"RooAbsPdf::plotOn({pdf.GetName()})")
    nset = _norm_vars(pdf, frame)
    _announce_plot(pdf, frame, nset)
    chosen, suffix = _selected(pdf, options)
    made = plan(pdf, frame, options, nset)
    scale = made.scale
    if made.post_scale:
        scale /= _range_fraction(pdf, frame, nset, made.norm_range, made.pieces)
    for low, high in made.pieces:
        _add_curve(pdf, frame, options, nset, scale, chosen, suffix, (low, high, made.wings))
    return frame


def _range_fraction(pdf: Any, frame: Any, nset: frozenset[str], rng: Any,
                    pieces: list[tuple[float, float]]) -> float:  # fmt: skip
    """The fraction of the curve's projection in the normalisation range: ``postRangeFracScale``."""
    var = frame.getPlotVar()
    if not rng:
        var.setRange("plotRange", pieces[0][0], pieces[0][1])
        rng = "plotRange"
    name = frozenset([var.GetName()])
    return float(np.asarray(pdf.fraction(name | (nset - name), {}, nset, rng)))


def _announce_plot(pdf: Any, frame: Any, nset: frozenset[str]) -> None:
    """What RooFit says when it plots: the projection, and the two integrals it makes for it."""
    from ..integration import announce, integral_name

    projected = nset - {frame.getPlotVar().GetName()}
    if projected:
        order = [one.GetName() for one in pdf.leaves() if one.GetName() in projected]
        log(pdf, INFO, "Plotting", f"RooAbsReal::plotOn({pdf.GetName()}) plot on "
            f"{frame.getPlotVar().GetName()} integrates over variables ({','.join(order)})")  # fmt: skip
    announce(pdf, nset)
    if not projected:
        announce(pdf, nset)
        return
    norm = ",".join(one.GetName() for one in pdf.leaves() if one.GetName() in nset)
    announce(pdf, projected, label=f"{integral_name(pdf, projected, None)}_Norm[{norm}]")


def plot_function(func: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``RooAbsReal::plotOn``: a function, drawn as it is - scaled only if asked."""
    options = commands(args, kwargs)
    scale = float(options.get("Normalization", 0, 1.0))
    low, high, _ = _range(frame, options)
    return _add_curve(func, frame, options, frozenset(), scale, None, "", (low, high, "Range" not in options))


def _add_curve(func: Any, frame: Any, options: Commands, nset: frozenset[str], scale: float,
               chosen: set[str] | None, suffix: str, piece: Any = None) -> Any:  # fmt: skip
    """Sample the projection over the frame's variable and put the curve on the frame."""
    var = frame.getPlotVar()
    low, high, wings = piece if piece is not None else (*_range(frame, options)[:2], True)
    projected = frozenset(nset - {var.GetName()})
    name = var.GetName()

    def curve_at(xs: Any) -> Any:
        ctx = {name: np.asarray(xs, dtype=np.float64)}
        if not nset:
            return func.compute(ctx) * np.ones(len(xs)) * scale
        if projected:
            return func.fraction(projected, ctx, nset, None) * scale
        return func.value(ctx, nset) * scale

    precision = float(options.get("Precision", 0, 1e-3))
    with selection.selecting(chosen):
        xs, ys = sample(curve_at, low, high, frame.GetNbinsX(), precision, wings)
    norm = ",".join(sorted(nset, key=lambda n: [v.GetName() for v in func.leaves()].index(n)))
    label = f"{func.GetName()}_Norm[{norm}]" if nset else func.GetName()
    curve = RooCurve(label + suffix, f"Projection of {func.GetTitle()}", xs, ys)
    if "Name" in options:
        curve.SetName(str(options.get("Name")))
    style(curve, options)
    option = str(options.get("DrawOption", 0, "L"))
    invisible = bool(options.get("Invisible", 0, False))
    frame.add_plotable(curve, option, invisible)
    if "MoveToBack" in options:
        frame.items.insert(0, frame.items.pop())
    return frame
