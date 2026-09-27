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
from ..cmdargs import Commands
from ..messages import INFO, log
from .curve import RooCurve, sample

__all__ = ["plot_pdf", "plot_function", "style"]

#: The colour words PyROOT's keywords take: ``LineColor="r"``.
COLOURS = {"r": 632, "b": 600, "g": 416, "k": 1, "m": 616, "c": 432, "y": 400, "w": 0}
#: The plotting options that set an attribute, and the member each sets.
STYLE = {
    "LineColor": ("TAttLine", "fLineColor"),
    "LineStyle": ("TAttLine", "fLineStyle"),
    "LineWidth": ("TAttLine", "fLineWidth"),
    "FillColor": ("TAttFill", "fFillColor"),
    "FillStyle": ("TAttFill", "fFillStyle"),
    "MarkerColor": ("TAttMarker", "fMarkerColor"),
    "MarkerStyle": ("TAttMarker", "fMarkerStyle"),
    "MarkerSize": ("TAttMarker", "fMarkerSize"),
}


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
    direct, suffix = _direct(branches, spec)
    log(
        pdf,
        INFO,
        "Plotting",
        f"RooAbsPdf::plotOn({pdf.GetName()}) directly selected PDF components: "
        f"({','.join(b.GetName() for b in direct)})",
    )
    indirect = _indirect(pdf, branches, direct)
    log(
        pdf,
        INFO,
        "Plotting",
        f"RooAbsPdf::plotOn({pdf.GetName()}) indirectly selected PDF components: "
        f"({','.join(b.GetName() for b in indirect)})",
    )
    return {b.GetName() for b in direct + indirect}, suffix


def _direct(branches: list[Any], spec: Any) -> tuple[list[Any], str]:
    """The components chosen by name patterns - each pattern in turn, as ``selectByName``
    matches them - or by the objects themselves, and the curve's name suffix."""
    if not isinstance(spec, str):
        wanted = [one.GetName() for one in _items(spec)]
        return [b for b in branches if b.GetName() in wanted], "_Comp[" + ",".join(wanted) + "]"
    direct: list[Any] = []
    for pattern in filter(None, spec.split(",")):
        direct += [b for b in _matching(branches, pattern) if b not in direct]
    return direct, f"_Comp[{spec}]"


def _matching(branches: list[Any], pattern: str) -> list[Any]:
    return [b for b in branches if fnmatch.fnmatchcase(b.GetName(), pattern)]


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


def _norm_vars(pdf: Any, frame: Any) -> frozenset[str]:
    """The observables the curve is normalised over: the frame's variable and the data's others."""
    frame.update_norm_vars([frame.getPlotVar()])
    names = {one.GetName() for one in frame.norm_vars or ()}
    return frozenset(names & pdf.dependents()) | {frame.getPlotVar().GetName()}


def plot_pdf(pdf: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``pdf.plotOn(frame, options...)``: RooFit's own sequence, :func:`.pdfplot.pdf_plot`."""
    from .cmdlist import CmdList
    from .pdfplot import pdf_plot

    return pdf_plot(pdf, frame, CmdList.of(args, kwargs))


def _range_fraction(
    pdf: Any, frame: Any, nset: frozenset[str], rng: Any, pieces: list[tuple[float, float]]
) -> float:
    """The fraction of the curve's projection in the normalisation range: ``postRangeFracScale``."""
    var = frame.getPlotVar()
    if not rng:
        var.setRange("plotRange", pieces[0][0], pieces[0][1])
        rng = "plotRange"
    name = frozenset([var.GetName()])
    return float(np.asarray(pdf.fraction(name | (nset - name), {}, nset, rng)))


def _announce_plot(pdf: Any, frame: Any, nset: frozenset[str], seen: Any) -> None:
    """What RooFit says when it plots: the projection - in the frame's order - and the
    integral it makes for it."""
    from ..integration import announce
    from .projections import announce_average

    plot_var = frame.getPlotVar().GetName()
    order = list(seen.projected)
    projected = frozenset(order)
    if projected:
        log(
            pdf,
            INFO,
            "Plotting",
            f"RooAbsReal::plotOn({pdf.GetName()}) plot on "
            f"{plot_var} integrates over variables ({','.join(order)})",
        )
    announce_average(pdf, frame, seen)
    announce(pdf, nset, normalising=True)
    if projected:
        _announce_projection(pdf, projected, nset)


def _announce_projection(pdf: Any, projected: frozenset[str], nset: frozenset[str]) -> None:
    """The projection integral's line: the density's own, or ``Int[y]_Norm[x,y]``."""
    from ..integration import announce, integral_name

    special = getattr(pdf, "announce_projection", None)
    if special is None or not special(projected, nset):
        norm = ",".join(one.GetName() for one in pdf.leaves() if one.GetName() in nset)
        label = f"{integral_name(pdf, projected, None)}_Norm[{norm}]"
        announce(pdf, projected, label=label, normalising=True)


def plot_function(func: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``RooAbsReal::plotOn``: a function, drawn as it is - scaled only if asked."""
    from .cmdlist import CmdList
    from .realplot import real_plot

    return real_plot(func, frame, CmdList.of(args, kwargs))


def _add_curve(
    func: Any,
    frame: Any,
    options: Commands,
    nset: frozenset[str],
    scale: float,
    chosen: set[str] | None,
    suffix: str,
    piece: Any = None,
    seen: Any = None,
) -> Any:
    """Sample the projection over the frame's variable and put the curve on the frame."""
    var = frame.getPlotVar()
    low, high, wings = piece
    projected = frozenset(seen.projected) if seen is not None else frozenset(nset - {var.GetName()})
    name = var.GetName()

    def projection(ctx: Any) -> Any:
        if projected:
            return func.fraction(projected, ctx, nset, None)
        return func.value(ctx, nset)

    def curve_at(xs: Any) -> Any:
        ctx = {name: np.asarray(xs, dtype=np.float64)}
        if not nset:
            return func.compute(ctx) * np.ones(len(xs)) * scale
        if seen is not None and seen.averaged:
            return seen.average(projection, ctx) * scale
        return projection(ctx) * scale

    precision = float(options.get("Precision", 0, 1e-3))
    with selection.selecting(chosen):
        xs, ys = sample(curve_at, low, high, frame.GetNbinsX(), precision, wings)
    norm = ",".join(sorted(nset, key=lambda n: [v.GetName() for v in func.leaves()].index(n)))
    label = f"{func.GetName()}_Norm[{norm}]" if nset else func.GetName()
    if seen is not None and seen.averaged:
        label += f"_DataAvg[{','.join(seen.averaged)}]"
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
