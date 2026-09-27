"""``VisualizeError``: the band a fit's uncertainty makes round a curve (``plotOnWithErrorBand``).

The curve is drawn again with each parameter moved up and down by ``Z``
of its errors, and at each point of the central curve the band's half-width
is ``sqrt(F C F)``, ``F`` the half-differences of the moved curves there and
``C`` the fit's correlation matrix - RooFit's linear propagation. Every
curve is drawn through the whole ``plotOn`` - saying what it says - and
taken back off the frame; only the band stays, filled cyan.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..cmdargs import RooCmdArg
from ..messages import WARNING, log
from ..printing import g
from .cmdlist import CmdList
from .curve import RooCurve
from .curves import style

__all__ = ["band"]

#: ``kCyan``: the band's fill and line.
CYAN = 432


def _plot_again(target: Any, frame: Any, arguments: CmdList) -> Any:
    """``plotFunc``: the whole ``plotOn`` again, fill colour aside, and the curve it added."""
    from .pdfplot import pdf_plot
    from .realplot import real_plot

    copy = arguments.copy()
    copy.strip("FillColor")
    if hasattr(target, "canBeExtended"):
        pdf_plot(target, frame, copy)
    else:
        real_plot(target, frame, copy)
    curve = frame.findObject(None, RooCurve)
    frame.remove(None)
    return curve


def _arguments(cmds: CmdList) -> CmdList:
    """The options the curves are drawn with: no ``VisualizeError``, no scale ``plotOn`` made itself."""
    kept = CmdList(one for one in cmds.items if one.name not in ("VisualizeError", "MoveToBack"))
    kept.items = [one for one in kept.items
                  if not (one.name == "Normalization" and one.value(2, 0))]  # fmt: skip
    return kept


def _moved(func: Any, frame: Any, arguments: CmdList, par: Any, value: float) -> Any:
    before = par.getVal()
    par.setVal(value)
    try:
        return _plot_again(func, frame, arguments)
    finally:
        par.setVal(before)


def _variations(func: Any, frame: Any, arguments: CmdList, fit: Any, z: float) -> tuple[list[Any], Any]:
    """The curves with each parameter up and down by ``z`` errors, and the correlations."""
    finals = [p for p in fit.floatParsFinal() if p.getError() > p.getVal() * np.finfo(float).eps]
    mine = {p.GetName(): p for p in func.getParameters(frame.norm_vars or [])}
    chosen = [p for p in finals if p.GetName() in mine]
    names = fit.floatParsFinal().names()
    index = [names.index(p.GetName()) for p in chosen]
    cov = np.array(fit.covarianceMatrix().values)[np.ix_(index, index)]
    pairs = []
    for i, final in enumerate(chosen):
        par, centre, error = mine[final.GetName()], final.getVal(), math.sqrt(cov[i, i])
        _warn_outside(func, par, centre, error, z)
        up = _moved(func, frame, arguments, par, min(centre + z * error, par.getMax()))
        down = _moved(func, frame, arguments, par, max(centre - z * error, par.getMin()))
        pairs.append((up, down))
    sigma = np.sqrt(np.diag(cov))
    return pairs, cov / np.outer(sigma, sigma)


def _warn_outside(func: Any, par: Any, centre: float, error: float, z: float) -> None:
    low, high = centre - z * error, centre + z * error
    if par.inRange(high) and par.inRange(low):
        return
    log(func, WARNING, "Plotting", f"RooAbsReal::plotOn({func.GetName()}): the {g(z)}-sigma error band for "
        f"the parameter \"{par.GetName()}\" is invalid because the variations ({g(low)}, {g(high)}) are "
        f"outside the defined range [{g(par.getMin())}, {g(par.getMax())}]!\n                         The "
        "variations will be clipped inside the range. This might or might not be acceptable in your "
        "usecase.")  # fmt: skip


def _band_curve(centre: Any, pairs: list[Any], corr: Any) -> RooCurve:
    """``makeErrorBand``: along the central curve and back, ``sqrt(F C F)`` either side."""
    f = np.array([(up.interpolate(centre.x) - down.interpolate(centre.x)) / 2 for up, down in pairs])
    half = np.sqrt(np.einsum("in,ij,jn->n", f, corr, f)) if len(pairs) else np.zeros(len(centre.x))
    xs = np.concatenate([centre.x, centre.x[::-1]])
    ys = np.concatenate([centre.y + half, (centre.y - half)[::-1]])
    made = RooCurve(f"{centre.GetName()}_errorband", "", xs, ys)
    made._core["TAttLine"]["fLineWidth"] = 1
    made._core["TAttLine"]["fLineColor"] = CYAN
    made._core["TAttFill"]["fFillColor"] = CYAN
    return made


def band(func: Any, frame: Any, cmds: CmdList, options: Any) -> Any:
    fit = options.get("VisualizeError")
    z = float(options.get("VisualizeError", 1, 1.0))
    arguments = _arguments(cmds)
    centre = _plot_again(func, frame, arguments)
    pairs, corr = _variations(func, frame, arguments, fit, z)
    made = _band_curve(centre, pairs, corr)
    final = cmds.process(f"RooAbsPdf::plotOn({func.GetName()})")
    style(made, final)
    if "Name" in final:
        made.SetName(str(final.get("Name")))
    elif final.get("CurveNameSuffix"):
        made.SetName(made.GetName() + str(final.get("CurveNameSuffix")))
    frame.add_plotable(made, str(final.get("DrawOption", 0, "F")), bool(final.get("Invisible", 0, False)))
    if "MoveToBack" in final:
        frame.items.insert(0, frame.items.pop())
    return frame


def moved_command(name: str, *args: Any) -> RooCmdArg:
    return RooCmdArg(name, *args)
