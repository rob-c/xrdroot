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
from ..messages import INFO, WARNING, log
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


def _variations(func: Any, frame: Any, arguments: CmdList, fit: Any, z: float,
                wanted: Any) -> tuple[list[Any], Any]:  # fmt: skip
    """The curves with each parameter up and down by ``z`` errors, and the correlations."""
    finals = [p for p in fit.floatParsFinal() if p.getError() > p.getVal() * np.finfo(float).eps]
    mine = {p.GetName(): p for p in func.getParameters(frame.norm_vars or [])
            if wanted is None or p.GetName() in wanted}  # fmt: skip
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


def _sampled(func: Any, frame: Any, arguments: CmdList, fit: Any, z: float, centre: Any,
             wanted: Any) -> RooCurve:  # fmt: skip
    """The band from curves of parameters drawn from the fit's Gaussian: their central quantiles."""
    finals = set(fit.floatParsFinal().names())
    params = [p for p in func.getObservables(fit.floatParsFinal())
              if p.GetName() in finals and (wanted is None or p.GetName() in wanted)]  # fmt: skip
    density = fit.createHessePdf(params)
    n = max(int(100.0 / math.erfc(z / math.sqrt(2.0))), 100)
    log(func, INFO, "Plotting", f"RooAbsReal::plotOn({func.GetName()}) INFO: visualizing {g(z)}-sigma "
        f"uncertainties in parameters ({','.join(p.GetName() for p in params)}) from fit result "
        f"{fit.GetName()} using {n} samplings.")  # fmt: skip
    ymin, ymax = frame.GetMinimum(), frame.GetMaximum()
    drawn = density.generate(params, n)
    saved = [(p, p.getVal()) for p in params]
    curves = []
    for i in range(drawn.numEntries()):
        for par in params:
            par.setVal(float(drawn.column(par.GetName())[i]))
        curves.append(_plot_again(func, frame, arguments))
    for par, value in saved:
        par.setVal(value)
    frame.SetMinimum(ymin)
    frame.SetMaximum(ymax)
    delta = int(len(curves) * math.erfc(z / math.sqrt(2.0)) / 2 + 0.5)
    ys = np.sort(np.array([c.interpolate(centre.x) for c in curves]), axis=0)
    low, high = ys[delta], ys[len(curves) - delta]
    made = RooCurve(f"{centre.GetName()}_errorband", "", np.concatenate([centre.x, centre.x[::-1]]),
                    np.concatenate([low, high[::-1]]))  # fmt: skip
    made._core["TAttLine"]["fLineWidth"] = 1
    made._core["TAttLine"]["fLineColor"] = CYAN
    made._core["TAttFill"]["fFillColor"] = CYAN
    return made


def _visualize(given: tuple[Any, ...]) -> tuple[Any, Any, float, bool]:
    """``VisualizeError(fit, [params,] Z=1, linear=True)``: its fit, parameters, ``Z`` and method."""
    fit, rest = given[0], list(given[1:])
    wanted = None
    if rest and not isinstance(rest[0], (int, float)):
        wanted = {one.GetName() for one in _as_list(rest.pop(0))}
    z = float(rest[0]) if rest else 1.0
    linear = bool(rest[1]) if len(rest) > 1 else True
    return fit, wanted, z, linear


def _as_list(items: Any) -> list[Any]:
    from ..collections import as_list

    return as_list(items)


def band(func: Any, frame: Any, cmds: CmdList, options: Any) -> Any:
    fit, wanted, z, linear = _visualize(options.args("VisualizeError"))
    arguments = _arguments(cmds)
    centre = _plot_again(func, frame, arguments)
    if linear:
        pairs, corr = _variations(func, frame, arguments, fit, z, wanted)
        made = _band_curve(centre, pairs, corr)
    else:
        made = _sampled(func, frame, arguments, fit, z, centre, wanted)
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
