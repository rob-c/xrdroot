"""``plotOn(frame, Asymmetry(cat))``: a density's asymmetry in a category of two signs, as a curve.

``RooAbsReal::plotAsymOn`` draws ``(P+ - P-) / (P+ + P-)``, where ``P+``
and ``P-`` are the density's projections with the category set to ``+1``
and to ``-1`` - each integrated over the frame's other variables and
normalised over all of them - unscaled, and red unless told otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cmdargs import Commands
from ..messages import ERROR, INFO, log
from .curve import RooCurve, sample
from .curves import style
from .projections import view

__all__ = ["plot_asymmetry"]

#: ``kRed``: the colour ``plotAsymOn`` gives its curve.
RED = 2


def _is_sign_type(category: Any) -> bool:
    """``isSignType``: two or three states, indexed ``-1``, ``0`` and ``1``."""
    indices = set(category.states().values())
    return 2 <= len(indices) <= 3 and indices <= {-1, 0, 1}


def plot_asymmetry(pdf: Any, frame: Any, options: Commands) -> Any:
    """The asymmetry curve of ``pdf`` in the category ``options`` names, on ``frame``."""
    category = options.get("Asymmetry")
    name, var = category.GetName(), frame.getPlotVar()
    if name not in pdf.dependents():
        log(pdf, ERROR, "Plotting", f"RooAbsReal::plotAsymOn({pdf.GetName()}) function doesn't depend on "
            f"asymmetry category {name}")  # fmt: skip
        return frame
    if not _is_sign_type(category):
        log(pdf, ERROR, "Plotting", f"RooAbsReal::plotAsymOn({pdf.GetName()}) asymmetry category must have 2 "
            "or 3 states with index values -1,0,1")  # fmt: skip
        return frame
    seen = view(pdf, frame, options, "plotAsymOn")
    projected = [one for one in seen.projected if one != name]
    if projected:
        log(pdf, INFO, "Plotting", f"RooAbsReal::plotAsymOn({pdf.GetName()}) plot on {var.GetName()} "
            f"projects variables ({','.join(projected)})")  # fmt: skip
    nset = frozenset([var.GetName(), name, *projected, *seen.sliced])
    scale = float(options.get("Normalization", 0, 1.0))

    def projection(ctx: Any) -> Any:
        return (
            pdf.fraction(frozenset(projected), ctx, nset, None)
            if projected
            else pdf.value(ctx, nset)
        )

    def asymmetry(xs: Any) -> Any:
        ctx = {var.GetName(): np.asarray(xs, dtype=np.float64)}
        plus, minus = projection({**ctx, name: 1.0}), projection({**ctx, name: -1.0})
        return np.broadcast_to((plus - minus) / (plus + minus), np.shape(xs)) * scale

    xs, ys = sample(asymmetry, frame.GetXmin(), frame.GetXmax(), frame.GetNbinsX(),
                    float(options.get("Precision", 0, 1e-3)), True)  # fmt: skip
    curve = RooCurve(
        f"{pdf.GetName()}_Asym[{name}]", f"{name} Asymmetry of {pdf.GetTitle()}", xs, ys
    )
    curve._core["TAttLine"]["fLineColor"] = RED
    style(curve, options)
    frame.add_plotable(
        curve, str(options.get("DrawOption", 0, "L")), bool(options.get("Invisible", 0, False))
    )
    return frame
