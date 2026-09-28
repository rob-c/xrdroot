"""``createChi2`` and ``chi2FitTo``: a density's chi-square against binned data, as RooFit sums it.

For each bin the prediction is the density at the bin's centre times the bin's volume and
the number of events - the data's, or the expected number of an extended fit - and the
term is ``(mu - n)^2 / sigma^2``, ``sigma^2`` the prediction itself by default (``Expected``),
the sum of squared weights for weighted data, or the Poisson interval's side facing the
prediction (``RooNLLVarNew::doEvalChi2``). Empty bins predicted empty are passed over.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..cmdargs import commands
from ..collections import RooArgSet
from ..messages import INFO, log
from ..real import Context, RooAbsReal
from .kahan import Kahan

__all__ = ["RooChi2Var", "chi2_fit_to", "create_chi2"]

#: ``RooAbsData::ErrorType``: Poisson, SumW2, None, Auto, Expected.
POISSON, SUMW2, NONE, AUTO, EXPECTED = 0, 1, 2, 3, 4


class RooChi2Var(RooAbsReal):
    """``chi2_<pdf>_<data>``: the chi-square of a density against a binned dataset."""

    def __init__(self, pdf: Any, data: Any, etype: int, extended: bool) -> None:
        super().__init__(
            f"chi2_{pdf.GetName()}_{data.GetName()}", f"chi2_{pdf.GetName()}_{data.GetName()}"
        )
        self.pdf = self._proxy("function", pdf)
        self.data = data
        self.etype = etype
        self.extended = extended
        self.nset = frozenset(one.GetName() for one in pdf.getObservables(data))
        self.offset = False

    def _parameters(self, observables: Any) -> RooArgSet:
        return RooArgSet(self.pdf.getParameters(self.data))

    def defaultErrorLevel(self) -> float:
        return 1.0

    def compute(self, ctx: Context) -> Any:
        return self.evaluate()

    def evaluate(self) -> float:
        columns = {k: np.asarray(v) for k, v in self.data.columns().items()}
        weights = np.asarray(self.data.weights(), dtype=np.float64)
        preds = np.broadcast_to(
            np.asarray(self.pdf.value(columns, self.nset), dtype=np.float64), weights.shape
        )
        norm = self.pdf.expected(self.nset) if self.extended else float(np.sum(weights))
        mus = preds * norm * np.asarray(self.data.binVolumes(), dtype=np.float64)
        total = Kahan()
        for i, (n, mu) in enumerate(zip(weights.tolist(), mus.tolist())):
            sigma2 = self._sigma2(i, n, mu)
            if sigma2 == 0.0 and n == 0.0 and mu == 0.0:
                continue
            total.add((mu - n) ** 2 / sigma2 if sigma2 > 0.0 else math.nan)
        return float(total.total)

    def _sigma2(self, i: int, n: float, mu: float) -> float:
        """The bin's variance: the prediction, the data's sum of squares, or its Poisson side."""
        if self.etype == SUMW2:
            return float(self.data.weights_squared()[i])
        if self.etype == POISSON:
            from ..plot.hist import _poisson_bar

            low, high = _poisson_bar(self.data.GetName(), n)
            return (high if mu > n else low) ** 2
        return mu

    def getVal(self, nset: Any = None) -> float:
        return self.evaluate()


def create_chi2(pdf: Any, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> RooChi2Var:
    """``createChi2(data, DataError(...), Extended(...))``, said as RooFit says it."""
    from .fit import _SAID_LIBRARY

    options = commands(args, kwargs)
    etype = int(options.get("DataError", 0, AUTO))
    if etype == AUTO:
        etype = SUMW2 if data.isNonPoissonWeighted() else EXPECTED
    extended = bool(options.get("Extended", 0, False)) and pdf.canBeExtended()
    log(pdf, INFO, "Fitting", f"createChi2({pdf.GetName()}) fixing normalization set for "
        "coefficient determination to observables in data", flush=False)  # fmt: skip
    if not _SAID_LIBRARY[0]:
        _SAID_LIBRARY[0] = True
        log(pdf, INFO, "Fitting", "using generic CPU library compiled with no vectorizations")
    return RooChi2Var(pdf, data, etype, extended)


def chi2_fit_to(pdf: Any, data: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """``chi2FitTo(data, options...)``: MIGRAD and HESSE on the chi-square, a result if saved."""
    from .fit import _configure
    from .minimizer import RooMinimizer

    options = commands(args, kwargs)
    chi2 = create_chi2(pdf, data, args, kwargs)
    minimizer = RooMinimizer(chi2)
    _configure(minimizer, options)
    minimizer.minimize(options.get("Minimizer", 0, ""), options.get("Minimizer", 1, ""))
    if options.get("Hesse", 0, True):
        minimizer.hesse()
    if not options.get("Save", 0, False):
        return None
    return minimizer.save(
        f"fitresult_{pdf.GetName()}_{data.GetName()}",
        f"Result of fit of p.d.f. {pdf.GetName()} to dataset {data.GetName()}",
    )
