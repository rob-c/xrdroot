"""``TFractionFitter``: the fractions of Monte Carlo templates a data histogram is made of.

A data histogram is fitted as a sum of templates, each weighted by the fraction of
the data it accounts for, with each template's own bins varied within their
statistics as Barlow and Beeston's likelihood has it (see
:mod:`xrdroot.fit.fraction`, which is the fit itself). The histograms are not
copied: the fit reads them where they are, and ``GetPlot`` and ``GetMCPrediction``
hand back what it worked out.

The whole histogram is fitted by default, its under- and overflows left out;
``SetRangeX`` and the rest narrow that, ``ExcludeBin`` leaves single bins out, and
``Constrain`` bounds a fraction between two values. One part of ROOT's interface is
not modelled - ``GetFitter`` hands out the ``ROOT::Fit::Fitter`` object Minuit is
driven through, which xrdroot has no equivalent of - and neither is
``ErrorAnalysis``, which runs MINOS.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ...fit.fraction import FractionFit
from .objects import TObject
from .refs import store

__all__ = ["TFractionFitter"]


class TFractionFitter(TObject):
    """``TFractionFitter``: fits the fractions of a set of templates to a data histogram."""

    def __init__(self, data: Any = None, mcs: Any = None, option: str = "") -> None:
        """Fit ``data`` as a sum of the templates in the ``TObjArray`` (or list) ``mcs``.

        ``option`` is ROOT's print level for the minimisation, which xrdroot does not
        print: the fit is quiet, and the status it returns is the report.
        """
        del option  # ROOT's print level for Minuit's report; nothing here prints one
        self._data = data
        self._mcs: list[Any] = []
        self._weights: list[Any] = []
        self._low = [1, 1, 1]
        self._high = [0, 0, 0]
        self._excluded: list[int] = []
        self._bounds: list[Any] | None = None
        self._dimension = 0
        self._fit: FractionFit | None = None
        self._result: Any = None
        self._plot: Any = None
        self._aji: np.ndarray[Any, Any] | None = None
        if data is None and mcs is None:
            return
        for par in range(mcs.GetEntries() if hasattr(mcs, "GetEntries") else len(mcs)):
            self._mcs.append(mcs.At(par) if hasattr(mcs, "At") else mcs[par])
            self._weights.append(None)
        self._limits()
        self._check()

    def _limits(self) -> None:
        """The bin ranges of the fit: the whole histogram, one axis at a time as it has them."""
        self._dimension = 0 if self._data is None else _dimension(self._data)
        self._low = [1, 1, 1]
        self._high = [1, 1, 1]
        for axis, count in enumerate(_counts(self._data) if self._data is not None else ()):
            self._high[axis] = count

    # -- ranges, exclusions and constraints ------------------------------------------------

    def SetRangeX(self, low: int, high: int) -> None:
        """Keep the bins from ``low`` to ``high`` along x, both counted from 1."""
        self._range(0, low, high)

    def ReleaseRangeX(self) -> None:
        """Fit the whole histogram along x again."""
        self._range(0, 0, 0)

    def SetRangeY(self, low: int, high: int) -> None:
        """Keep the bins from ``low`` to ``high`` along y, both counted from 1."""
        self._range(1, low, high)

    def ReleaseRangeY(self) -> None:
        """Fit the whole histogram along y again."""
        self._range(1, 0, 0)

    def SetRangeZ(self, low: int, high: int) -> None:
        """Keep the bins from ``low`` to ``high`` along z, both counted from 1."""
        self._range(2, low, high)

    def ReleaseRangeZ(self) -> None:
        """Fit the whole histogram along z again."""
        self._range(2, 0, 0)

    def _range(self, axis: int, low: int, high: int) -> None:
        """ROOT's ``SetRangeX``: a bound of zero or less means that axis's own end."""
        if axis >= self._dimension:
            raise ValueError(
                f"this histogram has {self._dimension} axes, so it has no '{'xyz'[axis]}' axis to "
                "restrict the fit to: restrict the axes it has"
            )
        bins = _counts(self._data)[axis]
        self._low[axis] = max(int(low), 1)
        self._high[axis] = int(high) if 0 < int(high) <= bins else bins
        self._check()

    def ExcludeBin(self, bin: int) -> None:
        """Leave the global bin ``bin`` out of the fit, numbered as ``TH1::GetBin`` numbers it."""
        if int(bin) in self._excluded:
            raise ValueError(
                f"bin {int(bin)} is already excluded from the fit, so there is nothing to exclude"
            )
        self._excluded.append(int(bin))
        self._check()

    def IncludeBin(self, bin: int) -> None:
        """Fit the global bin ``bin`` again, if ``ExcludeBin`` left it out."""
        if int(bin) not in self._excluded:
            raise ValueError(
                f"bin {int(bin)} was never excluded from the fit, so there is nothing to include"
            )
        self._excluded.remove(int(bin))
        self._check()

    def Constrain(self, parm: int, low: float, high: float) -> None:
        """Keep fraction ``parm`` between ``low`` and ``high``, both ends included."""
        if self._bounds is None:
            self._bounds = [None] * len(self._mcs)
        self._bounds[self._checked_parm(parm)] = (float(low), float(high))

    def UnConstrain(self, parm: int) -> None:
        """Let fraction ``parm`` take any value again."""
        if self._bounds is None:
            self._bounds = [None] * len(self._mcs)
        self._bounds[self._checked_parm(parm)] = None

    def _checked_parm(self, parm: int) -> int:
        """``CheckParNo``: the template number, refused in a sentence when it names no template."""
        at = int(parm)
        if not 0 <= at < len(self._mcs):
            raise IndexError(
                f"fraction {at} does not exist: this fit has {len(self._mcs)} templates, numbered "
                f"from 0 to {len(self._mcs) - 1}"
            )
        return at

    # -- changing what is fitted -----------------------------------------------------------

    def SetData(self, data: Any) -> None:
        """Fit ``data`` instead; the ranges, exclusions and bounds are kept."""
        self._data = data
        self._dimension = 0 if data is None else _dimension(data)
        self._check()

    def SetMC(self, parm: int, mc: Any) -> None:
        """Use ``mc`` as template ``parm``; the bounds on that fraction are kept."""
        self._mcs[self._checked_parm(parm)] = mc
        self._check()

    def SetWeight(self, parm: int, weight: Any) -> None:
        """Weight template ``parm``'s bins by ``weight``'s contents, or not at all for ``None``."""
        self._weights[self._checked_parm(parm)] = weight
        self._check()

    # -- the fit ---------------------------------------------------------------------------

    def _check(self) -> None:
        """``CheckConsistency``: data and templates must describe the same bins, and hold events."""
        self._check_data()
        for par, mc in enumerate(self._mcs):
            self._check_template(par, mc)

    def _check_data(self) -> None:
        """The data exists, two templates are given, and its fitted bins hold events."""
        if self._data is None:
            raise ValueError(
                "TFractionFitter fits a data histogram, and none was given: pass one in"
            )
        if len(self._mcs) < 2:
            raise ValueError(
                f"TFractionFitter fits two templates' fractions at least, but {len(self._mcs)} was "
                "given: with one, its fraction is the normalisation, which the fit cannot tell"
            )
        if not self._fitted().size or not self._contents(self._data).any():
            raise ValueError(
                "the data holds no events in the bins being fitted: there is nothing to fit"
            )

    def _check_template(self, par: int, mc: Any) -> None:
        """Template ``par`` describes the data's bins, holds events in the fitted ones, and has
        weights - if it has them - over the same bins."""
        if mc is None:
            raise ValueError(
                f"template {par} does not exist: TFractionFitter was given no histogram for it"
            )
        if _dimension(mc) != self._dimension or _counts(mc) != _counts(self._data):
            raise ValueError(
                f"template {par} has {_bins(mc)}, and the data has {_bins(self._data)}: the "
                "templates must have the same bins as the data"
            )
        weight = self._weights[par]
        if weight is not None and _counts(weight) != _counts(self._data):
            raise ValueError(
                f"the weights histogram of template {par} has {_bins(weight)}, and the data "
                f"has {_bins(self._data)}: they must have the same bins"
            )
        if not self._contents(mc)[self._fitted()].any():
            raise ValueError(
                f"template {par} holds no events in the bins being fitted, so its fraction is "
                "not measurable: exclude those bins or fit another template"
            )

    def _fitted(self) -> np.ndarray[Any, Any]:
        """The global bins the fit uses, in the histogram's own numbering."""
        spans = [range(self._low[axis], self._high[axis] + 1) for axis in range(self._dimension)]
        bins = [
            self._data.GetBin(*_place(coordinates, self._dimension))
            for coordinates in product(*spans)
        ]
        return np.array([bin for bin in bins if bin not in self._excluded], dtype=np.int64)

    def _contents(self, histogram: Any) -> np.ndarray[Any, Any]:
        """Every bin of ``histogram``, flow bins included, in the order the global bins count."""
        return np.asarray(histogram._contents(), dtype=np.float64)

    def _built(self) -> FractionFit:
        """The engine for the histograms as they stand: fitted bins' contents, a row each."""
        fitted = self._fitted()
        rows = [self._contents(mc)[fitted] for mc in self._mcs]
        weights = None if all(w is None for w in self._weights) else np.array([
            np.ones(fitted.size) if w is None else self._contents(w)[fitted] for w in self._weights
        ])  # fmt: skip
        return FractionFit(self._contents(self._data)[fitted], np.array(rows), weights)

    def Fit(self) -> int:
        """Perform the fit, and answer with the minimisation's status: 0 when it found a minimum."""
        self._plot, self._aji = None, None
        self._fit = self._built()
        self._result = self._fit.solve(limits=self._bounds)
        return int(self._result.status)

    def GetResult(self, parm: int, value: Any = None, error: Any = None) -> tuple[float, float]:
        """Fraction ``parm``'s fitted value and error, into ``value`` and ``error`` if given."""
        found, _ = self._fitted_once(self._checked_parm(parm), "GetResult")
        value_value, error_value = float(found.parameters[parm]), float(found.errors[parm])
        store(value, value_value)
        store(error, error_value)
        return value_value, error_value

    def _fitted_once(self, parm: int, what: str) -> tuple[Any, FractionFit]:
        """The last fit's result and engine, refused in a sentence when there has been none."""
        if self._result is None or self._fit is None:
            raise ValueError(
                f"{what} asks for the outcome of a fit not yet performed: call Fit() first"
            )
        return self._result, self._fit

    def GetPlot(self) -> Any:
        """The histogram of the data's bins as the fit predicts them, named after the data's."""
        self._done(*self._fitted_once(0, "GetPlot"))
        return self._plot

    def GetMCPrediction(self, parm: int) -> Any:
        """Template ``parm`` as the fit smears it: its bins within their own statistics."""
        at = self._checked_parm(parm)
        smeared = self._done(*self._fitted_once(at, "GetMCPrediction"))
        return self._filled(self._mcs[at], smeared[at], f"Prediction for MC sample {at}")

    def _done(self, result: Any, engine: FractionFit) -> np.ndarray[Any, Any]:
        """The smeared templates of the last fit, and with them its plot, worked out once."""
        if self._aji is None:
            total, self._aji = engine.predictions(result.parameters)
            self._plot = self._filled(
                self._data, total, f"Fraction fit to hist: {self._data.GetName()}"
            )
        return self._aji

    def _filled(self, histogram: Any, contents: Any, name: str) -> Any:
        """A copy of ``histogram`` named ``name`` holding ``contents``, kept in no directory."""
        made = histogram.Clone(name)
        made.SetDirectory(None)
        made.Reset()
        for at, bin in enumerate(self._fitted()):
            made.SetBinContent(int(bin), float(contents[at]))
        return made

    def GetChisquare(self) -> float:
        """``ComputeChisquareLambda``: the likelihood-ratio chi-square of the fit."""
        found, engine = self._fitted_once(0, "GetChisquare")
        total, fitted = engine.predictions(found.parameters)
        return engine.chisquare(total, fitted)

    def GetNDF(self) -> int:
        """The fit's degrees of freedom: the bins fitted less the number of templates."""
        return int(self._fitted().size - len(self._mcs))

    def GetProb(self) -> float:
        """The probability that a chi-square with these degrees of freedom is this large."""
        from ...stats import prob  # only a probability needs it

        return float(prob(self.GetChisquare(), self.GetNDF()))

    def GetFitter(self) -> Any:
        """Refused: ROOT hands out the ``ROOT::Fit::Fitter`` Minuit is driven through, and xrdroot
        has no such object - a fit here is :func:`xrdroot.fit.minimize`, with its own result."""
        raise UnsupportedFeatureError(
            "TFractionFitter::GetFitter hands out the ROOT::Fit::Fitter object Minuit is driven "
            "through, and xrdroot has no such object: its fits are xrdroot.fit.minimize, whose "
            "result carries the same parameters, errors and covariance. Use GetResult and "
            "GetChisquare for the outcome of the fit."
        )

    def ErrorAnalysis(self, up: float = 0.0) -> Any:
        """Refused: ROOT runs MINOS over the fit, with ``up`` as its error definition."""
        del up
        raise UnsupportedFeatureError(
            "TFractionFitter::ErrorAnalysis runs MINOS over the fitted fractions, which xrdroot's "
            "TFractionFitter does not do yet: the errors GetResult answers with are Minuit's "
            "symmetric ones from the fit itself."
        )


def _dimension(histogram: Any) -> int:
    """How many axes ``histogram`` has."""
    return int(histogram.GetDimension())


def _counts(histogram: Any) -> tuple[int, ...]:
    """How many bins ``histogram`` has along each axis it has: one number per axis."""
    return (histogram.GetNbinsX(), histogram.GetNbinsY(), histogram.GetNbinsZ())[
        : _dimension(histogram)
    ]


def _bins(histogram: Any) -> str:
    """A histogram's bins as a sentence can carry them: ``40 bins in x``."""
    return " and ".join(
        f"{count} bins in {'xyz'[at]}" for at, count in enumerate(_counts(histogram))
    )


def _place(coordinates: Any, dimension: int) -> tuple[int, ...]:
    """A bin's number along each axis, padded to three as ``TH1::GetBin`` takes them."""
    return tuple(coordinates) + (0,) * (3 - dimension)
