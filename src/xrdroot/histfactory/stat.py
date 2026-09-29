"""A channel's statistical uncertainty: its gammas, their errors and their constraints.

Each sample in the statistical error has an absolute uncertainty per bin -
its histogram's errors, or its error histogram times its contents - and
the channel's relative uncertainty is their quadrature sum over the sum of
the contents; each bin's gamma is then ranged and given that error, made
constant below the threshold, and constrained by a Gaussian about one or a
Poisson of ``1/sigma^2`` events, as ``createGammaConstraints`` makes them.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.messages import INFO, WARNING, log
from ..roofit.printing import g
from .model import _hf
from .systematics import Constraint, HistFactoryError

__all__ = ["absolute_uncertainty", "gamma_constraints", "hist_values", "relative_uncertainty"]


def _bins(hist: Any) -> list[int]:
    """The global numbers of the histogram's bins, under- and overflows skipped."""
    count = hist.GetNbinsX() * hist.GetNbinsY() * hist.GetNbinsZ()
    found: list[int] = []
    number = 0
    while len(found) < count:
        number += 1
        if not (hist.IsBinUnderflow(number) or hist.IsBinOverflow(number)):
            found.append(number)
    return found


def hist_values(hist: Any) -> list[float]:
    """``histToVector``: the contents of the bins, in order."""
    return [float(hist.GetBinContent(n)) for n in _bins(hist)]


def absolute_uncertainty(name: str, nominal: Any) -> Any:
    """``MakeAbsolUncertaintyHist``: a histogram of the nominal's bin errors."""
    made = nominal.Clone(name)
    made.Reset()
    for i, number in enumerate(_bins(nominal)):
        error = float(nominal.GetBinError(number))
        if error != error:
            raise HistFactoryError(f"HistFactory - bin {i} of {nominal.GetName()} has a NaN error")
        if error < 0:
            _hf(WARNING, f"Warning: In histogram {nominal.GetName()} bin error for bin {number} "
                "is < 0.  Setting Error to 0")  # fmt: skip
            error = 0.0
        made.SetBinContent(number, error)
    return made


def relative_uncertainty(name: str, pairs: list[tuple[Any, Any]]) -> Any:
    """``MakeScaledUncertaintyHist``: each bin's summed error over its summed content."""
    template = pairs[0][0]
    numbers = _bins(template)
    totals = [0.0] * len(numbers)
    squares = [0.0] * len(numbers)
    for i, number in enumerate(numbers):
        for nominal, error in pairs:
            totals[i] += float(nominal.GetBinContent(number))
            squares[i] += float(error.GetBinContent(number)) ** 2
    made = template.Clone(name)
    made.Reset()
    for i, number in enumerate(numbers):
        if totals[i] <= 0:
            _hf(WARNING, f"Warning: Sum of histograms for bin: {number} is <= 0.  Setting error "
                "to 0")  # fmt: skip
            made.SetBinContent(number, 0.0)
            continue
        relative = math.sqrt(squares[i]) / totals[i]
        made.SetBinError(number, totals[i])
        made.SetBinContent(number, relative)
        _hf(INFO, f"Making Total Uncertainty for bin {number} Error = {g(math.sqrt(squares[i]))}"
            f" CentralVal = {g(totals[i])} RelativeError = {g(relative)}")  # fmt: skip
    return made


def _configured(gammas: list[Any], sigmas: list[float], threshold: float) -> None:
    """``configureConstrainedGammas``: each gamma ranged to five sigma, given its error -
    constant, with none or below the threshold."""
    for gamma, sigma in zip(gammas, sigmas):
        if sigma <= 0:
            gamma.setConstant(True)
            continue
        gamma.setMax(1.0 + 5.0 * sigma)
        gamma.setMin(0.0)
        gamma.setError(sigma)
        if sigma < threshold:
            log(None, WARNING, "HistFactory", f'Warning: relative sigma {g(sigma)} for '
                f'"{gamma.GetName()}" falls below threshold of {g(threshold)}. Setting: '
                f'{gamma.GetName()} to constant')  # fmt: skip
            gamma.setConstant(True)


def gamma_constraints(gammas: list[Any], sigmas: list[float], threshold: float,
                      kind: int) -> tuple[list[Any], list[Any]]:  # fmt: skip
    """``createGammaConstraints``: each gamma's constraint and its global observable."""
    from ..roofit.functions import RooProduct
    from ..roofit.pdfs.basic import RooGaussian
    from ..roofit.pdfs.shapes import RooPoisson
    from ..roofit.variables import RooConstVar, RooRealVar

    if len(sigmas) != len(gammas):
        raise HistFactoryError(f"HistFactory - {len(sigmas)} relative sigmas were given for "
                               f"{len(gammas)} gammas")  # fmt: skip
    _configured(gammas, sigmas, threshold)
    terms, globs = [], []
    for i, (gamma, sigma) in enumerate(zip(gammas, sigmas)):
        name = gamma.GetName()
        _hf(INFO, f"Creating constraint for: {name}. Type of constraint: {kind}")
        if sigma <= 0:
            _hf(INFO, f"Not creating constraint term for {name} because sigma = {g(sigma)} "
                f"(sigma<=0) (bin number = {i})")  # fmt: skip
            continue
        if kind == Constraint.Gaussian:
            nominal = RooRealVar(f"nom_{name}", f"nom_{name}", 1.0, 0, 10)
            nominal.setConstant(True)
            width = RooConstVar(f"{name}_sigma", f"{name}_sigma", sigma)
            term: Any = RooGaussian(f"{name}_constraint", f"{name}_constraint", nominal, gamma,
                                    width)  # fmt: skip
        else:
            tau = 1.0 / (sigma * sigma)
            nominal = RooRealVar(f"nom_{name}", f"nom_{name}", tau)
            nominal.setMin(0)
            nominal.setConstant(True)
            scaling = RooConstVar(f"{name}_tau", f"{name}_tau", tau)
            mean = RooProduct(f"{name}_poisMean", f"{name}_poisMean", [gamma, scaling])
            term = RooPoisson(f"{name}_constraint", f"{name}_constraint", nominal, mean, True)
        terms.append(term)
        globs.append(nominal)
    return terms, globs
