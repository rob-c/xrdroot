"""``TFitResult``: what a fit found, and how sure it is of it.

A fit ends in a handful of numbers - the parameters, their errors and
covariance, the minimum, how many degrees of freedom it had - and ROOT keeps
them in a ``ROOT::Fit::FitResult``, which ``TH1::Fit`` hands back with
option ``"S"``. :class:`FitResult` is that object: the same members under
Python names, ``Parameter(i)`` and ``Error(i)`` as :meth:`parameter` and
:meth:`error`, and :meth:`summary` the lines ``FitResult::Print`` prints,
in its words and its column widths.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..stats import incomplete_gamma_c

__all__ = ["FitResult"]

#: ``FitResult::Print``'s column widths: the names, and the numbers.
NAME_WIDTH = 25
NUMBER_WIDTH = 12
#: The line of stars ``FitResult::Print`` starts with.
STARS = "*" * 40


def _g(value: float) -> str:
    """A double as ``std::ostream`` prints one by default: six significant figures."""
    return format(float(value), "g")


class FitResult:
    """The outcome of a fit, as ROOT's ``TFitResult`` holds it.

        >>> r = h.fit("gaus", "S")                                   # doctest: +SKIP
        >>> r.parameter(1), r.error("Sigma"), r.chi2 / r.ndf         # doctest: +SKIP
        >>> print(r.summary())                                       # doctest: +SKIP

    ``status`` is Minuit2's: 0 for a good minimum, 1 when the covariance had
    to be made positive definite, 2 when Hesse failed, 3 when the distance
    to the minimum stayed above what was asked for, 4 at the call limit;
    ``valid`` is whether Minuit believes the minimum. ``int(result)`` is the
    status, as a ``TFitResultPtr`` converts to one.
    """

    __slots__ = (
        "parameters",
        "errors",
        "covariance",
        "parameter_names",
        "chi2",
        "ndf",
        "fcn",
        "edm",
        "nfev",
        "status",
        "valid",
        "minos",
        "fixed",
        "bounded",
        "minimizer",
        "function",
        "npoints",
    )

    def __init__(
        self,
        *,
        parameters: Any,
        errors: Any,
        covariance: Any,
        names: Sequence[str],
        fcn: float,
        chi2: float | None = None,
        ndf: int = 0,
        edm: float = 0.0,
        nfev: int = 0,
        status: int = 0,
        valid: bool = True,
        minos: dict[str, tuple[float, float]] | None = None,
        fixed: Sequence[bool] | None = None,
        bounded: Sequence[bool] | None = None,
        minimizer: str = "Minuit2 / Migrad",
        npoints: int = 0,
    ) -> None:
        npar = len(names)
        #: The values at the minimum, one per parameter.
        self.parameters = np.asarray(parameters, dtype=np.float64).copy()
        #: The parabolic errors: the root of the covariance's diagonal, zero if fixed.
        self.errors = np.asarray(errors, dtype=np.float64).copy()
        #: The covariance of every parameter with every other, zero for a fixed one.
        self.covariance = np.asarray(covariance, dtype=np.float64).reshape(npar, npar).copy()
        #: Each parameter's name.
        self.parameter_names = tuple(str(name) for name in names)
        #: ``MinFcnValue``: the least value of what was minimised.
        self.fcn = float(fcn)
        #: ``Chi2``: the chi-square - for a binned likelihood, Baker and Cousins's.
        self.chi2 = float(fcn if chi2 is None else chi2)
        #: ``Ndf``: the points fitted less the parameters left free.
        self.ndf = int(ndf)
        #: ``Edm``: Minuit's estimate of the distance still to the minimum.
        self.edm = float(edm)
        #: ``NCalls``: how many times the function was evaluated.
        self.nfev = int(nfev)
        #: ``Status``: Minuit2's, 0 when all went well.
        self.status = int(status)
        #: ``IsValid``: whether the minimum is one.
        self.valid = bool(valid)
        #: ``LowerError`` and ``UpperError``: Minos's, by parameter name.
        self.minos = dict(minos or {})
        #: ``IsParameterFixed``, per parameter.
        self.fixed = tuple(bool(flag) for flag in (fixed or [False] * npar))
        #: ``IsParameterBound``, per parameter.
        self.bounded = tuple(bool(flag) for flag in (bounded or [False] * npar))
        #: ``MinimizerType``: ``"Minuit2 / Migrad"``, or ``"Linear"`` for linear least squares.
        self.minimizer = minimizer
        #: The :class:`~xrdroot.Function` fitted, as it was left - or ``None``.
        self.function: Any = None
        #: How many points were fitted: ``fNpfits`` of the function.
        self.npoints = int(npoints)

    # -- ROOT's accessors -------------------------------------------------------

    def _index(self, parameter: int | str) -> int:
        if isinstance(parameter, str):
            if parameter not in self.parameter_names:
                raise KeyError(
                    f"the fit has no parameter called {parameter!r}; it has "
                    f"{', '.join(self.parameter_names)}"
                )
            return self.parameter_names.index(parameter)
        if not 0 <= parameter < len(self.parameter_names):
            raise IndexError(
                f"the fit has no parameter {parameter}: it has {len(self.parameter_names)}"
            )
        return int(parameter)

    def parameter(self, parameter: int | str) -> float:
        """``Parameter(i)``: one parameter's value, by index or by name."""
        return float(self.parameters[self._index(parameter)])

    def error(self, parameter: int | str) -> float:
        """``ParError(i)``: one parameter's parabolic error."""
        return float(self.errors[self._index(parameter)])

    def lower_error(self, parameter: int | str) -> float:
        """``LowerError(i)``: Minos's error below, or the parabolic one without Minos."""
        index = self._index(parameter)
        found = self.minos.get(self.parameter_names[index])
        return float(found[0]) if found else float(self.errors[index])

    def upper_error(self, parameter: int | str) -> float:
        """``UpperError(i)``: Minos's error above, or the parabolic one without Minos."""
        index = self._index(parameter)
        found = self.minos.get(self.parameter_names[index])
        return float(found[1]) if found else float(self.errors[index])

    @property
    def correlation(self) -> np.ndarray[Any, Any]:
        """``GetCorrelationMatrix``: the covariance over the product of the two errors."""
        scale = np.sqrt(np.abs(np.diag(self.covariance)))
        outer = np.outer(scale, scale)
        return np.divide(
            self.covariance, outer, out=np.zeros_like(self.covariance), where=outer > 0
        )

    @property
    def prob(self) -> float:
        """``Prob``: the chance of a chi-square this large or larger with ``ndf`` degrees."""
        if self.ndf <= 0:
            return 1.0 if self.chi2 <= 0 else 0.0
        return incomplete_gamma_c(0.5 * self.ndf, 0.5 * self.chi2)

    def __int__(self) -> int:
        return self.status

    # -- printing -------------------------------------------------------------------

    def _line(self, label: str, value: Any) -> str:
        return f"{label:<{NAME_WIDTH}} = {value:>{NUMBER_WIDTH}}"

    def _parameter_line(self, index: int) -> str:
        name = self.parameter_names[index]
        text = f"{name:<{NAME_WIDTH}} = {_g(self.parameters[index]):>{NUMBER_WIDTH}}"
        if self.fixed[index]:
            return text + " " * 9 + " " * NUMBER_WIDTH + " \t (fixed)"
        text += f"   +/-   {_g(self.errors[index]):<{NUMBER_WIDTH}}"
        if name in self.minos:
            low, high = self.minos[name]
            text += f"  {_g(low):<{NUMBER_WIDTH}} +{_g(high):<{NUMBER_WIDTH}} (Minos) "
        if self.bounded[index]:
            text += " \t (limited)"
        return text

    def summary(self, covariance: bool = False) -> str:
        """``Print``: the lines ROOT prints after a fit, and with ``covariance`` the matrices."""
        lines = [STARS]
        if not self.valid:
            lines += [f"         Invalid FitResult  (status = {self.status} )", STARS]
        lines.append(f"Minimizer is {self.minimizer}")
        if self.fcn != self.chi2 or self.chi2 < 0:
            lines.append(self._line("MinFCN", _g(self.fcn)))
        lines.append(self._line("Chi2", _g(self.chi2)))
        lines.append(self._line("NDf", self.ndf))
        if "Linear" not in self.minimizer:
            lines.append(self._line("Edm", _g(self.edm)))
            lines.append(self._line("NCalls", self.nfev))
        lines += [self._parameter_line(index) for index in range(len(self.parameter_names))]
        text = "\n".join(lines)
        return text + self._matrices() if covariance else text

    def _matrices(self) -> str:
        """``PrintCovMatrix``: the covariance and the correlation, of the free parameters."""
        free = [i for i, flag in enumerate(self.fixed) if not flag]
        names = [self.parameter_names[i] for i in free]
        out = []
        for title, matrix in (("Covariance", self.covariance), ("Correlation", self.correlation)):
            out.append(f"\n\n{title} Matrix:\n")
            out.append(" " * 12 + "\t" + "".join(f"{name:>12}" for name in names))
            for i, name in zip(free, names):
                row = "".join(f"{format(matrix[i, j], '.5g'):>12}" for j in free)
                out.append(f"{name:<12}\t{row}")
        return "\n".join(out)

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{name}={value:.6g}±{error:.2g}"
            for name, value, error in zip(self.parameter_names, self.parameters, self.errors)
        )
        verdict = "valid" if self.valid else f"invalid, status {self.status}"
        return f"<FitResult chi2/ndf={self.chi2:.6g}/{self.ndf} ({verdict}): {pairs}>"
