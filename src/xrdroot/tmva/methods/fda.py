"""``FDA``: the function discriminant - a formula of the variables whose parameters are fitted.

The user writes the function, ``(0)+(1)*x0+...``, and the interval of each
parameter; the parameters are fitted to make the function 1 for signal and 0
for background (least squares, each class weighing the same), the targets
for a regression, or the classes by cross entropy for several. The fit is
TMVA's genetic algorithm (``FitMethod=GA``) or its Monte Carlo sampling
(``MC``), draw for draw; the simulated annealing and MINUIT fitters, and
MINUIT as a ``Converger``, xrdroot does not have and refuses by name.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

import numpy as np

from ..dataset import Events
from ..fdaformula import FormulaError, compile_fda, tformula_text
from ..genetic import GeneticFitter, Interval
from ..mcfitter import MCFitter
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..tools import CxxVector
from ..xmlfile import Node, children, number
from .linear import formatted_values

__all__ = ["MethodFDA"]


#: What ``stringstream >> float`` reads of the start of a text: the longest number there.
NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")


def _pieces(text: str) -> list[str]:
    """``Tools::ParseFormatLine(text, ";")``: the pieces between separators, empty ones skipped."""
    return [piece for piece in text.split(";") if piece]


def _number(text: str) -> float:
    """``stringstream >> Float_t``: the number the text starts with, in single precision, else 0."""
    found = NUMBER.match(text)
    return float(np.float32(found.group())) if found else 0.0


def _range(piece: str) -> tuple[float, float]:
    """One ``(a,b)`` of ``ParRanges``, as TMVA reads it."""
    low, _, high = piece[1:-1].partition(",")
    return _number(low), _number(high)


class MethodFDA(Method):
    """``TMVA::MethodFDA``."""

    type_name = "FDA"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults: ClassVar[dict[str, Any]] = {
        "Formula": "(0)",
        "ParRanges": "()",
        "FitMethod": "MINUIT",
        "Converger": "None",
    }

    def process_options(self) -> None:
        self._read_ranges()
        self.fit = str(self.opt("FitMethod"))
        for what, value, known in (
            ("FitMethod", self.fit, ("MC", "GA")),
            ("Converger", str(self.opt("Converger")), ("None",)),
        ):
            if value not in known:
                raise self.log.refuse(
                    f"{what}={value} is a fitter of TMVA's FDA xrdroot does not have; it has "
                    "the genetic algorithm (GA) and Monte Carlo sampling (MC), and no converger"
                )
        self.formula = str(self.opt("Formula"))
        self._compile()
        self.dims = 1
        if self.analysis == REGRESSION:
            self.dims = self.dsi.GetNTargets()
        elif self.analysis == MULTICLASS:
            self.dims = self.dsi.GetNClasses()
        self.parameters = [0.5 * (low + high) for low, high in self.intervals] * self.dims

    def _read_ranges(self) -> None:
        """``ParRanges``: a range per closing parenthesis, each's minimum below its maximum."""
        text = str(self.opt("ParRanges")).replace(" ", "")
        self.npars = text.count(")")
        pieces = _pieces(text)
        if len(pieces) != self.npars:
            raise self.log.fatal(
                f"<ProcessOptions> Mismatch in parameter string: the number of parameters: "
                f"{self.npars} != ranges defined: {len(pieces)}; the format of the "
                '"ParRanges" string must be: "(-1.2,3.4);(-2.3,4.55);...", where the numbers in '
                '"(a,b)" correspond to the a=min, b=max parameter ranges; each parameter defined '
                "in the function string must have a corresponding rang."
            )
        self.intervals = [_range(piece) for piece in pieces]
        for index, (low, high) in enumerate(self.intervals):
            if low > high:
                raise self.log.fatal(
                    f"<ProcessOptions> max > min in interval for parameter: [{index}] : "
                    f"[{low:g}, {high:g}] "
                )

    def _compile(self) -> None:
        try:
            self.tformula = tformula_text(self.formula, self.npars, self.dsi.GetNVariables())
            self.function = compile_fda(self.tformula)
        except FormulaError as error:
            raise self.log.fatal(str(error)) from None

    def _say_formula(self) -> None:
        self.log.info(f'User-defined formula string       : "{self.formula}"')
        self.log.info(f'TFormula-compatible formula string: "{self.tformula}"')

    def booked(self) -> None:
        for index, (low, high) in enumerate(self.intervals):
            self.log.info(f"Create parameter interval for parameter {index} : [{low:g},{high:g}]")
        self._say_formula()

    # -- the function ---------------------------------------------------------------------

    def _values(self, parameters: Any, values: Any) -> Any:
        """The formula for rows of parameter sets against every event: shape ``(sets, events)``."""
        columns = [parameters[:, i][:, None] for i in range(self.npars)]
        columns += [values[:, i] for i in range(values.shape[1])]
        return np.broadcast_to(self.function(columns), (len(parameters), len(values)))

    def batch_estimator(self, parameters: Any) -> Any:
        """``EstimatorFunction`` of many parameter sets at once."""
        parameters = np.asarray(parameters, dtype=np.float64)
        events = self._events
        weights = events.weights.astype(np.float64)
        if self.analysis == REGRESSION:
            found = np.zeros(len(parameters))
            for dim in range(self.dims):
                deviation = (self._values(parameters, events.values) - events.targets[:, dim]) ** 2
                found += deviation @ weights
            return found / self._sums[2]
        if self.analysis == MULTICLASS:
            outputs = np.stack(
                [
                    self._values(
                        parameters[:, d * self.npars : (d + 1) * self.npars], events.values
                    )
                    for d in range(self.dims)
                ]
            )
            chosen = np.take_along_axis(outputs, events.classes[None, None, :], axis=0)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                return (np.log(chosen) @ weights) / self._sums[2]
        signal = events.classes == self.dsi.GetSignalClassIndex()
        deviation = (self._values(parameters, events.values) - signal) ** 2
        return (deviation[:, ~signal] @ weights[~signal]) / self._sums[0] + (
            deviation[:, signal] @ weights[signal]
        ) / self._sums[1]

    def EstimatorFunction(self, parameters: Any) -> float:
        return float(self.batch_estimator(np.asarray([list(parameters)]))[0])

    # -- training -------------------------------------------------------------------------

    def train(self, events: Events) -> None:
        self._events = events
        self._weigh(events)
        ranges = [Interval(low, high) for low, high in self.intervals] * self.dims
        kind = GeneticFitter if self.fit == "GA" else MCFitter
        fitter = kind(self, f"{self.name}_Fitter_{self.fit}", ranges, self.options.text)
        best = CxxVector(interval.GetMean() for interval in ranges)
        estimator = fitter.Run(best)
        self.parameters = [float(value) for value in best]
        del self._events
        self._say_fit(estimator)

    def _weigh(self, events: Events) -> None:
        """The background's, the signal's and all the events' weight; a class without is refused."""
        weights = events.weights.astype(np.float64)
        signal = events.classes == self.dsi.GetSignalClassIndex()
        self._sums = (
            float(weights[~signal].sum()),
            float(weights[signal].sum()),
            float(weights.sum()),
        )
        if self.analysis == CLASSIFICATION and min(self._sums[:2]) <= 0:
            raise self.log.fatal(
                f"<Train> Troubles in sum of weights: {self._sums[1]:g} (S) : {self._sums[0]:g} (B)"
            )

    def _say_fit(self, estimator: float) -> None:
        """The fitted parameters, the formula and the least estimator, as TMVA prints them."""
        self.log.header(f'Results for parameter fit using "{self.fit}" fitter:')
        names = [f"Par({index})" for index in range(len(self.parameters))]
        for line in formatted_values(names, self.parameters, "Parameter", "Fit result", "{:g}"):
            self.log.info(line)
        self.log.info(f'Discriminator expression: "{self.formula}"')
        self.log.info(f"Value of estimator at minimum: {estimator:g}")

    # -- evaluation -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        values = np.asarray(values, dtype=np.float64)
        sets = np.asarray(self.parameters).reshape(self.dims, self.npars)
        if self.analysis == CLASSIFICATION:
            return self._values(sets[:1], values)[0]
        if self.analysis == REGRESSION:
            return self.handler.inverse_targets(self._values(sets[:1], values)[0][:, None])
        outputs = np.stack([self._values(sets[d : d + 1], values)[0] for d in range(self.dims)], 1)
        shifted = outputs[:, None, :] - outputs[:, :, None]
        return 1.0 / np.exp(shifted).sum(axis=2)

    # -- the weight file ------------------------------------------------------------------

    def add_weights(self, node: Node) -> None:
        weights = node.add("Weights", NPars=self.npars, NDim=self.dims)
        for index, value in enumerate(self.parameters):
            weights.add("Parameter", Index=index, Value=number(value))
        weights.set("Formula", self.formula)

    def read_weights(self, node: Any) -> None:
        self.npars = int(node.get("NPars"))
        self.dims = int(node.get("NDim", 1))
        self.parameters = [0.0] * (self.npars * self.dims)
        for item in children(node, "Parameter"):
            self.parameters[int(str(item.get("Index")))] = float(str(item.get("Value")))
        self.formula = str(node.get("Formula"))
        self._compile()
        self._say_formula()
