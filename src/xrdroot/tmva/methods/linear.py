"""``LD`` and ``Fisher``: the linear discriminants, worked out in closed form as TMVA does.

``LD`` is the least-squares fit of the class (1 for signal, 0 for
background) - or of the targets, for a regression - to a linear function of
the variables; ``Fisher`` projects onto the axis that best separates the
class means given the covariance within the classes (``Method=Fisher``) or
of all events (``Mahalanobis``). Both are TMVA's sums and inversions,
event weights and all, so the coefficients are TMVA's to the last digits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, REGRESSION, Method
from ..xmlfile import Node, children, number

__all__ = ["MethodFisher", "MethodLD", "coefficient_table", "formatted_values"]


def coefficient_table(labels: list[str], values: list[float]) -> list[str]:
    """The coefficients' table: ``Variable:  Coefficient:`` and the rows."""
    return formatted_values(labels, values, "Variable", "Coefficient", "{:+1.3f}")


def formatted_values(
    labels: list[str], values: list[float], title_vars: str, title_values: str, fmt: str
) -> list[str]:
    """``Tools::FormattedOutput`` of values: a titled column of names and one of values."""
    width = max([7, len(title_vars), *(len(label) for label in labels)])
    column = max(len(title_values) + 1, width)
    rule = "-" * (width + column + 3)
    rows = [rule, title_vars.rjust(width) + ":" + title_values.rjust(column + 1) + ":", rule]
    for label, value in zip(labels, values):
        rows.append(label.rjust(width) + ":" + fmt.format(value).rjust(column + 1))
    return [*rows, rule]


def _print_coefficients(method: Method, title: str, values: list[float]) -> None:
    """``PrintCoefficients``: the table, and the transformations it applies to, as TMVA prints."""
    method.log.header(f"Results for {title} coefficients:")
    if method.handler.transforms:
        method.log.info("NOTE: The coefficients must be applied to TRANFORMED variables")
        method.log.info("  List of the transformation: ")
        for transform in method.handler.transforms:
            method.log.info(f"  -- {transform.name}")
    labels = [*(variable.label for variable in method.dsi.variables), "(offset)"]
    for line in coefficient_table(labels, values):
        method.log.info(line)


def _checked_inverse(matrix: Any, log: Logger, where: str) -> Any:
    """The inverse, after TMVA's warning of a nearly singular matrix, or its refusal of one."""
    determinant = abs(float(np.linalg.det(matrix)))
    if determinant < 10e-24:
        log.warning(
            f"<{where}> matrix is almost singular with determinant={determinant:g} did you use "
            "the variables that are linear combinations or highly correlated?"
        )
    if determinant < 10e-120:
        raise log.fatal(
            f"<{where}> matrix is singular with determinant={determinant:g} did you use the "
            "variables that are linear combinations?"
        )
    return np.linalg.inv(matrix)


class MethodLD(Method):
    """``TMVA::MethodLD``: the linear discriminant, for classification and regression."""

    type_name = "LD"
    analyses = frozenset({CLASSIFICATION, REGRESSION})
    needs_data = True

    def train(self, events: Events) -> None:
        weights = events.weights
        if self.opt("IgnoreNegWeightsInTraining"):
            events = events.take(weights > 0)
            weights = events.weights
        design = np.column_stack([np.ones(len(events)), events.values])
        sums = (design * weights[:, None]).T @ design
        if self.analysis == REGRESSION:
            targets = events.targets
        else:
            targets = (events.classes == self.dsi.GetSignalClassIndex()).astype(np.float64)[:, None]
        right = (design * weights[:, None]).T @ targets
        coefficients = _checked_inverse(sums, self.log, "GetCoeff") @ right
        if self.analysis != REGRESSION:
            offset = float(np.dot(coefficients[1:, 0], sums[0, 1:])) / sums[0, 0]
            coefficients[0, 0] = offset / -2.0
        self.coefficients = coefficients.T
        _print_coefficients(self, "LD", [*self.coefficients[0, 1:], self.coefficients[0, 0]])

    def evaluate(self, values: Any) -> Any:
        output = self.coefficients[:, :1].T + np.asarray(values) @ self.coefficients[:, 1:].T
        return output[:, 0] if self.analysis != REGRESSION else output

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        labels = [variable.label for variable in self.dsi.variables]
        return "Discr. power", list(zip(labels, np.abs(self.coefficients[0, 1:])))

    def add_weights(self, node: Node) -> None:
        nout, ncoeff = self.coefficients.shape
        weights = node.add("Weights", NOut=nout, NCoeff=ncoeff)
        for out in range(nout):
            for index in range(ncoeff):
                weights.add(
                    "Coefficient",
                    IndexOut=out,
                    IndexCoeff=index,
                    Value=number(self.coefficients[out, index]),
                )

    def read_weights(self, node: Any) -> None:
        nout, ncoeff = int(node.get("NOut", 1)), int(node.get("NCoeff"))
        self.coefficients = np.zeros((nout, ncoeff))
        for item in children(node, "Coefficient"):
            out, index = int(item.get("IndexOut", 0)), int(item.get("IndexCoeff", 0))
            self.coefficients[out, index] = float(item.get("Value"))


class MethodFisher(Method):
    """``TMVA::MethodFisher``: Fisher's discriminant, or Mahalanobis's with ``Method=Mahalanobis``."""

    type_name = "Fisher"
    defaults = {"Method": "Fisher"}
    needs_data = True

    def train(self, events: Events) -> None:
        signal = events.classes == self.dsi.GetSignalClassIndex()
        weights, values = events.weights, events.values
        total_s, total_b = float(np.sum(weights[signal])), float(np.sum(weights[~signal]))
        mean_s = weights[signal] @ values[signal] / total_s
        mean_b = weights[~signal] @ values[~signal] / total_b
        mean = weights[signal] @ values[signal] + weights[~signal] @ values[~signal]
        mean = mean / (total_s + total_b)
        within = _scatter(values[signal], weights[signal], mean_s) / total_s
        within = within + _scatter(values[~signal], weights[~signal], mean_b) / total_b
        between = total_s * np.outer(mean_s - mean, mean_s - mean)
        between = (between + total_b * np.outer(mean_b - mean, mean_b - mean)) / (total_s + total_b)
        covariance = within + between
        chosen = covariance if str(self.opt("Method")) == "Mahalanobis" else within
        inverse = _checked_inverse(chosen, self.log, "GetFisherCoeff")
        factor = np.sqrt(total_s * total_b) / (total_s + total_b)
        self.coefficients = inverse @ (mean_s - mean_b) * factor
        self.offset = -float(np.dot(self.coefficients, mean_s + mean_b)) / 2.0
        diagonal = np.diag(covariance)
        self.power = np.where(
            diagonal != 0, np.diag(between) / np.where(diagonal != 0, diagonal, 1), 0
        )
        _print_coefficients(self, "Fisher", [*self.coefficients, self.offset])

    def evaluate(self, values: Any) -> Any:
        return self.offset + np.asarray(values) @ self.coefficients

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        labels = [variable.label for variable in self.dsi.variables]
        return "Discr. power", list(zip(labels, self.power))

    def add_weights(self, node: Node) -> None:
        weights = node.add("Weights", NCoeff=len(self.coefficients) + 1)
        weights.add("Coefficient", Index=0, Value=number(self.offset))
        for index, value in enumerate(self.coefficients, start=1):
            weights.add("Coefficient", Index=index, Value=number(value))

    def read_weights(self, node: Any) -> None:
        values = {int(item.get("Index")): float(item.get("Value")) for item in children(node)}
        self.offset = values.get(0, 0.0)
        self.coefficients = np.array([values[index] for index in range(1, len(values))])
        self.power = np.zeros(len(self.coefficients))


def _scatter(values: Any, weights: Any, mean: Any) -> Any:
    centred = values - mean
    return (centred * weights[:, None]).T @ centred
