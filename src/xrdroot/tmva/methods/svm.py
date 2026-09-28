"""``SVM``: a support vector machine, trained by scikit-learn, written as TMVA's support vectors.

``Kernel=RBF`` (``K = exp(-Gamma |x - y|^2)``), ``Polynomial`` or
``MultiGauss``, the cost ``C`` - weighted per class by the ratio of their
numbers of events, as TMVA's ``CSig`` and ``CBkg`` are, and per event by its
weight, as TMVA's ``SVEvent`` weighs it - and the tolerance
``Tol`` are scikit-learn's ``SVC``'s (``SVR``'s for a regression). The
output is TMVA's: ``1 / (1 + exp(f))`` of ``f = sum(alpha y K) - b``, the
support vectors stored in TMVA's layout with their signs arranged so that
the signal is high, and TMVA's own SVM weight files are read.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import Events
from ..log import Logger
from ..method import CLASSIFICATION, REGRESSION, Method
from ..xmlfile import Node, children, floats, number

__all__ = ["MethodSVM"]


def _kernel(kind: str, gamma: Any, order: int, theta: float, a: Any, b: Any) -> Any:
    """The kernel between every row of ``a`` and every row of ``b``."""
    if kind == "Polynomial":
        return (a @ b.T + theta) ** order
    if kind == "MultiGauss":
        scaled_a, scaled_b = a / np.sqrt(gamma), b / np.sqrt(gamma)
        squared = (
            (scaled_a**2).sum(1)[:, None]
            + (scaled_b**2).sum(1)[None, :]
            - 2 * scaled_a @ scaled_b.T
        )
        return np.exp(-np.maximum(squared, 0.0))
    squared = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T
    return np.exp(-gamma * np.maximum(squared, 0.0))


class MethodSVM(Method):
    """``TMVA::MethodSVM``."""

    type_name = "SVM"
    analyses = frozenset({CLASSIFICATION, REGRESSION})
    defaults = {
        "Kernel": "RBF",
        "Gamma": 1.0,
        "Order": 3,
        "Theta": 1.0,
        "GammaList": "",
        "Tune": "All",
        "KernelList": "None",
        "Loss": "hinge",
        "C": 1.0,
        "Tol": 0.01,
        "MaxIter": 1000,
    }

    def process_options(self) -> None:
        if self.analysis == REGRESSION and not self.options.given("C"):
            self.option_values["C"] = 0.002
        self.kind = str(self.opt("Kernel"))
        if self.kind not in ("RBF", "Polynomial", "MultiGauss"):
            raise self.log.fatal(f"{self.kind} is not a recognised kernel function.")
        self.gamma: Any = float(self.opt("Gamma"))
        if self.kind == "MultiGauss":
            listed = [float(x) for x in str(self.opt("GammaList")).split(",") if x.strip()]
            self.gamma = np.array(listed or [1.0] * self.dsi.GetNVariables())

    def _sklearn_kernel(self) -> dict[str, Any]:
        if self.kind == "Polynomial":
            return {
                "kernel": "poly",
                "degree": int(self.opt("Order")),
                "coef0": float(self.opt("Theta")),
                "gamma": 1.0,
            }
        if self.kind == "MultiGauss":
            return {"kernel": lambda a, b: _kernel("MultiGauss", self.gamma, 0, 0.0, a, b)}
        return {"kernel": "rbf", "gamma": self.gamma}

    def train(self, events: Events) -> None:
        try:
            from sklearn import svm
        except ImportError as why:  # pragma: no cover - scikit-learn comes with the tmva extra
            raise self.log.fatal(
                "Training an SVM needs scikit-learn, which is not installed; "
                "pip install xrdroot[tmva] installs it"
            ) from why
        events = events.take(events.weights != 0)
        values = np.asarray(events.values, dtype=np.float64)
        self.log.info(f"Building SVM Working Set...with {len(values)} event instances")
        self.log.info("Sorry, no computing time forecast available for SVM, please wait ...")
        options = {
            "C": float(self.opt("C")),
            "tol": float(self.opt("Tol")),
            **self._sklearn_kernel(),
        }
        if self.analysis == REGRESSION:
            fitted = svm.SVR(**options).fit(
                values, events.targets[:, 0], sample_weight=events.weights
            )
            self._store(fitted, values, 1.0)
            return
        signal = events.classes == self.dsi.GetSignalClassIndex()
        nsig, nbkg = int(signal.sum()), int((~signal).sum())
        ratio = nsig / nbkg if nbkg else 1.0
        weights = {1: 1.0, 0: ratio} if nsig < nbkg else {1: ratio, 0: 1.0}
        fitted = svm.SVC(class_weight=weights, **options)
        fitted.fit(values, signal.astype(int), sample_weight=events.weights)
        self._store(fitted, values, 1.0)

    def _store(self, fitted: Any, values: Any, sign: float) -> None:
        """The support vectors, their ``alpha * y`` as TMVA's ``alpha`` and ``flag``, and ``b``."""
        dual = np.asarray(fitted.dual_coef_, dtype=np.float64)[0] * sign
        self.alphas = np.abs(dual)
        self.flags = -np.sign(dual)
        self.vectors = np.asarray(values[fitted.support_], dtype=np.float64)
        self.bias = float(np.asarray(fitted.intercept_)[0]) * sign

    def _decision(self, values: Any) -> Any:
        kernel = _kernel(
            self.kind,
            self.gamma,
            int(self.opt("Order")),
            float(self.opt("Theta")),
            self.vectors,
            np.asarray(values, dtype=np.float64),
        )
        return (self.alphas * self.flags) @ kernel - self.bias

    def evaluate(self, values: Any) -> Any:
        f = self._decision(values)
        if self.analysis == REGRESSION:
            return self.handler.inverse_targets(-f[:, None])
        return 1.0 / (1.0 + np.exp(f))

    def add_weights(self, node: Node) -> None:
        weights = node.add(
            "Weights",
            fBparm=number(self.bias),
            fGamma=number(np.mean(self.gamma)),
            fGammaList=str(self.opt("GammaList")),
            fTheta=number(self.opt("Theta")),
            fOrder=int(self.opt("Order")),
            NSupVec=len(self.alphas),
        )
        for number_, (alpha, flag, vector) in enumerate(zip(self.alphas, self.flags, self.vectors)):
            row = [number_ + 1, flag, alpha, 0.0, *vector]
            weights.add("SupportVector", Rows=1, Columns=len(row)).block(row, 15)
        infos = self.dsi.variables
        weights.add("Maxima", **{f"Var{i}": number(v.maximum) for i, v in enumerate(infos)})
        weights.add("Minima", **{f"Var{i}": number(v.minimum) for i, v in enumerate(infos)})

    def read_weights(self, node: Any) -> None:
        self.bias = float(node.get("fBparm"))
        if self.kind != "MultiGauss":
            self.gamma = float(node.get("fGamma"))
        rows = np.array([floats(item) for item in children(node, "SupportVector")])
        if not len(rows):
            raise Logger("SVM").fatal("The weight file holds no support vectors")
        single = rows.astype(np.float32).astype(np.float64)
        self.flags, self.alphas, self.vectors = rows[:, 1], single[:, 2], single[:, 4:]
