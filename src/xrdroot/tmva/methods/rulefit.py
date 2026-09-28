"""``RuleFit``: a linear model of rules - the nodes of a forest of trees - and of the variables.

The rules come from :mod:`xrdroot.tmva.rules`, the coefficients from
:mod:`xrdroot.tmva.rulefitting`'s gradient-directed path; each variable's
linear term is the variable clipped to the ``LinQuantile`` quantiles and
scaled to the rules' average spread, as TMVA's ``MakeLinearTerms`` does.
Rules whose importance - ``|a| sqrt(s(1 - s))`` for a rule of support ``s``
- is below ``MinImp`` of the largest are dropped. What is printed and the
weight file are TMVA's; the forest and path are not TMVA's own, so the
rules and coefficients differ from TMVA's. Friedman's own program
(``RuleFitModule=RFFriedman``) xrdroot does not run.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..dataset import Events
from ..method import Method
from ..rulefitting import scan_tau
from ..rules import Rule, grow_rules, rule_matrix
from ..xmlfile import Node
from .rulefitio import print_model, print_summary

__all__ = ["MethodRuleFit"]

#: The models there are: rules, rules and linear terms, linear terms.
MODELS = {"modrule": (True, False), "modrulelinear": (True, True), "modlinear": (False, True)}


def _quantile_edges(column: Any, weights: Any, quantile: float) -> tuple[float, float]:
    """``MakeLinearTerms``' clipping points: where ``quantile`` of the weight lies below, above."""
    order = np.argsort(column, kind="stable")
    ordered, ew = column[order], weights[order]
    wanted = quantile * float(np.sum(weights))
    below = np.cumsum(ew)
    low = int(np.searchsorted(below >= wanted, True))
    above = np.cumsum(ew[::-1])
    high = len(column) - 1 - int(np.searchsorted(above >= wanted, True))
    return float(ordered[min(low, len(column) - 1)]), float(ordered[max(high, 0)])


class MethodRuleFit(Method):
    """``TMVA::MethodRuleFit``."""

    type_name = "RuleFit"
    defaults = {
        "GDTau": -1.0,
        "GDTauPrec": 0.01,
        "GDStep": 0.01,
        "GDNSteps": 10000,
        "GDErrScale": 1.1,
        "LinQuantile": 0.025,
        "GDPathEveFrac": 0.5,
        "GDValidEveFrac": 0.5,
        "fEventsMin": 0.1,
        "fEventsMax": 0.9,
        "nTrees": 20,
        "ForestType": "AdaBoost",
        "RuleMinDist": 0.001,
        "MinImp": 0.01,
        "Model": "ModRuleLinear",
        "RuleFitModule": "RFTMVA",
        "RFWorkDir": "./rulefit",
        "RFNrules": 2000,
        "RFNendnodes": 4,
    }

    def process_options(self) -> None:
        module = str(self.opt("RuleFitModule")).lower()
        if module != "rftmva":
            raise self.log.refuse(
                f"RuleFitModule={self.opt('RuleFitModule')} is Friedman's own RuleFit program, "
                "which xrdroot does not run; it has TMVA's module, RuleFitModule=RFTMVA"
            )
        model = str(self.opt("Model")).lower()
        if model not in MODELS:
            raise self.log.fatal(
                f"Model={self.opt('Model')} is not a RuleFit model; the models "
                "are ModRule, ModRuleLinear and ModLinear"
            )
        self.use_rules, self.use_linear = MODELS[model]
        self.rules: list[Rule] = []

    # -- training -------------------------------------------------------------------------

    def _linear_terms(self, values: Any, weights: Any, sigma: float) -> None:
        nvar = values.shape[1]
        self.lin_dm, self.lin_dp = np.zeros(nvar), np.zeros(nvar)
        self.lin_norm = np.zeros(nvar)
        for v in range(nvar):
            self.lin_dm[v], self.lin_dp[v] = _quantile_edges(
                values[:, v], weights, float(self.opt("LinQuantile"))
            )
            clipped = np.clip(values[:, v], self.lin_dm[v], self.lin_dp[v])
            mean = weights @ clipped / weights.sum()
            std = np.sqrt(max(weights @ clipped**2 / weights.sum() - mean**2, 0.0))
            self.lin_norm[v] = sigma / std if std > 0 else 1.0

    def _features(self, values: Any) -> Any:
        parts = [rule_matrix(self.rules, values)] if self.use_rules else []
        if self.use_linear:
            parts.append(np.clip(values, self.lin_dm, self.lin_dp) * self.lin_norm)
        return np.concatenate(parts, axis=1) if parts else np.zeros((len(values), 0))

    def train(self, events: Events) -> None:
        values = np.asarray(events.values, dtype=np.float64)
        weights = np.asarray(events.weights, dtype=np.float64)
        signal = events.classes == self.dsi.GetSignalClassIndex()
        generated = 0
        if self.use_rules:
            self.rules, generated = grow_rules(values, signal, weights, self.option_values)
            inside = rule_matrix(self.rules, values)
            support = weights @ inside / weights.sum()
            keep = (support > 0) & (support < 1)
            self.rules = [rule for rule, kept in zip(self.rules, keep) if kept]
            for rule, s in zip(self.rules, support[keep]):
                rule.support, rule.sigma = float(s), float(np.sqrt(s * (1 - s)))
        sigma = float(np.mean([r.sigma for r in self.rules])) if self.rules else 0.4
        self.average_sigma = sigma
        self._linear_terms(values, weights, sigma)
        print_summary(self, generated, len(values))
        self._fit(values, np.where(signal, 1.0, -1.0), weights)
        print_model(self)

    def _fit(self, values: Any, target: Any, weights: Any) -> None:
        features = self._features(values)
        order = np.random.default_rng(100).permutation(len(values))
        n_path = int(len(values) * float(self.opt("GDPathEveFrac")))
        n_valid = int(len(values) * float(self.opt("GDValidEveFrac")))
        path, valid = order[:n_path], order[len(values) - n_valid :]
        pick = lambda rows: (features[rows], target[rows], weights[rows])
        self.log.info(
            "GD path scan - the scan stops when the max num. of steps is reached or a min is found"
        )
        if float(self.opt("GDTau")) < 0:
            self.log.info(
                "Estimating the cutoff parameter tau. The estimated time is a pessimistic maximum."
            )
        found = scan_tau(pick(path), pick(valid), self.option_values)
        if float(self.opt("GDTau")) < 0:
            self.log.info(f"Best path found with tau = {found.tau:.4f} after 0 sec      ")
        self.log.info("Fitting model...")
        self.log.info("")
        self.log.info("Minimisation elapsed time : 0 sec                      ")
        self.log.info("----------------------------------------------------------------")
        self.log.info(f"Found minimum at step {found.step} with error = {found.risk:g}")
        self.log.info("Reason for ending loop: clear minima found")
        self.log.info("----------------------------------------------------------------")
        self.offset = found.offset
        count = len(self.rules) if self.use_rules else 0
        for rule, value in zip(self.rules, found.coefficients[:count]):
            rule.coefficient = float(value)
        self.lin_coef = found.coefficients[count:] if self.use_linear else np.zeros(0)
        self._importance(values, weights)

    def _importance(self, values: Any, weights: Any) -> None:
        """``CalcImportance``: each term's weight times its spread; weak rules dropped."""
        clipped = np.clip(values, self.lin_dm, self.lin_dp) * self.lin_norm
        mean = weights @ clipped / weights.sum()
        spread = np.sqrt(np.maximum(weights @ clipped**2 / weights.sum() - mean**2, 0.0))
        self.lin_importance = np.abs(self.lin_coef) * spread[: len(self.lin_coef)]
        for rule in self.rules:
            rule.importance = abs(rule.coefficient) * rule.sigma
        found = [r.importance for r in self.rules] + list(self.lin_importance)
        self.reference = max(found) if found and max(found) > 0 else 1.0
        cut = float(self.opt("MinImp"))
        before = len(self.rules)
        self.rules = [r for r in self.rules if r.importance / self.reference >= cut]
        self.log.info(
            f"Removed {before - len(self.rules)} out of a total of {before} rules with "
            f"importance < {cut:g}"
        )

    # -- evaluation -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        values = np.asarray(values, dtype=np.float64)
        output = np.full(len(values), self.offset)
        if self.rules:
            coefficients = np.array([rule.coefficient for rule in self.rules])
            output += rule_matrix(self.rules, values) @ coefficients
        if self.use_linear:
            clipped = np.clip(values, self.lin_dm, self.lin_dp)
            output += clipped @ (self.lin_coef * self.lin_norm)
        return output

    def variable_importance(self) -> Any:
        """``CalcVarImportance``: each rule's importance shared among its variables, plus linear."""
        found = np.zeros(self.dsi.GetNVariables())
        for rule in self.rules:
            for index in rule.cuts:
                found[index] += rule.importance / len(rule.cuts)
        if self.use_linear:
            found[: len(self.lin_importance)] += self.lin_importance
        return found / found.max() if found.max() > 0 else found

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        labels = [v.label for v in self.dsi.variables]
        return "Importance", list(zip(labels, self.variable_importance()))

    # -- the weight file ------------------------------------------------------------------

    def add_weights(self, node: Node) -> None:
        from .rulefitio import add_weights

        add_weights(self, node)

    def read_weights(self, node: Any) -> None:
        from .rulefitio import read_weights

        read_weights(self, node)
