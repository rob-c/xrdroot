"""What ``RuleFit`` prints of its ensemble and model, and its weight file, in TMVA's formats."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..rules import Rule
from ..xmlfile import Node, children, number

__all__ = ["add_weights", "print_model", "print_summary", "read_weights"]

#: ``ELearningModel``, as the weight file numbers it.
LEARNING = {(True, True): 0, (True, False): 1, (False, True): 2}
#: A rule of ``=`` signs.
BAR = "================================================================"
#: A rule of ``-`` signs.
DASHES = "----------------------------------------------------------------"


def print_summary(method: Any, generated: int, events: int) -> None:
    """``PrintRuleGen``: the ensemble's summary."""
    log = method.log
    cuts = np.array([len(rule.cuts) for rule in method.rules], dtype=np.float64)
    log.header("-------------------RULE ENSEMBLE SUMMARY------------------------")
    boost = str(method.opt("ForestType")).lower() == "adaboost"
    log.info(f"Tree training method               : {'AdaBoost' if boost else 'Random'}")
    log.info(f"Number of events per tree          : {events}")
    log.info(f"Number of trees                    : {int(method.opt('nTrees'))}")
    log.info(f"Number of generated rules          : {generated}")
    log.info(f"Idem, after cleanup                : {len(method.rules)}")
    log.info(f"Average number of cuts per rule    : {cuts.mean() if len(cuts) else 0:8.2f}")
    log.info(
        f"Spread in number of cuts per rules : {cuts.std(ddof=1) if len(cuts) > 1 else 0:8.2f}"
    )
    log.info(DASHES)
    log.info("")


def _cut_line(rule: Rule, labels: list[str]) -> list[str]:
    lines = []
    for number_, (index, (low, high)) in enumerate(sorted(rule.cuts.items()), start=1):
        text = f"            Cut {number_:2d} : "
        text += f"{low:10.3g} < " if low is not None else "             "
        text += labels[index]
        text += f" < {high:10.3g}" if high is not None else "             "
        lines.append(text)
    return lines


def print_model(method: Any) -> None:
    """``RuleEnsemble::Print``: the offset, the linear terms, and the ten most important rules."""
    log, labels = method.log, [v.label for v in method.dsi.variables]
    width = max(len(label) for label in labels)
    for line in ("", BAR, "                          M o d e l                             ", BAR):
        log.info(line)
    log.header(f"Offset (a0) = {method.offset:g}")
    if method.use_linear:
        rule_ = "------------------------------------"
        log.info(rule_)
        log.info("Linear model (weights unnormalised)")
        log.info(rule_)
        log.info(f"{'Variable':>{width}} : {' Weights':>11} : Importance")
        log.info(rule_)
        for index, label in enumerate(labels):
            weight = method.lin_coef[index] * method.lin_norm[index]
            share = method.lin_importance[index] / method.reference
            log.info(f"{label:>{max(width, 8)}} :  {weight:10.3e} :  {share:3.3f}")
        log.info(rule_)
    else:
        log.info("Linear terms were disabled")
    if method.use_rules and method.rules:
        log.info(f"Number of rules = {len(method.rules)}")
        shown = min(10, len(method.rules))
        log.info(f"Printing the first {shown} rules, ordered in importance.")
        ranked = sorted(method.rules, key=lambda rule: rule.importance, reverse=True)
        for place, rule in enumerate(ranked[:shown], start=1):
            log.info(f"Rule {place:4d} : Importance  = {rule.importance / method.reference:1.4f}")
            for line in _cut_line(rule, labels):
                log.info(line)
        rest = len(method.rules) - shown
        log.info("All rules printed" if not rest else f"Skipping the next {rest} rules")
    elif not method.use_rules:
        log.info("Rule terms were disabled")
    else:
        log.info("Even though rules were included in the model, none passed! 0")
    log.info(BAR)
    log.info("")


def add_weights(method: Any, node: Node) -> None:
    """``RuleFit::AddXMLTo``: the ensemble, its rules and its linear terms."""
    supports = [rule.support for rule in method.rules]
    weights = node.add(
        "Weights",
        NRules=len(method.rules),
        NLinear=len(method.lin_norm),
        LearningModel=LEARNING[(method.use_rules, method.use_linear)],
        ImportanceCut=number(method.opt("MinImp")),
        LinQuantile=number(method.opt("LinQuantile")),
        AverageSupport=number(np.mean(supports) if supports else 0.0),
        AverageRuleSigma=number(method.average_sigma),
        Offset=number(method.offset),
    )
    for rule in method.rules:
        made = weights.add(
            "Rule",
            Importance=number(rule.importance),
            Ref=number(method.reference),
            Coeff=number(rule.coefficient),
            Support=number(rule.support),
            Sigma=number(rule.sigma),
            Norm=number(1.0 / rule.sigma if rule.sigma else 1.0),
            SSB=number(0.0),
            SSBNeve=number(0.0),
            Nvars=len(rule.cuts),
        )
        for index, (low, high) in sorted(rule.cuts.items()):
            made.add(
                "Cut",
                Selector=index,
                Min=number(low or 0.0),
                Max=number(high or 0.0),
                DoMin="T" if low is not None else "F",
                DoMax="T" if high is not None else "F",
            )
    coefficients = method.lin_coef if method.use_linear else np.zeros(len(method.lin_norm))
    importance = method.lin_importance if method.use_linear else np.zeros(len(method.lin_norm))
    for v in range(len(method.lin_norm)):
        weights.add(
            "Linear",
            OK=int(method.use_linear),
            Coeff=number(coefficients[v]),
            Norm=number(method.lin_norm[v]),
            DM=number(method.lin_dm[v]),
            DP=number(method.lin_dp[v]),
            Importance=number(importance[v]),
        )


def read_weights(method: Any, node: Any) -> None:
    """``RuleFit::ReadFromXML``: TMVA's weight files as well as xrdroot's."""
    method.offset = float(node.get("Offset"))
    method.average_sigma = float(node.get("AverageRuleSigma", 0.4))
    method.rules = []
    for item in children(node, "Rule"):
        rule = Rule(
            coefficient=float(item.get("Coeff")),
            importance=float(item.get("Importance")),
            support=float(item.get("Support")),
            sigma=float(item.get("Sigma")),
        )
        for cut in children(item, "Cut"):
            low = float(cut.get("Min")) if cut.get("DoMin") == "T" else None
            high = float(cut.get("Max")) if cut.get("DoMax") == "T" else None
            rule.cuts[int(cut.get("Selector"))] = (low, high)
        method.rules.append(rule)
    linear = children(node, "Linear")
    method.use_linear = any(item.get("OK") == "1" for item in linear)
    method.use_rules = bool(method.rules)
    method.lin_coef = np.array([float(i.get("Coeff")) * (i.get("OK") == "1") for i in linear])
    method.lin_norm = np.array([float(i.get("Norm")) for i in linear])
    method.lin_dm = np.array([float(i.get("DM")) for i in linear])
    method.lin_dp = np.array([float(i.get("DP")) for i in linear])
    method.lin_importance = np.array([float(i.get("Importance")) for i in linear])
