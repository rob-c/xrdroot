"""Every name of the ``RooStats`` namespace this package has, by module - for ``import ROOT``.

:mod:`xrdroot.pyroot.roostats` makes ``ROOT.RooStats`` of these, so that
``ROOT.RooStats.ProfileLikelihoodCalculator`` - and, in a translated macro
that says ``using namespace RooStats``, a bare ``ProfileLikelihoodCalculator`` -
is the class here, and a name RooStats has that is not listed is refused by
that name.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["MODULES", "members"]

#: Each module, and the names of RooStats it defines.
MODULES: dict[str, list[str]] = {
    "belt": ["AcceptanceRegion", "ConfidenceBelt", "PointSetInterval"],
    "neyman": ["FeldmanCousins", "NeymanConstruction"],
    "nuisance": ["NuisanceParametersSampler"],
    "sampling": ["SamplingDistribution"],
    "teststats": ["ProfileLikelihoodTestStat", "TestStatistic"],
    "toymc": ["ToyMCSampler"],
    "combined": ["CombinedCalculator", "ProfileLikelihoodCalculator"],
    "intervals": ["ConfInterval", "SimpleInterval"],
    "likelihoodinterval": ["LikelihoodInterval"],
    "modelconfig": ["ModelConfig"],
    "bayesian": ["BayesianCalculator"],
    "hypotest": ["HypoTestResult"],
    "markov": ["MarkovChain", "MetropolisHastings", "ProposalFunction", "SequentialProposal",
               "UniformProposal"],
    "mcmc": ["MCMCCalculator", "MCMCInterval"],
    "proposals": ["PdfProposal", "ProposalHelper"],
    "calculators": ["FrequentistCalculator", "HybridCalculator", "HypoTestCalculatorGeneric"],
    "moretests": [
        "MaxLikelihoodEstimateTestStat",
        "NumEventsTestStat",
        "RatioOfProfiledLikelihoodsTestStat",
        "SimpleLikelihoodRatioTestStat",
    ],
    "numbercounting": ["NumberCountingPdfFactory"],
    "asymcalc": ["AsymptoticCalculator"],
    "config": ["GetGlobalRooStatsConfig", "NLLOffsetMode", "SetNLLOffsetMode", "UseNLLOffset"],
    "utils": [
        "AsimovSignificance",
        "FactorizePdf",
        "MakeNuisancePdf",
        "MakeUnconstrainedPdf",
        "StripConstraints",
        "NumberCountingUtils",
        "PValueToSignificance",
        "RemoveConstantParameters",
        "SetAllConstant",
        "SetParameters",
        "SignificanceToPValue",
    ],
}


def members() -> dict[str, Any]:
    """Every name, against what it is."""
    found: dict[str, Any] = {}
    for module, names in MODULES.items():
        loaded = importlib.import_module(f"xrdroot.roostats.{module}")
        for name in names:
            found[name] = getattr(loaded, name)
    return found
