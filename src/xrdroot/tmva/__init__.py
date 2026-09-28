"""TMVA - ROOT's Toolkit for MultiVariate Analysis - over NumPy, scikit-learn, XGBoost and PyTorch.

    >>> import xrdroot.pyroot as ROOT                                   # doctest: +SKIP
    >>> loader = ROOT.TMVA.DataLoader("dataset")                         # doctest: +SKIP
    >>> loader.AddVariable("var1", "F")                                  # doctest: +SKIP
    >>> factory = ROOT.TMVA.Factory("job", output, "AnalysisType=Classification")  # doctest: +SKIP
    >>> factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT", "NTrees=200")      # doctest: +SKIP

The classes are TMVA's, by TMVA's names, taking TMVA's option strings: a
``DataLoader`` of variables and trees, a ``Factory`` that books, trains,
tests and evaluates methods and writes TMVA's output file and weight
files, and a ``Reader`` that applies them. What TMVA works out in closed
form - the data set's split, the transformations, the linear discriminants,
the projective likelihood, the efficiencies and ROC integrals - is worked
out here the same way, to TMVA's numbers; what it trains iteratively is
trained by scikit-learn, XGBoost or PyTorch, to results as good but not
bit for bit the same. See ``docs/root.md``, "TMVA", for which is which.
"""

from __future__ import annotations

from typing import Any

from .crossval import CrossValidation
from .factory import Factory
from .genetic import GeneticAlgorithm, GeneticFitter, IFitterTarget, Interval
from .loader import DataLoader
from .log import TMVAError
from .methods.cuts import MethodCuts
from .reader import Reader
from .tools import Tools, gConfig, gTools
from .types import Types

__all__ = [
    "CrossValidation",
    "DataLoader",
    "Factory",
    "GeneticAlgorithm",
    "GeneticFitter",
    "IFitterTarget",
    "Interval",
    "MethodCuts",
    "Reader",
    "TMVAError",
    "Tools",
    "Types",
    "gConfig",
    "gTools",
]

#: ``TMVA::Experimental``'s classes, by name.
EXPERIMENTAL: dict[str, Any] = {}
