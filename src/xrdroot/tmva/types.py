"""``TMVA::Types``: the numbers TMVA names its methods, analyses and samples by.

``TMVA::Types::kBDT`` is 9 and names the method ``"BDT"``; a macro books a
method by either, and ``Types::Instance().GetMethodName(kBDT)`` turns one
into the other, as here.
"""

from __future__ import annotations

from typing import Any

__all__ = ["METHOD_NAMES", "Types"]

#: ``Types::EMVA``, in order: each method's number is its place here.
METHOD_NAMES = (
    "Variable",
    "Cuts",
    "Likelihood",
    "PDERS",
    "HMatrix",
    "Fisher",
    "KNN",
    "CFMlpANN",
    "TMlpANN",
    "BDT",
    "DT",
    "RuleFit",
    "SVM",
    "MLP",
    "BayesClassifier",
    "FDA",
    "Boost",
    "PDEFoam",
    "LD",
    "Plugins",
    "Category",
    "DNN",
    "DL",
    "PyRandomForest",
    "PyAdaBoost",
    "PyGTB",
    "PyKeras",
    "PyTorch",
    "C50",
    "RSNNS",
    "RSVM",
    "RXGB",
    "CrossValidation",
)


class Types:
    """``TMVA::Types``: every enumeration as a class attribute, and the name lookups."""

    # -- EAnalysisType --
    kClassification, kRegression, kMulticlass, kNoAnalysisType, kMaxAnalysisType = range(5)
    # -- ETreeType --
    kTraining, kTesting, kMaxTreeType, kValidation, kTrainingOriginal = range(5)
    # -- ESBType --
    kSignal, kBackground, kSBBoth, kMaxSBType, kTrueType = range(5)
    # -- EVariableTransform --
    (kIdentity, kDecorrelated, kNormalized, kPCA, kRearranged, kGauss, kUniform) = range(7)
    kMaxMethod = len(METHOD_NAMES)

    _instance: Types | None = None

    @classmethod
    def Instance(cls) -> Types:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def GetMethodName(method: Any) -> str:
        """The method's name for its number - ``"BDT"`` for ``kBDT`` - or the name itself."""
        if isinstance(method, int):
            return METHOD_NAMES[method] if 0 <= method < len(METHOD_NAMES) else ""
        return str(method)

    @staticmethod
    def GetMethodType(name: Any) -> int:
        """The method's number for its name, ``kMaxMethod`` for one TMVA has not."""
        text = str(name)
        return METHOD_NAMES.index(text) if text in METHOD_NAMES else len(METHOD_NAMES)


for _number, _name in enumerate(METHOD_NAMES):
    setattr(Types, f"k{_name}", _number)
