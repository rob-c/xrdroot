"""TMVA's methods, one module each, and the registry the Factory and the Reader book them from.

Each method is registered under its ``GetMethodTypeName`` - ``"BDT"``,
``"Likelihood"`` - which is what ``Types::kBDT`` names and what a weight
file's ``Method="BDT::..."`` says. A method TMVA has and this does not is
refused by name when it is booked.
"""

from __future__ import annotations

from ..method import Method
from .bdt import MethodBDT
from .cuts import MethodCuts
from .dl import MethodDL, MethodDNN
from .fda import MethodFDA
from .knn import MethodKNN
from .likelihood import MethodLikelihood
from .linear import MethodFisher, MethodLD
from .mlp import MethodMLP
from .svm import MethodSVM

__all__ = ["REGISTRY", "Method"]

#: Every method there is, by its type name.
REGISTRY: dict[str, type[Method]] = {
    kind.type_name: kind
    for kind in (
        MethodLD,
        MethodFisher,
        MethodLikelihood,
        MethodBDT,
        MethodMLP,
        MethodDL,
        MethodDNN,
        MethodKNN,
        MethodSVM,
        MethodCuts,
        MethodFDA,
    )
}
