"""TMVA's methods, one module each, and the registry the Factory and the Reader book them from.

Each method is registered under its ``GetMethodTypeName`` - ``"BDT"``,
``"Likelihood"`` - which is what ``Types::kBDT`` names and what a weight
file's ``Method="BDT::..."`` says. A method TMVA has and this does not is
refused by name when it is booked.
"""

from __future__ import annotations

from ..method import Method
from .linear import MethodFisher, MethodLD

__all__ = ["REGISTRY", "Method"]

#: Every method there is, by its type name.
REGISTRY: dict[str, type[Method]] = {
    kind.type_name: kind
    for kind in (
        MethodLD,
        MethodFisher,
    )
}
