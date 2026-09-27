"""The factory's operators: ``SUM``, ``PROD``, ``EXPR``, ``expr``, ``sum``, ``prod``, ``SIMUL``.

Each takes the factory, the name of what it makes, and its arguments as
text, and builds them as ``RooFactoryWSTool``'s operator of that name does.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import UnsupportedFeatureError
from .factory import split

__all__ = ["OPERATORS"]


def _sum(factory: Any, name: str, args: list[str]) -> Any:
    """``SUM::m(f1*a, f2*b, c)``: a :class:`RooAddPdf` of coefficients times densities."""
    from .pdfs.addpdf import RooAddPdf

    pdfs, coefs = [], []
    for arg in args:
        parts = split(arg, "*")
        if len(parts) == 2:
            coefs.append(factory.build(parts[0]))
        pdfs.append(factory.build(parts[-1]))
    return RooAddPdf(name, name, pdfs, coefs)


def _prod(factory: Any, name: str, args: list[str]) -> Any:
    """``PROD::p(a, b|x)``: a :class:`RooProdPdf`, ``b|x`` conditional on ``x``."""
    from .cmdargs import RooCmdArg
    from .pdfs.prodpdf import RooProdPdf

    plain, conditional = [], []
    for arg in args:
        parts = split(arg, "|")
        pdf = factory.build(parts[0])
        if len(parts) == 2:
            given = [factory.build(p) for p in split(parts[1].strip("{}"))]
            conditional.append(RooCmdArg("Conditional", [pdf], given, True))
        else:
            plain.append(pdf)
    return RooProdPdf(name, name, plain, *conditional)


def _formula(kind: str) -> Callable[[Any, str, list[str]], Any]:
    def make(factory: Any, name: str, args: list[str]) -> Any:
        from .functions import RooFormulaVar
        from .pdfs.generic import RooGenericPdf

        expression = factory.build(args[0])
        variables = []
        for arg in args[1:]:
            found = factory.build(arg)
            variables.extend(list(found) if hasattr(found, "_list") else [found])
        cls = RooGenericPdf if kind == "pdf" else RooFormulaVar
        return cls(name, name, str(expression), variables)

    return make


def _values(kind: str) -> Callable[[Any, str, list[str]], Any]:
    def make(factory: Any, name: str, args: list[str]) -> Any:
        from .functions import RooAddition, RooProduct

        items = [factory.build(arg) for arg in args]
        return (RooProduct if kind == "prod" else RooAddition)(name, name, items)

    return make


def _simul(factory: Any, name: str, args: list[str]) -> Any:
    """``SIMUL::s(cat, A=pdfA, B=pdfB)``: a simultaneous density over a category's states."""
    from .pdfs.simultaneous import RooSimultaneous

    index = factory.build(args[0])
    made = RooSimultaneous(name, name, index)
    for arg in args[1:]:
        label, _, spec = arg.partition("=")
        made.addPdf(factory.build(spec), label.strip())
    return made


def _refused(kind: str) -> Callable[[Any, str, list[str]], Any]:
    def make(factory: Any, name: str, args: list[str]) -> Any:
        raise UnsupportedFeatureError(
            f"the factory's {kind} operator is not here yet: build the object with its class "
            "and import it instead"
        )

    return make


#: The operators by name.
OPERATORS: dict[str, Callable[[Any, str, list[str]], Any]] = {
    "SUM": _sum, "PROD": _prod, "EXPR": _formula("pdf"), "expr": _formula("function"),
    "GENERIC": _formula("pdf"), "sum": _values("sum"), "prod": _values("prod"), "SIMUL": _simul,
    **{kind: _refused(kind) for kind in ("FCONV", "NCONV", "SIMCLONE", "EDIT", "CEXPR", "cexpr",
                                         "taylorexpand", "int", "deriv", "cdf", "PROJ", "nconv",
                                         "lagrangianmorph", "dataobs")},
}  # fmt: skip
