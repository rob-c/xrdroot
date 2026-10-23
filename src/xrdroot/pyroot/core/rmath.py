"""``ROOT.Math``: the MathCore namespace - distributions, special functions and vectors.

Densities are ``*_pdf``, lower tails ``*_cdf``, upper tails ``*_cdf_c`` and
their inverses ``*_quantile`` and ``*_quantile_c``, with ROOT's argument
orders: the variable first, then the shape, then the location ``x0``. The
vectors are those of :mod:`.genvector`, and ``VectorUtil`` beside them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ...fit.defaults import DEFAULTS, minimizer_algo, minimizer_type, set_minimizer
from ...function.rootmath import *  # noqa: F403 - scalar ROOT::Math compatibility exports
from ...function.rootmath import __all__ as _scalars
from .fitdata import WrappedMultiTF1  # noqa: F401 - ROOT::Math's, by name
from .genvector import *  # noqa: F403
from .genvector import __all__ as _vectors
from .legendres import (  # noqa: F401 - ROOT::Math's Legendre polynomials, by name
    assoc_legendre,
    legendre,
    sph_legendre,
)
from .lfsr import LFSR  # noqa: F401 - ROOT::Math::LFSR, by name
from .mathtools import (  # noqa: F401 - ROOT::Math's function objects and tools, by name
    Factory,
    Functor,
    Functor1D,
    GradFunctor,
    GradFunctor1D,
    Integrator,
    IntegratorOneDim,
    Minimizer,
    RootFinder,
)


class MinimizerOptions:
    """``ROOT::Math::MinimizerOptions``: the minimiser a fit uses, which is Minuit2's MIGRAD here.

    The defaults can be set and read back; fits run through iminuit whatever
    is set, so a script choosing ``"Minuit"`` or ``"Minuit2"`` gets the same.
    """

    #: The table the engine reads too: RooStats' calculators minimise with these.
    _defaults: ClassVar[dict[str, Any]] = DEFAULTS

    @classmethod
    def SetDefaultMinimizer(cls, name: Any, algorithm: Any = None) -> None:
        set_minimizer(name, algorithm)

    @classmethod
    def DefaultMinimizerType(cls) -> str:
        return minimizer_type()

    @classmethod
    def DefaultMinimizerAlgo(cls) -> str:
        return minimizer_algo()


class MathMoreLibrary:
    """``ROOT::Math::MathMoreLibrary``: what loads MathMore, whose functions - the quantiles,
    the special functions - are here already."""

    @staticmethod
    def Load() -> None:
        """``Load()``: nothing to load."""


def _default_accessors(key: str) -> None:
    setattr(MinimizerOptions, f"SetDefault{key}",
            classmethod(lambda cls, value: cls._defaults.__setitem__(key, value)))  # fmt: skip
    setattr(MinimizerOptions, f"Default{key}", classmethod(lambda cls: cls._defaults[key]))


for _key in ("Tolerance", "Precision", "PrintLevel", "MaxFunctionCalls", "MaxIterations",
             "Strategy", "ErrorDef"):  # fmt: skip
    _default_accessors(_key)


def _exported(name: str, value: object) -> bool:
    return (
        callable(value)
        and not name.startswith("_")
        and getattr(value, "__module__", "") == __name__
    )


__all__ = sorted(
    [name for name, value in list(globals().items()) if _exported(name, value)]
    + list(_scalars)
    + list(_vectors)
    + [
        "Factory",
        "Minimizer",
        "Functor",
        "Functor1D",
        "GradFunctor",
        "GradFunctor1D",
        "Integrator",
        "IntegratorOneDim",
        "LFSR",
        "WrappedMultiTF1",
        "RootFinder",
    ]
)
