"""The distributed backends, each a package with its ``RDataFrame``: Dask, and Spark refused.

:func:`build_backends_submodules` puts a module for each - ``Dask``,
``Spark`` - in ``ROOT.RDF.Distributed``, holding the backend's ``RDataFrame``,
without importing the scheduler it needs until a frame is made.
"""

from __future__ import annotations

import types

from . import Dask, Spark

__all__ = ["BACKENDS", "build_backends_submodules"]

#: The backends, by the name of their package here and in ``ROOT.RDF.Distributed``.
BACKENDS = {"Dask": Dask, "Spark": Spark}


def build_backends_submodules(parentmodule: types.ModuleType) -> types.ModuleType:
    """``parentmodule`` with a module for each backend, holding its ``RDataFrame``."""
    for name, actual in BACKENDS.items():
        made = types.ModuleType(f"ROOT.RDF.Distributed.{name}")
        made.__file__ = "<module ROOT.RDF.Distributed>"
        made.__path__ = []
        made.__package__ = parentmodule.__name__
        made.RDataFrame = actual.RDataFrame  # type: ignore[attr-defined]
        setattr(parentmodule, name, made)
    return parentmodule
