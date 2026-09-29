"""RooStats under ``import ROOT``: ``ROOT.RooStats.ModelConfig`` and the calculators.

``ROOT.RooStats`` is a namespace of :mod:`xrdroot.roostats`' classes and
functions by RooStats' names - the engine speaks RooStats' API already, so
this only gathers them (:mod:`xrdroot.roostats.registry`), with
``RooStats::HistFactory`` inside it as ROOT nests it. A name of RooStats'
that is not here is refused by that name.
"""

from __future__ import annotations

from typing import Any

from ...roostats.registry import members
from . import plots, sdplot

__all__ = ["RooStats"]


class _Namespace:
    """A C++ namespace: its members as attributes, and a refusal naming any it has not."""

    def __init__(self, name: str, content: dict[str, Any]) -> None:
        self._name = name
        self.__dict__.update(content)

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        raise AttributeError(f"ROOT has {self._name}::{attribute}; xrdroot.pyroot does not yet")

    def __repr__(self) -> str:
        return f"<namespace {self._name}>"


def _histfactory() -> dict[str, Any]:
    """``RooStats::HistFactory``'s classes and functions."""
    from ...histfactory import channel, factory, make, measurement, model, shapes, systematics
    from ...roofit.pdfs.histfactory import FlexibleInterpVar

    found: dict[str, Any] = {"FlexibleInterpVar": FlexibleInterpVar}
    for module in (systematics, shapes, model, channel, measurement, factory, make):
        found.update({name: getattr(module, name) for name in module.__all__})
    return found


#: ``ROOT.RooStats``.
RooStats = _Namespace(
    "RooStats",
    {
        **members(),
        **{name: getattr(module, name) for module in (plots, sdplot) for name in module.__all__},
        "HistFactory": _Namespace("RooStats::HistFactory", _histfactory()),
    },
)
