"""TMVA under ``import ROOT``: ``ROOT.TMVA.Factory``, ``ROOT.TMVA.Types.kBDT``, ``ROOT.TMVA.Reader``.

``ROOT.TMVA`` is a namespace of :mod:`xrdroot.tmva`'s classes by TMVA's
names - the engine speaks TMVA's API already, so this only gathers it, as
``TMVA::Experimental`` gathers the tensor, reader and scaler classes. A
class of TMVA's this does not have is refused by that name.
"""

from __future__ import annotations

from typing import Any

from .. import tmva as _tmva

__all__ = ["TMVA"]


class _Namespace:
    """A C++ namespace: its members as attributes, and a refusal naming any it has not."""

    def __init__(self, name: str, members: dict[str, Any]) -> None:
        self._name = name
        self.__dict__.update(members)

    def __getattr__(self, attribute: str) -> Any:
        if attribute.startswith("__"):
            raise AttributeError(attribute)
        raise AttributeError(f"ROOT has {self._name}::{attribute}; xrdroot.pyroot does not yet")

    def __repr__(self) -> str:
        return f"<namespace {self._name}>"


def _gui(*_: Any) -> None:
    """``TMVA::TMVAGui``: a window, which batch mode never opens."""


TMVA = _Namespace(
    "TMVA",
    {
        **{name: getattr(_tmva, name) for name in _tmva.__all__},
        "TMVAGui": _gui,
        "Experimental": _Namespace("TMVA::Experimental", dict(_tmva.EXPERIMENTAL)),
    },
)
