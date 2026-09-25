"""``ROOT.ROOT``: C++'s ``ROOT::`` namespace - implicit multithreading, and everything else.

``ROOT::EnableImplicitMT()`` is spelt ``ROOT.ROOT.EnableImplicitMT()`` from
Python, and ``ROOT::RDataFrame`` ``ROOT.ROOT.RDataFrame``; the namespace
here has the threading switches - which are xrdroot's worker processes -
and hands any other name to the top of the namespace, where the class of
that name is if the kit has it.
"""

from __future__ import annotations

import sys
from typing import Any

from ...rdf.frame import DisableImplicitMT, EnableImplicitMT, GetThreadPoolSize, IsImplicitMTEnabled

__all__ = [
    "ROOT",
    "EnableImplicitMT",
    "DisableImplicitMT",
    "IsImplicitMTEnabled",
    "GetThreadPoolSize",
]


class _Namespace:
    """``ROOT::``: its own few functions, then whatever the namespace's top has by that name."""

    EnableImplicitMT = staticmethod(EnableImplicitMT)
    DisableImplicitMT = staticmethod(DisableImplicitMT)
    IsImplicitMTEnabled = staticmethod(IsImplicitMTEnabled)
    GetThreadPoolSize = staticmethod(GetThreadPoolSize)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(sys.modules["xrdroot.pyroot"], name)

    def __repr__(self) -> str:
        return "<namespace ROOT>"


#: ``ROOT.ROOT``.
ROOT = _Namespace()
