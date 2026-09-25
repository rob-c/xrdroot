"""ROOT's ``RDataFrame``, evaluated lazily, in one pass, a whole batch of entries at a time.

    >>> from xrdroot import RDataFrame
    >>> df = RDataFrame("Events", "events.root")                    # doctest: +SKIP
    >>> h = (df.Filter("nMuon == 2", "two muons")
    ...        .Define("m", "InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)")
    ...        .Histo1D(("m", "", 300, 0.25, 300), "m"))           # doctest: +SKIP
    >>> print(df.Report().GetValue())                                # doctest: +SKIP

The interface is ROOT's - ``Define``, ``Filter``, ``Histo1D``, ``Report``,
``Snapshot`` and the rest, with results that are computed the first time
one is asked for, all of them in one event loop - and the expressions are
C++'s, with ``ROOT::VecOps`` in them. What is different is what runs: an
expression, or a Python callable, is evaluated over a whole batch of entries
at once, with NumPy, rather than once per entry, and the loop can be shared
across worker processes with results that are the same to the last bit.

:mod:`xrdroot.rdf.vecops` is ``ROOT::VecOps`` for the callables.
"""

from __future__ import annotations

from . import vecops
from .frame import (
    DisableImplicitMT,
    EnableImplicitMT,
    GetThreadPoolSize,
    IsImplicitMTEnabled,
    RDataFrame,
    Result,
    RNode,
    RunGraphs,
)
from .report import CutFlowReport, CutInfo, Display
from .sources import SampleInfo

__all__ = [
    "RDataFrame",
    "RNode",
    "Result",
    "RunGraphs",
    "EnableImplicitMT",
    "DisableImplicitMT",
    "IsImplicitMTEnabled",
    "GetThreadPoolSize",
    "CutFlowReport",
    "CutInfo",
    "Display",
    "SampleInfo",
    "vecops",
]
