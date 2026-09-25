"""``ROOT.RDataFrame``, ``ROOT.RDF``, ``ROOT.RVec`` and ``ROOT.VecOps``, for PyROOT scripts.

See :mod:`.frame` for the frame and its results, and :mod:`.rvec` for
``RVec`` and ``VecOps``. ``EnableImplicitMT`` and its kin are xrdroot's own,
which share an event loop across worker processes.
"""

from __future__ import annotations

from ...rdf import DisableImplicitMT, EnableImplicitMT, GetThreadPoolSize, IsImplicitMTEnabled
from .frame import RDF, RDataFrame
from .rvec import RVec, RVecB, RVecD, RVecF, RVecI, RVecL, VecOps

__all__ = [
    "RDataFrame",
    "RDF",
    "RVec",
    "RVecF",
    "RVecD",
    "RVecI",
    "RVecL",
    "RVecB",
    "VecOps",
    "EnableImplicitMT",
    "DisableImplicitMT",
    "IsImplicitMTEnabled",
    "GetThreadPoolSize",
]
