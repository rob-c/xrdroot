"""Interpreter cmath keeps shared numerics and installs the ROOT vector constructor."""

from __future__ import annotations

from typing import Any

from ...cppstd import cmath as _cmath
from ...cppstd.cmath import *  # noqa: F403
from ...cppstd.cmath import INF as INF
from ...cppstd.cmath import NAN as NAN
from ...cppstd.cmath import NUMBERS as NUMBERS
from ...cppstd.cmath import __all__ as __all__
from ...cppstd.cmath import _c as _c
from ...cppstd.cmath import _elementwise as _elementwise
from ...cppstd.cmath import _logarithm as _logarithm
from ...cppstd.cmath import _numbers as _numbers
from ...cppstd.cmath import _promoted as _promoted


def _rvec(found: Any) -> Any:
    """Construct the same ROOT vector only when an array operation needs its result."""
    from ...pyroot.rdf.rvec import RVec

    return RVec[found.dtype](found)


_cmath.set_vector_result(_rvec)
