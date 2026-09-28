"""RooFit's compute kernels: how a likelihood evaluates a density at its events.

A fit in ROOT 6.40 evaluates the density at the dataset's events through
``RooBatchCompute``, a compiled library of kernels, one per density - not
through the density's ``evaluate()``, which is what ``getVal`` and a plot
call. The kernels are the same formulas rounded their own way: the Gaussian
kernel squares ``x - mean`` and multiplies by ``-0.5 / sigma^2`` where
``evaluate()`` divides, and the Gaussian and exponential kernels take the
exponential with VDT's ``fast_exp`` (ROOT is built with VDT) rather than the
C library's ``exp``. The two agree to a bit or two; a likelihood summed over
thousands of events is a different number in its last places, and MIGRAD's
steps, the last digits of a fit, follow those places.

So a likelihood evaluates its events inside :func:`likelihood`, and the
densities with a kernel ask :func:`active` which formula to use. A
normalisation integral a likelihood needs is RooFit's ``RooRealIntegral``,
which evaluates the density one point at a time: numerical integration
steps back out with :func:`scalar`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np

__all__ = ["active", "likelihood", "scalar", "fast_exp"]

#: Whether densities are being evaluated as RooBatchCompute's kernels evaluate them.
_ACTIVE: ContextVar[bool] = ContextVar("xrdroot_roofit_kernels", default=False)


def active() -> bool:
    """Whether a likelihood is evaluating its events now, through RooFit's kernels."""
    return _ACTIVE.get()


@contextmanager
def _set(flag: bool) -> Iterator[None]:
    token = _ACTIVE.set(flag)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def likelihood() -> Any:
    """Evaluate densities as a likelihood's kernels do, until the block ends."""
    return _set(True)


def scalar() -> Any:
    """Evaluate densities as ``evaluate()`` does, until the block ends - inside an integral."""
    return _set(False)


# --- VDT's fast_exp -------------------------------------------------------------

#: ``vdt/exp.h``'s constants: the Cephes rational approximation of ``e^x`` on
#: ``[-ln2/2, ln2/2]``, the two parts of ``ln 2`` the argument is reduced by,
#: and the argument beyond which the answer is infinity or zero.
_P = (1.26177193074810590878e-4, 3.02994407707441961300e-2, 9.99999999999999999910e-1)
_Q = (
    3.00198505138664455042e-6,
    2.52448340349684104192e-3,
    2.27265548208155028766e-1,
    2.00000000000000000009e0,
)
_LOG2E = 1.4426950408889634073599
_LN2_HIGH, _LN2_LOW = 6.93145751953125e-1, 1.42860682030941723212e-6
_LIMIT = 708.0


def fast_exp(values: Any) -> Any:
    """``vdt::fast_exp``, operation for operation, on an array or a number.

    VDT's floor is its own - the argument truncated, less one if its sign
    bit is set - so a negative integer rounds down one further than
    ``floor`` would, and that is kept: it only moves which power of two the
    reduced argument is paired with, and it is VDT's answer.
    """
    x0 = np.asarray(values, dtype=np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = _LOG2E * x0 + 0.5
        whole = np.trunc(scaled) - np.signbit(scaled)
        n = np.where(np.isfinite(whole) & (np.abs(whole) < 2000.0), whole, 0.0).astype(np.int64)
        x = x0 - whole * _LN2_HIGH
        x = x - whole * _LN2_LOW
        xx = x * x
        p = (((_P[0] * xx) + _P[1]) * xx + _P[2]) * x
        q = (((_Q[0] * xx + _Q[1]) * xx + _Q[2]) * xx) + _Q[3]
        found = 1.0 + 2.0 * (p / (q - p))
        found = found * np.ldexp(1.0, n)
    found = np.where(x0 > _LIMIT, np.inf, np.where(x0 < -_LIMIT, 0.0, found))
    return found if found.ndim else float(found)
