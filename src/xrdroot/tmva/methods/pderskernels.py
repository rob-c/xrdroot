"""The kernels ``PDERS`` weighs the events in a box by, of their normalised distance.

The distance is 0 at the event and 1 at the box's inscribed ellipsoid;
TMVA's ``KernelNormalization`` is always 1 (its cached factor starts at 1
and is never worked out), so the kernels are used as they are. ``Trim``,
which needs the ``kNN`` volume mode's farthest distance, xrdroot does not
have.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["KERNELS", "kernel_values"]

#: The kernels, by the names ``KernelEstimator`` takes.
KERNELS = (
    "Box",
    "Sphere",
    "Teepee",
    "Gauss",
    "Sinc3",
    "Sinc5",
    "Sinc7",
    "Sinc9",
    "Sinc11",
    "Lanczos2",
    "Lanczos3",
    "Lanczos5",
    "Lanczos8",
)


def _power(value: Any, nvar: int) -> Any:
    """``sinc^n``, kept positive for an even ``n`` as TMVA keeps it."""
    if nvar % 2:
        return value**nvar
    return np.abs(value) * value ** (nvar - 1)


def _sinc(x: Any) -> Any:
    tiny = np.abs(x) < 10e-10
    pix = np.pi * np.where(tiny, 1.0, x)
    return np.where(tiny, 1.0, np.sin(pix) / pix)


def kernel_values(name: str, distance: Any, sigma: float, nvar: int) -> Any:
    """``ApplyKernelFunction`` of every distance."""
    if name in ("Box", "Sphere"):
        return np.ones_like(distance)
    if name == "Teepee":
        return 1.0 - distance
    if name == "Gauss":
        return np.exp(-0.5 * (distance / sigma) ** 2)
    if name.startswith("Sinc"):
        crossings = 2 + (int(name[4:]) - 3) // 2
        return np.where(np.abs(distance) < 10e-10, 1.0, _power(_sinc(crossings * distance), nvar))
    level = int(name[7:])
    tiny = np.abs(distance) < 10e-10
    value = _sinc(distance) * _sinc(level * distance)
    return np.where(tiny, 1.0, _power(value, nvar))
