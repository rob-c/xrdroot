"""How a geometry drawn in a pad is seen: ``TView3D``'s perspective, as ``TGeoPainter`` sets it.

Drawing a volume in a pad makes a ``TView3D`` of kind 11, a perspective
view from the pad's angles (``fLongitude = -90 - phi``, ``fLatitude = 90 -
theta``, 30 degrees each by default) whose range is first found by
painting everything once (``Paint("range")``) and taking the extent of
every point. The eye is then three extents from the centre of that range,
the screen half an extent in front of it (``SetDefaultWindow``), and a point
is taken to the pad's ``-1..1`` by ``DefinePerspectiveView``'s matrix and a
division by its depth (``WCtoNDC``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["PerspectiveView"]

#: Degrees to radians, as ``TView3D``'s ``kRad`` is.
RAD = math.pi / 180.0
#: How much further than the depth ``WCtoNDC`` pushes a point behind the eye.
BEHIND = 1000.0


class PerspectiveView:
    """``TView3D`` in perspective over the box ``rmin..rmax``, for a pad ``aspect`` high by wide."""

    def __init__(
        self,
        rmin: Sequence[float],
        rmax: Sequence[float],
        longitude: float = -120.0,
        latitude: float = 60.0,
        psi: float = 0.0,
        aspect: float = 1.0,
    ) -> None:
        self.rmin = np.asarray(rmin, dtype=np.float64)
        self.rmax = np.asarray(rmax, dtype=np.float64)
        self.longitude, self.latitude, self.psi = longitude, latitude, psi
        extent = float(np.sqrt(np.sum((0.5 * (self.rmax - self.rmin)) ** 2))) or 1.0
        self.dview, self.dproj = 3 * extent, 0.5 * extent
        du = 0.5 * self.dproj
        self.window = (du, du * aspect)
        self.tnorm = self._matrix()

    @classmethod
    def from_pad(cls, rmin: Any, rmax: Any, theta: float, phi: float,
                 aspect: float) -> PerspectiveView:  # fmt: skip
        """The view a pad's ``theta`` and ``phi`` make, as ``TView3D``'s constructor sets it."""
        return cls(rmin, rmax, -90.0 - phi, 90.0 - theta, 0.0, aspect)

    def _rotation(self) -> np.ndarray[Any, Any]:
        """``DefinePerspectiveView``'s ``t12``: the turn to the eye, as rows ``x, y, z``."""
        c1, s1 = math.cos(self.psi * RAD), math.sin(self.psi * RAD)
        c2, s2 = math.cos(self.latitude * RAD), math.sin(self.latitude * RAD)
        s3, c3 = math.cos(self.longitude * RAD), -math.sin(self.longitude * RAD)
        return np.array([[c1 * c3 - s1 * c2 * s3, c1 * s3 + s1 * c2 * c3, s1 * s2],
                         [-s1 * c3 - c1 * c2 * s3, -s1 * s3 + c1 * c2 * c3, c1 * s2],
                         [s2 * s3, -s2 * c3, c2]])  # fmt: skip

    def _matrix(self) -> np.ndarray[Any, Any]:
        """World to screen: rows ``u, v, depth`` and a column for the shift (3 by 4)."""
        turn = self._rotation()
        centre = 0.5 * (self.rmax + self.rmin)
        shift = -(turn @ centre)
        shift[2] -= self.dview
        turn[2] *= -1.0
        shift[2] *= -1.0
        du, dv = self.window
        scale = np.array([1.0 / du, 1.0 / dv, 1.0 / self.dproj])
        return np.column_stack([turn, shift]) * scale[:, None]

    def project(self, points: Any) -> np.ndarray[Any, Any]:
        """World points (N by 3) on the pad, in its ``-1..1`` range (N by 2)."""
        world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        screen = world @ self.tnorm[:, :3].T + self.tnorm[:, 3]
        depth = screen[:, 2:3]
        ahead = depth > 0
        safe = np.where(ahead, depth, 1.0)
        return np.where(ahead, screen[:, :2] / safe, screen[:, :2] * BEHIND)
