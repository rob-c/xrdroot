"""``TView3D``: how a lego or surface plot's three dimensions are laid flat.

ROOT looks at a histogram drawn ``LEGO`` or ``SURF`` from the pad's angles
(``fTheta`` and ``fPhi``, 30 degrees each unless set) with a parallel
projection: the box ``fRmin`` to ``fRmax`` is centred, scaled by
``sqrt(3)/2`` of each of its sides so that it fits a unit sphere, and turned
to face the eye - ``TView3D::DefineViewDirection``'s matrices. The pad's
range is then set so that the box's shadow fills the frame
(``TView3D::PadRange``), and a point of the box is a point of the pad
(``WCtoNDC``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

__all__ = ["View3D"]

#: Degrees to radians, as ``TView3D`` spells it.
RAD = math.atan(1.0) * 4.0 / 180.0


class Corners(NamedTuple):
    """``TView3D::AxisVertex``: the box's eight corners in drawing order, and each axis's ends."""

    vertices: tuple[tuple[float, float, float], ...]
    x: tuple[int, int]
    y: tuple[int, int]
    z: tuple[int, int]


def _direction(scale: list[float], centre: list[float], angles: tuple[float, float, float, float, float, float]) -> list[float]:
    """``DefineViewDirection``: the matrix taking the box's points to the normalised view."""
    cosphi, sinphi, costhe, sinthe, cospsi, sinpsi = angles
    tran = [1 / scale[0], 0, 0, -centre[0] / scale[0],
            0, 1 / scale[1], 0, -centre[1] / scale[1],
            0, 0, 1 / scale[2], -centre[2] / scale[2],
            0, 0, 0, 1]  # fmt: skip
    c1, s1, c2, s2, c3, s3 = cospsi, sinpsi, costhe, sinthe, -sinphi, cosphi
    rota = [c1 * c3 - s1 * c2 * s3, c1 * s3 + s1 * c2 * c3, s1 * s2, 0,
            -s1 * c3 - c1 * c2 * s3, -s1 * s3 + c1 * c2 * c3, c1 * s2, 0,
            s2 * s3, -s2 * c3, c2, 0]  # fmt: skip
    made = [0.0] * 16
    for row in range(3):
        for column in range(4):
            made[4 * row + column] = sum(rota[4 * row + j] * tran[4 * j + column] for j in range(4))
    return made


class View3D:
    """A parallel view of the box ``rmin`` to ``rmax`` from ``longitude`` and ``latitude``."""

    def __init__(self, rmin: tuple[float, float, float], rmax: tuple[float, float, float],
                 longitude: float, latitude: float, psi: float = 0.0) -> None:  # fmt: skip
        self.rmin, self.rmax = list(rmin), list(rmax)
        half = 0.5 * math.sqrt(3.0)
        scale = [half * (high - low) for low, high in zip(self.rmin, self.rmax)]
        centre = [0.5 * (high + low) for low, high in zip(self.rmin, self.rmax)]
        c1, s1 = math.cos(longitude * RAD), math.sin(longitude * RAD)
        c2, s2 = math.cos(latitude * RAD), math.sin(latitude * RAD)
        c3, s3 = math.cos(psi * RAD), math.sin(psi * RAD)
        #: ``fTnorm``: the view with its screen turned by ``psi``; ``fTN`` without.
        self.tnorm = _direction(scale, centre, (c1, s1, c2, s2, c3, s3))
        self.tn = _direction(scale, centre, (c1, s1, c2, s2, 1.0, 0.0))

    def to_ndc(self, point: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        """``WCtoNDC``: a point of the box as the pad's coordinates, and its depth."""
        x, y, z = point
        t = self.tnorm
        return (t[0] * x + t[1] * y + t[2] * z + t[3], t[4] * x + t[5] * y + t[6] * z + t[7],
                t[8] * x + t[9] * y + t[10] * z + t[11])  # fmt: skip

    def screen(self, point: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
        """A point through ``fTN``, which the moving screen hides lines by."""
        x, y, z = point
        t = self.tn
        return (t[0] * x + t[1] * y + t[2] * z + t[3], t[4] * x + t[5] * y + t[6] * z + t[7],
                t[8] * x + t[9] * y + t[10] * z + t[11])  # fmt: skip

    def normal(self, x: float, y: float, z: float) -> float:
        """``FindNormal``: which way a face with normal ``(x, y, z)`` faces, by its sign."""
        t = self.tn
        return (x * (t[1] * t[6] - t[2] * t[5]) + y * (t[2] * t[4] - t[0] * t[6])
                + z * (t[0] * t[5] - t[1] * t[4]))  # fmt: skip

    def corners(self) -> Corners:
        """``AxisVertex``: the box's corners, in the order the painters walk them, and the axes' ends."""
        lo, hi = self.rmin, self.rmax
        p = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
             (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]  # fmt: skip
        nodes = ((2, 3, 4, 1, 6, 7, 8, 5), (3, 4, 1, 2, 7, 8, 5, 6), (1, 2, 3, 4, 5, 6, 7, 8), (4, 1, 2, 3, 8, 5, 6, 7))
        ends = ((3, 2, 1, 2), (2, 1, 3, 2), (1, 2, 2, 3), (2, 3, 2, 1), (4, 1, 4, 3), (3, 4, 4, 1),
                (4, 3, 1, 4), (1, 4, 3, 4), (8, 5, 8, 7), (7, 8, 8, 5), (8, 7, 5, 8), (5, 8, 7, 8),
                (7, 6, 5, 6), (6, 5, 7, 6), (5, 6, 6, 7), (6, 7, 6, 5))  # fmt: skip
        case = (1 if self.tnorm[8] <= 0 else 0) + (2 if self.tnorm[9] <= 0 else 0)
        vertices = tuple(p[k - 1] for k in nodes[case])
        case += (4 if self.tnorm[10] < 0 else 0) + (8 if self.tnorm[6] < 0 else 0)
        ix1, ix2, iy1, iy2 = ends[case]
        return Corners(vertices, (ix1, ix2), (iy1, iy2), (1, 5) if case < 8 else (3, 7))

    def extent(self) -> tuple[float, float]:
        """How far the box's shadow reaches either side of the middle, across and up."""
        reach = []
        for row in range(2):
            total = self.tnorm[4 * row + 3]
            for k in range(3):
                weight = self.tnorm[4 * row + k]
                total += weight * (self.rmin[k] if weight < 0 else self.rmax[k])
            reach.append(total)
        return reach[0], reach[1]

    def pad_range(self, margins: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """``PadRange``: the pad's ``x1, y1, x2, y2`` that put the box's shadow in the frame."""
        left, right, bottom, top = margins
        across, up = self.extent()
        wide = 2 * across / (1 - left - right)
        high = 2 * up / (1 - bottom - top)
        return -across - wide * left, -up - high * bottom, across + wide * right, up + high * top
