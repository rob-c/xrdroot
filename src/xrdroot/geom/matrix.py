"""Where a volume sits in its mother: a rotation and a translation, as ``TGeoMatrix`` keeps them.

A placement takes a point of the daughter's own frame to the mother's:
``master = R @ local + t``, ``R`` a 3 by 3 rotation (``fRotationMatrix``,
row by row) and ``t`` the translation. Placements compose by multiplying,
the daughter's on the right, so a point deep in the tree reaches the world
through each mother's matrix in turn.

The ways ROOT has of spelling a rotation are all here: GEANT3's six angles
(each local axis's polar and azimuthal angle in the mother), Euler's three
(``phi`` about z, ``theta`` about the new x, ``psi`` about the new z) and
turns about the mother's own axes, each in degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Matrix", "IDENTITY"]

#: Degrees to radians, as ``TMath::Pi() / 180`` is.
DEGRAD = math.pi / 180.0
#: How near to 0 or to 1 an element of a GEANT3 rotation is snapped to it.
SNAP = 1e-15
#: How near to a pole (theta 0 or 180) ``GetAngles`` takes a rotation to be at one.
EULER_POLE = 1e-9


def _snapped(value: float) -> float:
    """GEANT3 angles' rounding trick: an element within ``SNAP`` of 0, 1 or -1 made it."""
    for exact in (0.0, 1.0, -1.0):
        if abs(value - exact) < SNAP:
            return exact
    return value


class Matrix:
    """A rotation ``rotation`` (3 by 3) then a translation ``translation`` (3)."""

    def __init__(self, rotation: Any = None, translation: Sequence[float] = (0.0, 0.0, 0.0)):
        self.rotation = np.eye(3) if rotation is None else np.array(rotation, dtype=np.float64)
        self.translation = np.array(translation, dtype=np.float64)

    @classmethod
    def geant(cls, *angles: float) -> Matrix:
        """GEANT3's ``theta1, phi1, theta2, phi2, theta3, phi3``: each axis's direction."""
        columns = []
        for theta, phi in zip(angles[0::2], angles[1::2], strict=False):
            t, p = DEGRAD * theta, DEGRAD * phi
            columns.append([math.cos(p) * math.sin(t), math.sin(p) * math.sin(t), math.cos(t)])
        rotation = np.array([[_snapped(v) for v in row] for row in np.transpose(columns)])
        return cls(rotation)

    @classmethod
    def euler(cls, phi: float, theta: float, psi: float) -> Matrix:
        """Euler's angles: ``phi`` about z, then ``theta`` about the new x, ``psi`` about z."""
        sf, cf = math.sin(DEGRAD * phi), math.cos(DEGRAD * phi)
        st, ct = math.sin(DEGRAD * theta), math.cos(DEGRAD * theta)
        sp, cp = math.sin(DEGRAD * psi), math.cos(DEGRAD * psi)
        return cls([[cp * cf - ct * sf * sp, -sp * cf - ct * sf * cp, st * sf],
                    [cp * sf + ct * cf * sp, -sp * sf + ct * cf * cp, -st * cf],
                    [sp * st, cp * st, ct]])  # fmt: skip

    def rotated(self, axis: int, angle: float) -> Matrix:
        """Turned by ``angle`` degrees about the mother's axis ``axis`` (0, 1, 2 for x, y, z)."""
        c, s = math.cos(DEGRAD * angle), math.sin(DEGRAD * angle)
        turn = np.eye(3)
        a, b = [(1, 2), (2, 0), (0, 1)][axis]
        turn[a, a], turn[a, b], turn[b, a], turn[b, b] = c, -s, s, c
        return Matrix(turn @ self.rotation, turn @ self.translation)

    def reflected(self, axis: int) -> Matrix:
        """Mirrored in the plane through the origin across the mother's axis ``axis``."""
        mirror = np.eye(3)
        mirror[axis, axis] = -1.0
        return Matrix(mirror @ self.rotation, mirror @ self.translation)

    def __matmul__(self, other: Matrix) -> Matrix:
        """This placement after ``other``: a point of ``other``'s frame taken through both."""
        return Matrix(
            self.rotation @ other.rotation, self.rotation @ other.translation + self.translation
        )

    def inverse(self) -> Matrix:
        rotation = self.rotation.T
        return Matrix(rotation, -(rotation @ self.translation))

    def to_master(self, points: Any) -> np.ndarray[Any, Any]:
        """Points (N by 3) of the local frame, in the mother's."""
        return np.asarray(points, dtype=np.float64) @ self.rotation.T + self.translation

    def is_identity(self) -> bool:
        return bool(np.array_equal(self.rotation, np.eye(3)) and not self.translation.any())

    def is_rotation(self) -> bool:
        return not np.array_equal(self.rotation, np.eye(3))

    def is_reflection(self) -> bool:
        return bool(np.linalg.det(self.rotation) < 0)

    def geant_angles(self) -> tuple[float, ...]:
        """The six GEANT3 angles back, in degrees: each local axis's theta and phi (0..360)."""
        found: list[float] = []
        for column in self.rotation.T:
            theta = math.degrees(math.acos(max(-1.0, min(1.0, column[2]))))
            flat = abs(column[0]) < 1e-6 and abs(column[1]) < 1e-6
            phi = 0.0 if flat else math.degrees(math.atan2(column[1], column[0]))
            found += [theta, phi + 360.0 if phi < 0 else phi]
        return tuple(found)

    def euler_angles(self) -> tuple[float, float, float]:
        """Euler's ``phi, theta, psi`` back, in degrees, as ``TGeoRotation::GetAngles`` finds them.

        With theta 0 or 180 only ``phi + psi`` is fixed, and ``psi`` is taken as 0.
        """
        m = self.rotation.reshape(9)
        if abs(1.0 - abs(m[8])) < EULER_POLE:
            return math.degrees(math.atan2(-m[8] * m[1], m[0])), math.degrees(math.acos(m[8])), 0.0
        phi = math.atan2(m[2], -m[5])
        sphi = math.sin(phi)
        pole = abs(sphi) < EULER_POLE
        theta = -math.asin(m[5] / math.cos(phi)) if pole else math.asin(m[2] / sphi)
        return math.degrees(phi), math.degrees(theta), math.degrees(math.atan2(m[6], m[7]))

#: The placement that leaves a daughter where its mother's origin is: ``gGeoIdentity``.
IDENTITY = Matrix()
