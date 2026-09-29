"""``TGeoMatrix`` and its kinds: where a daughter volume sits in its mother.

Each is a :class:`~xrdroot.geom.Matrix` underneath (``._xrd``), a rotation
and a translation; the ROOT classes differ only in which of the two they
are made from and let be changed. A named matrix that is
``RegisterYourself``-ed is found by name, as a composite shape's
expression finds ``"A:t1"``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...geom import Matrix
from ..core.objects import TNamed

__all__ = [
    "TGeoMatrix", "TGeoIdentity", "TGeoTranslation", "TGeoRotation", "TGeoCombiTrans",
    "TGeoHMatrix", "TGeoScale", "gGeoIdentity",
]  # fmt: skip

#: The matrices registered, by name, for composite shapes' expressions.
REGISTERED: dict[str, TGeoMatrix] = {}


class TGeoMatrix(TNamed):
    """``TGeoMatrix``: a rotation and a translation, kept as an xrdroot matrix in ``_xrd``."""

    def __init__(self, name: Any = "", title: Any = "") -> None:
        super().__init__(name, title)
        self._xrd = Matrix()

    def _set(self, matrix: Matrix) -> None:
        self._xrd = matrix

    def GetTranslation(self) -> np.ndarray[Any, Any]:
        return self._xrd.translation

    def GetRotationMatrix(self) -> np.ndarray[Any, Any]:
        """The rotation, row by row, as ROOT's nine ``Double_t``."""
        return self._xrd.rotation.reshape(9)

    def IsIdentity(self) -> bool:
        return self._xrd.is_identity()

    def IsTranslation(self) -> bool:
        return bool(self._xrd.translation.any())

    def IsRotation(self) -> bool:
        return self._xrd.is_rotation()

    def IsReflection(self) -> bool:
        return self._xrd.is_reflection()

    def IsCombi(self) -> bool:
        return self.IsTranslation() and self.IsRotation()

    def RegisterYourself(self) -> None:
        """``RegisterYourself``: be found by name, as a composite shape's expression looks."""
        REGISTERED[self.GetName()] = self

    def LocalToMaster(self, local: Any, master: Any) -> None:
        master[:3] = self._xrd.to_master([list(local[:3])])[0]

    def MasterToLocal(self, master: Any, local: Any) -> None:
        local[:3] = self._xrd.inverse().to_master([list(master[:3])])[0]

    def Inverse(self) -> TGeoHMatrix:
        made = TGeoHMatrix()
        made._set(self._xrd.inverse())
        return made

    def _turned(self, axis: int, angle: float) -> None:
        self._set(self._xrd.rotated(axis, float(angle)))

    def RotateX(self, angle: float) -> None:
        self._turned(0, angle)

    def RotateY(self, angle: float) -> None:
        self._turned(1, angle)

    def RotateZ(self, angle: float) -> None:
        self._turned(2, angle)

    def ReflectX(self, leftside: bool = True, rotonly: bool = False) -> None:
        self._set(self._xrd.reflected(0))

    def ReflectY(self, leftside: bool = True, rotonly: bool = False) -> None:
        self._set(self._xrd.reflected(1))

    def ReflectZ(self, leftside: bool = True, rotonly: bool = False) -> None:
        self._set(self._xrd.reflected(2))

    def Print(self, option: str = "") -> None:
        """``Print``: what the matrix is, then its rotation's rows beside the translation."""
        rotation, translation = self._xrd.rotation, self._xrd.translation
        flags = [int(flag) for flag in (self.IsTranslation(), self.IsRotation(),
                                        self.IsReflection(), self.GetName() in REGISTERED)]
        print("matrix {} - tr={}  rot={}  refl={}  scl=0 shr=0 reg={} own=0".format(
            self.GetName(), *flags))  # fmt: skip
        for row, shift, axis in zip(rotation, translation, "xyz"):
            x, y, z = row
            print(f"{x:10.6f}{y:12.6f}{z:12.6f}    T{axis} = {shift:10.6f}")


class TGeoIdentity(TGeoMatrix):
    """``TGeoIdentity``: no rotation, no translation - ``gGeoIdentity``."""

    def __init__(self, name: Any = "Identity") -> None:
        super().__init__(name)


class TGeoTranslation(TGeoMatrix):
    """``TGeoTranslation(dx, dy, dz)`` or ``(name, dx, dy, dz)``: a shift, no turn."""

    def __init__(self, *args: Any) -> None:
        named = bool(args) and isinstance(args[0], str)
        super().__init__(args[0] if named else "")
        shift = args[1:] if named else args
        if len(shift) == 1 and isinstance(shift[0], TGeoMatrix):
            shift = tuple(shift[0].GetTranslation())
        self._set(Matrix(translation=[float(v) for v in (*shift, 0, 0, 0)[:3]]))

    def SetTranslation(self, dx: float, dy: float, dz: float) -> None:
        self._set(Matrix(self._xrd.rotation, [dx, dy, dz]))

    def SetDx(self, dx: float) -> None:
        self._xrd.translation[0] = float(dx)

    def SetDy(self, dy: float) -> None:
        self._xrd.translation[1] = float(dy)

    def SetDz(self, dz: float) -> None:
        self._xrd.translation[2] = float(dz)


def _rotation_of(args: tuple[Any, ...]) -> Matrix:
    """``TGeoRotation``'s angles: none, Euler's three, or GEANT3's six."""
    angles = [float(a) for a in args]
    if len(angles) == 3:
        return Matrix.euler(*angles)
    if len(angles) == 6:
        return Matrix.geant(*angles)
    return Matrix()


class TGeoRotation(TGeoMatrix):
    """``TGeoRotation``: a turn, made from Euler's angles, GEANT3's, a copy, or none."""

    def __init__(self, *args: Any) -> None:
        named = bool(args) and isinstance(args[0], str)
        super().__init__(args[0] if named else "")
        rest = args[1:] if named else args
        if rest and isinstance(rest[0], TGeoMatrix):
            self._set(Matrix(rest[0]._xrd.rotation))
            return
        self._set(_rotation_of(rest))

    def SetAngles(self, *angles: float) -> None:
        self._set(_rotation_of(angles))

    def GetAngles(self, *out: Any) -> tuple[float, ...]:
        """``GetAngles(phi, theta, psi)`` - Euler's - or GEANT3's six, into what is given."""
        from ..core.refs import store

        found = self._xrd.euler_angles() if len(out) == 3 else self._xrd.geant_angles()
        for target, value in zip(out, found):
            store(target, value)
        return tuple(found)

    def SetMatrix(self, rotation: Any) -> None:
        self._set(Matrix(np.asarray(rotation, dtype=np.float64).reshape(3, 3)))

    def MultiplyBy(self, other: TGeoMatrix, after: bool = True) -> None:
        mine, theirs = self._xrd, Matrix(other._xrd.rotation)
        self._set(theirs @ mine if after else mine @ theirs)


def _combined(rest: list[Any]) -> tuple[Matrix, Any]:
    """``TGeoCombiTrans``'s arguments as a placement, and the rotation it was given, if any.

    They are a copy of another matrix, a translation and a rotation, or
    ``dx, dy, dz`` and a rotation (or none).
    """
    if len(rest) == 1 and isinstance(rest[0], TGeoMatrix):
        return Matrix(rest[0]._xrd.rotation, rest[0]._xrd.translation), None
    if len(rest) == 2 and all(isinstance(one, TGeoMatrix) for one in rest):
        rest = [*rest[0].GetTranslation(), rest[1]]
    shift = [float(v) for v in [*rest[:3], 0.0, 0.0, 0.0][:3]]
    turn = rest[3] if len(rest) > 3 else None
    return Matrix(None if turn is None else turn._xrd.rotation, shift), turn


def _named(args: tuple[Any, ...]) -> tuple[str, list[Any]]:
    """A matrix constructor's name, if it was given one first, and the rest."""
    if args and isinstance(args[0], str):
        return args[0], list(args[1:])
    return "", list(args)


class TGeoCombiTrans(TGeoMatrix):
    """``TGeoCombiTrans([name,] dx, dy, dz, rot)``: a turn, then a shift."""

    def __init__(self, *args: Any) -> None:
        name, rest = _named(args)
        super().__init__(name)
        matrix, self._rotation = _combined(rest)
        self._set(matrix)

    def GetRotation(self) -> Any:
        return self._rotation

    def SetTranslation(self, dx: float, dy: float, dz: float) -> None:
        self._set(Matrix(self._xrd.rotation, [dx, dy, dz]))

    def SetRotation(self, rot: TGeoMatrix) -> None:
        self._rotation = rot
        self._set(Matrix(rot._xrd.rotation, self._xrd.translation))


class TGeoHMatrix(TGeoCombiTrans):
    """``TGeoHMatrix``: any placement, made by multiplying others."""

    def Multiply(self, right: TGeoMatrix) -> None:
        self._set(self._xrd @ right._xrd)

    def MultiplyLeft(self, left: TGeoMatrix) -> None:
        self._set(left._xrd @ self._xrd)


class TGeoScale(TGeoMatrix):
    """``TGeoScale(sx, sy, sz)``: stretched along each axis."""

    def __init__(self, *args: Any) -> None:
        named = bool(args) and isinstance(args[0], str)
        super().__init__(args[0] if named else "")
        scales = [float(v) for v in (args[1:] if named else args)] or [1.0, 1.0, 1.0]
        self._set(Matrix(np.diag(scales[:3])))


#: ``gGeoIdentity``: the placement a volume added with none is given.
gGeoIdentity = TGeoIdentity()
