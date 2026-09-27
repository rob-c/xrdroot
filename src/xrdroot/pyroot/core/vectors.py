"""``TVector2``, ``TVector3``, ``TLorentzVector``: ROOT's physics vectors, operation for operation.

Each is ROOT's legacy class as its source has it - ``Phi`` of the null
vector is zero, ``PseudoRapidity`` along the beam is ``±10e10``, ``Mag`` of
a space-like four-vector is minus the root of minus its square - with
Python's operators standing for C++'s: ``+``, ``-``, ``*`` by a number, and
``*`` between two vectors their dot product.
"""

from __future__ import annotations

import math
from typing import Any

from .cformat import c_format
from .messages import message
from .objects import TObject

__all__ = ["TVector2", "TVector3", "TLorentzVector", "TRotation"]

TWO_PI = 2.0 * math.pi
RAD_TO_DEG = 180.0 / math.pi


def _wrapped(x: float, low: float, where: str) -> float:
    """``x`` brought into ``[low, low + 2 pi)``, as ROOT's ``Phi_`` functions bring it."""
    if math.isnan(x):
        message("Error", where, "function called with NaN")
        return x
    while x >= low + TWO_PI:
        x -= TWO_PI
    while x < low:
        x += TWO_PI
    return x


def _signed_root(squared: float) -> float:
    """The root of a square that may be negative, keeping the sign: ROOT's ``Mag``."""
    return -math.sqrt(-squared) if squared < 0.0 else math.sqrt(squared)


def boosted(b: tuple[float, float, float], v: tuple[float, ...]) -> tuple[float, ...]:
    """``TLorentzVector::Boost``: the four-vector ``v`` boosted by the velocity ``b``."""
    b2 = b[0] * b[0] + b[1] * b[1] + b[2] * b[2]
    gamma = 1.0 / math.sqrt(1.0 - b2)
    bp = b[0] * v[0] + b[1] * v[1] + b[2] * v[2]
    along = ((gamma - 1.0) / b2 if b2 > 0 else 0.0) * bp + gamma * v[3]
    return v[0] + along * b[0], v[1] + along * b[1], v[2] + along * b[2], gamma * (v[3] + bp)


class TVector2(TObject):
    """``TVector2``: a vector in the plane."""

    CLASS_TITLE = "A 2D physics vector"

    def __init__(self, x: Any = 0.0, y: float = 0.0) -> None:
        super().__init__()
        if isinstance(x, TVector2):
            x, y = x.X(), x.Y()
        elif not isinstance(x, (int, float)):
            x, y = x[0], x[1]
        self._x, self._y = float(x), float(y)

    @staticmethod
    def Phi_0_2pi(x: float) -> float:
        return _wrapped(float(x), 0.0, "TVector2::Phi_0_2pi")

    @staticmethod
    def Phi_mpi_pi(x: float) -> float:
        return _wrapped(float(x), -math.pi, "TVector2::Phi_mpi_pi")

    def X(self) -> float:
        return self._x

    def Y(self) -> float:
        return self._y

    Px, Py = X, Y

    def Set(self, x: Any, y: float = 0.0) -> None:
        other = TVector2(x, y)
        self._x, self._y = other._x, other._y

    def SetX(self, x: float) -> None:
        self._x = float(x)

    def SetY(self, y: float) -> None:
        self._y = float(y)

    def Mod2(self) -> float:
        return self._x * self._x + self._y * self._y

    def Mod(self) -> float:
        return math.sqrt(self.Mod2())

    def Phi(self) -> float:
        """``Phi``: in ``[0, 2 pi)``, as ``pi + atan2(-y, -x)``."""
        return math.pi + math.atan2(-self._y, -self._x)

    def DeltaPhi(self, v: TVector2) -> float:
        return self.Phi_mpi_pi(v.Phi() - self.Phi())

    def Unit(self) -> TVector2:
        return self / self.Mod() if self.Mod2() else TVector2()

    def Ort(self) -> TVector2:
        return self.Unit()

    def Rotate(self, phi: float) -> TVector2:
        c, s = math.cos(phi), math.sin(phi)
        return TVector2(self._x * c - self._y * s, self._x * s + self._y * c)

    def Proj(self, v: TVector2) -> TVector2:
        made: TVector2 = v * ((self * v) / v.Mod2())
        return made

    def Norm(self, v: TVector2) -> TVector2:
        return self - self.Proj(v)

    def SetMagPhi(self, mag: float, phi: float) -> None:
        self._x, self._y = abs(mag) * math.cos(phi), abs(mag) * math.sin(phi)

    def Print(self, option: str = "") -> None:
        print(c_format("%s %s (x,y)=(%f,%f) (rho,phi)=(%f,%f)", self.GetName(), self.GetTitle(),
                       self._x, self._y, self.Mod(), self.Phi() * RAD_TO_DEG))  # fmt: skip

    def __add__(self, v: TVector2) -> TVector2:
        return TVector2(self._x + v._x, self._y + v._y)

    def __sub__(self, v: TVector2) -> TVector2:
        return TVector2(self._x - v._x, self._y - v._y)

    def __neg__(self) -> TVector2:
        return TVector2(-self._x, -self._y)

    def __mul__(self, other: Any) -> Any:
        """``v * a`` scales; ``v * w`` is the dot product."""
        if isinstance(other, TVector2):
            return self._x * other._x + self._y * other._y
        return TVector2(self._x * other, self._y * other)

    __rmul__ = __mul__

    def __truediv__(self, a: float) -> TVector2:
        return TVector2(self._x / a, self._y / a)

    def __xor__(self, v: TVector2) -> float:
        """``v ^ w``: the cross product's one component."""
        return self._x * v._y - self._y * v._x

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TVector2) and (self._x, self._y) == (other._x, other._y)

    def __hash__(self) -> int:
        return hash((self._x, self._y))

    def __getitem__(self, i: int) -> float:
        return (self._x, self._y)[i]

    def __call__(self, i: int) -> float:
        return self[i]


def _three(x: Any, y: float, z: float) -> tuple[float, float, float]:
    """A vector's three components, from numbers, another vector or an array."""
    if isinstance(x, TVector3):
        return x._x, x._y, x._z
    if isinstance(x, (int, float)):
        return float(x), float(y), float(z)
    return float(x[0]), float(x[1]), float(x[2])


class TVector3(TObject):
    """``TVector3``: a vector in space."""

    CLASS_TITLE = "A 3D physics vector"

    def __init__(self, x: Any = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        super().__init__()
        self._x, self._y, self._z = _three(x, y, z)

    def X(self) -> float:
        return self._x

    def Y(self) -> float:
        return self._y

    def Z(self) -> float:
        return self._z

    Px, Py, Pz, x, y, z = X, Y, Z, X, Y, Z

    def SetX(self, x: float) -> None:
        self._x = float(x)

    def SetY(self, y: float) -> None:
        self._y = float(y)

    def SetZ(self, z: float) -> None:
        self._z = float(z)

    def SetXYZ(self, x: float, y: float, z: float) -> None:
        self._x, self._y, self._z = float(x), float(y), float(z)

    def GetXYZ(self, carray: Any) -> None:
        for index, value in enumerate((self._x, self._y, self._z)):
            carray[index] = value

    def Mag2(self) -> float:
        return self._x * self._x + self._y * self._y + self._z * self._z

    def Mag(self) -> float:
        return math.sqrt(self.Mag2())

    def Perp2(self, p: TVector3 | None = None) -> float:
        """``Perp2()``: across the z axis; ``Perp2(p)``: across ``p``."""
        if p is None:
            return self._x * self._x + self._y * self._y
        tot, ss, per = p.Mag2(), self.Dot(p), self.Mag2()
        if tot > 0.0:
            per -= ss * ss / tot
        return max(per, 0.0)

    def Perp(self, p: TVector3 | None = None) -> float:
        return math.sqrt(self.Perp2(p))

    Pt = Perp

    def Phi(self) -> float:
        return 0.0 if self._x == 0.0 and self._y == 0.0 else math.atan2(self._y, self._x)

    def Theta(self) -> float:
        if self._x == 0.0 and self._y == 0.0 and self._z == 0.0:
            return 0.0
        return math.atan2(self.Perp(), self._z)

    def CosTheta(self) -> float:
        total = self.Mag()
        return 1.0 if total == 0.0 else self._z / total

    def PseudoRapidity(self) -> float:
        """``PseudoRapidity``: ``±10e10`` along the beam, 0 for the null vector."""
        cos = self.CosTheta()
        if cos * cos < 1:
            return -0.5 * math.log((1.0 - cos) / (1.0 + cos))
        if self._z == 0:
            return 0.0
        return 10e10 if self._z > 0 else -10e10

    Eta = PseudoRapidity

    def DeltaPhi(self, v: TVector3) -> float:
        return TVector2.Phi_mpi_pi(self.Phi() - v.Phi())

    def DeltaR(self, v: TVector3) -> float:
        return math.hypot(self.Eta() - v.Eta(), self.DeltaPhi(v))

    DrEtaPhi = DeltaR

    def EtaPhiVector(self) -> TVector2:
        return TVector2(self.Eta(), self.Phi())

    def XYvector(self) -> TVector2:
        return TVector2(self._x, self._y)

    def Angle(self, q: TVector3) -> float:
        total = self.Mag2() * q.Mag2()
        if total <= 0:
            return 0.0
        return math.acos(max(-1.0, min(1.0, self.Dot(q) / math.sqrt(total))))

    def Unit(self) -> TVector3:
        tot2 = self.Mag2()
        scale = 1.0 / math.sqrt(tot2) if tot2 > 0 else 1.0
        return TVector3(self._x * scale, self._y * scale, self._z * scale)

    def Orthogonal(self) -> TVector3:
        xx, yy, zz = abs(self._x), abs(self._y), abs(self._z)
        if xx < yy:
            return TVector3(0, self._z, -self._y) if xx < zz else TVector3(self._y, -self._x, 0)
        return TVector3(-self._z, 0, self._x) if yy < zz else TVector3(self._y, -self._x, 0)

    def Dot(self, p: TVector3) -> float:
        return self._x * p._x + self._y * p._y + self._z * p._z

    def Cross(self, p: TVector3) -> TVector3:
        return TVector3(self._y * p._z - p._y * self._z, self._z * p._x - p._z * self._x,
                        self._x * p._y - p._x * self._y)  # fmt: skip

    def SetMag(self, ma: float) -> None:
        factor = self.Mag()
        if factor == 0:
            self.Warning("SetMag", "zero vector can't be stretched")
            return
        self.SetXYZ(*(value * ma / factor for value in (self._x, self._y, self._z)))

    def SetPerp(self, r: float) -> None:
        p = self.Perp()
        if p != 0.0:
            self._x, self._y = self._x * r / p, self._y * r / p

    def SetTheta(self, th: float) -> None:
        ma, ph = self.Mag(), self.Phi()
        self.SetXYZ(ma * math.sin(th) * math.cos(ph), ma * math.sin(th) * math.sin(ph),
                    ma * math.cos(th))  # fmt: skip

    def SetPhi(self, ph: float) -> None:
        xy = self.Perp()
        self._x, self._y = xy * math.cos(ph), xy * math.sin(ph)

    def SetMagThetaPhi(self, mag: float, theta: float, phi: float) -> None:
        amag = abs(mag)
        self.SetXYZ(amag * math.sin(theta) * math.cos(phi),
                    amag * math.sin(theta) * math.sin(phi), amag * math.cos(theta))  # fmt: skip

    def SetPtEtaPhi(self, pt: float, eta: float, phi: float) -> None:
        apt = abs(pt)
        self.SetXYZ(apt * math.cos(phi), apt * math.sin(phi),
                    apt / math.tan(2.0 * math.atan(math.exp(-eta))))  # fmt: skip

    def SetPtThetaPhi(self, pt: float, theta: float, phi: float) -> None:
        tan = math.tan(theta)
        self.SetXYZ(pt * math.cos(phi), pt * math.sin(phi), pt / tan if tan else 0.0)

    # -- rotations -------------------------------------------------------------------

    def RotateX(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        self._y, self._z = c * self._y - s * self._z, s * self._y + c * self._z

    def RotateY(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        self._z, self._x = c * self._z - s * self._x, s * self._z + c * self._x

    def RotateZ(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        self._x, self._y = c * self._x - s * self._y, s * self._x + c * self._y

    def Rotate(self, angle: float, axis: TVector3) -> None:
        """``Rotate``: by ``angle`` about ``axis``, as ``TRotation::Rotate`` turns it."""
        turned = TRotation().Rotate(angle, axis) * self
        self.SetXYZ(turned._x, turned._y, turned._z)

    def RotateUz(self, uz: TVector3) -> None:
        """``RotateUz``: from the frame whose z axis is the unit vector ``uz``."""
        u1, u2, u3 = uz._x, uz._y, uz._z
        up = u1 * u1 + u2 * u2
        if up:
            up = math.sqrt(up)
            px, py, pz = self._x, self._y, self._z
            self._x = (u1 * u3 * px - u2 * py + u1 * up * pz) / up
            self._y = (u2 * u3 * px + u1 * py + u2 * up * pz) / up
            self._z = (u3 * u3 * px - px + u3 * up * pz) / up
        elif u3 < 0.0:
            self._x, self._z = -self._x, -self._z

    def Transform(self, m: TRotation) -> TVector3:
        turned = m * self
        self.SetXYZ(turned._x, turned._y, turned._z)
        return self

    # -- printing and operators ------------------------------------------------------------

    def Print(self, option: str = "") -> None:
        print(c_format("%s %s (x,y,z)=(%f,%f,%f) (rho,theta,phi)=(%f,%f,%f)", self.GetName(),
                       self.GetTitle(), self._x, self._y, self._z, self.Mag(),
                       self.Theta() * RAD_TO_DEG, self.Phi() * RAD_TO_DEG))  # fmt: skip

    def __add__(self, p: TVector3) -> TVector3:
        return TVector3(self._x + p._x, self._y + p._y, self._z + p._z)

    def __sub__(self, p: TVector3) -> TVector3:
        return TVector3(self._x - p._x, self._y - p._y, self._z - p._z)

    def __neg__(self) -> TVector3:
        return TVector3(-self._x, -self._y, -self._z)

    def __mul__(self, other: Any) -> Any:
        """``v * a`` scales; ``v * w`` is the dot product."""
        if isinstance(other, TVector3):
            return self.Dot(other)
        return TVector3(self._x * other, self._y * other, self._z * other)

    def __rmul__(self, other: Any) -> Any:
        return self * other

    def __truediv__(self, a: float) -> TVector3:
        return TVector3(self._x / a, self._y / a, self._z / a)

    def __iadd__(self, p: TVector3) -> TVector3:
        self.SetXYZ(self._x + p._x, self._y + p._y, self._z + p._z)
        return self

    def __isub__(self, p: TVector3) -> TVector3:
        self.SetXYZ(self._x - p._x, self._y - p._y, self._z - p._z)
        return self

    def __imul__(self, a: Any) -> TVector3:
        turned = a * self if isinstance(a, TRotation) else self * a
        self.SetXYZ(turned._x, turned._y, turned._z)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TVector3):
            return NotImplemented
        return (self._x, self._y, self._z) == (other._x, other._y, other._z)

    def __hash__(self) -> int:
        return hash((self._x, self._y, self._z))

    def __getitem__(self, i: int) -> float:
        if not 0 <= i < 3:
            self.Error("operator()(i)", "bad index (%d) returning 0", i)
            return 0.0
        return (self._x, self._y, self._z)[i]

    def __setitem__(self, i: int, value: float) -> None:
        parts = [self._x, self._y, self._z]
        parts[i] = float(value)
        self.SetXYZ(*parts)

    def __call__(self, i: int) -> float:
        return self[i]

    def __iter__(self) -> Any:
        return iter((self._x, self._y, self._z))


class TRotation(TObject):
    """``TRotation``: a rotation of space, as the three-by-three matrix it is."""

    CLASS_TITLE = "Rotations of TVector3 objects"

    def __init__(self) -> None:
        super().__init__()
        self._m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _after(self, other: list[list[float]]) -> TRotation:
        """This rotation followed by ``other``: ``other * self``, in place."""
        self._m = [[sum(other[i][k] * self._m[k][j] for k in range(3)) for j in range(3)]
                   for i in range(3)]  # fmt: skip
        return self

    def Rotate(self, angle: float, axis: TVector3) -> TRotation:
        """``Rotate(a, axis)``: then by ``a`` about ``axis``, Rodrigues's matrix."""
        if angle == 0:
            return self
        length = axis.Mag()
        if length == 0.0:
            self.Error("Rotate(angle,axis)", "zero axis")
            return self
        u = [axis.X() / length, axis.Y() / length, axis.Z() / length]
        c, s = math.cos(angle), math.sin(angle)
        made = [[(1 - c) * u[i] * u[j] + (c if i == j else 0.0) for j in range(3)]
                for i in range(3)]  # fmt: skip
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            made[i][j] -= s * u[k]
            made[j][i] += s * u[k]
        return self._after(made)

    def RotateX(self, angle: float) -> TRotation:
        return self.Rotate(angle, TVector3(1, 0, 0))

    def RotateY(self, angle: float) -> TRotation:
        return self.Rotate(angle, TVector3(0, 1, 0))

    def RotateZ(self, angle: float) -> TRotation:
        return self.Rotate(angle, TVector3(0, 0, 1))

    def Inverse(self) -> TRotation:
        made = TRotation()
        made._m = [list(row) for row in zip(*self._m)]
        return made

    def __call__(self, i: int, j: int) -> float:
        return self._m[i][j]

    def __mul__(self, other: Any) -> Any:
        if isinstance(other, TRotation):
            made = TRotation()
            made._m = [list(row) for row in other._m]
            return made._after(self._m)
        values = [sum(self._m[i][k] * other[k] for k in range(3)) for i in range(3)]
        return TVector3(*values)


class TLorentzVector(TObject):
    """``TLorentzVector``: a four-vector, space then time, with the (-,-,-,+) metric."""

    CLASS_TITLE = "A four vector with (-,-,-,+) metric"

    def __init__(self, x: Any = 0.0, y: Any = 0.0, z: float = 0.0, t: float = 0.0) -> None:
        super().__init__()
        self._p: TVector3
        self._e: float
        if isinstance(x, TLorentzVector):
            self._p, self._e = TVector3(x._p), x._e
        elif isinstance(x, TVector3):
            self._p, self._e = TVector3(x), float(y)
        elif isinstance(x, (int, float)):
            self._p, self._e = TVector3(x, y, z), float(t)
        else:
            self._p, self._e = TVector3(x), float(x[3])

    # -- components ------------------------------------------------------------------------

    def X(self) -> float:
        return self._p._x

    def Y(self) -> float:
        return self._p._y

    def Z(self) -> float:
        return self._p._z

    def T(self) -> float:
        return self._e

    Px, Py, Pz, E, Energy = X, Y, Z, T, T

    def SetX(self, a: float) -> None:
        self._p.SetX(a)

    def SetY(self, a: float) -> None:
        self._p.SetY(a)

    def SetZ(self, a: float) -> None:
        self._p.SetZ(a)

    def SetT(self, a: float) -> None:
        self._e = float(a)

    SetPx, SetPy, SetPz, SetE = SetX, SetY, SetZ, SetT

    def Vect(self) -> TVector3:
        return TVector3(self._p)

    def SetVect(self, p: TVector3) -> None:
        self._p = TVector3(p)

    def SetXYZT(self, x: float, y: float, z: float, t: float) -> None:
        self._p.SetXYZ(x, y, z)
        self._e = float(t)

    SetPxPyPzE = SetXYZT

    def SetXYZM(self, x: float, y: float, z: float, m: float) -> None:
        """``SetXYZM``: the energy of mass ``m`` - or, for ``m < 0``, of ``p² - m²``."""
        p2 = x * x + y * y + z * z
        self.SetXYZT(x, y, z, math.sqrt(p2 + m * m) if m >= 0 else math.sqrt(max(p2 - m * m, 0.0)))

    def SetPtEtaPhiM(self, pt: float, eta: float, phi: float, m: float) -> None:
        pt = abs(pt)
        self.SetXYZM(pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), m)

    def SetPtEtaPhiE(self, pt: float, eta: float, phi: float, e: float) -> None:
        pt = abs(pt)
        self.SetXYZT(pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta), e)

    def SetVectM(self, spatial: TVector3, mass: float) -> None:
        self.SetXYZM(spatial.X(), spatial.Y(), spatial.Z(), mass)

    SetVectMag = SetVectM

    def GetXYZT(self, carray: Any) -> None:
        for index, value in enumerate((self.X(), self.Y(), self.Z(), self._e)):
            carray[index] = value

    # -- what it measures ------------------------------------------------------------------

    def P(self) -> float:
        return self._p.Mag()

    Rho = P

    def Perp2(self, v: TVector3 | None = None) -> float:
        return self._p.Perp2(v)

    def Perp(self, v: TVector3 | None = None) -> float:
        return self._p.Perp(v)

    Pt = Perp

    def SetPerp(self, r: float) -> None:
        self._p.SetPerp(r)

    def Phi(self) -> float:
        return self._p.Phi()

    def Theta(self) -> float:
        return self._p.Theta()

    def CosTheta(self) -> float:
        return self._p.CosTheta()

    def SetTheta(self, th: float) -> None:
        self._p.SetTheta(th)

    def SetPhi(self, phi: float) -> None:
        self._p.SetPhi(phi)

    def SetRho(self, rho: float) -> None:
        self._p.SetMag(rho)

    def Mag2(self) -> float:
        return self._e * self._e - self._p.Mag2()

    def Mag(self) -> float:
        return _signed_root(self.Mag2())

    M2, M = Mag2, Mag

    def Mt2(self) -> float:
        return self._e * self._e - self.Z() * self.Z()

    def Mt(self) -> float:
        return _signed_root(self.Mt2())

    def Et2(self, v: TVector3 | None = None) -> float:
        pt2 = self._p.Perp2(v)
        along = self.Z() if v is None else self._p.Dot(v.Unit())
        return 0.0 if pt2 == 0 else self._e * self._e * pt2 / (pt2 + along * along)

    def Et(self, v: TVector3 | None = None) -> float:
        root = math.sqrt(self.Et2(v))
        return -root if self._e < 0.0 else root

    def Beta(self) -> float:
        return self._p.Mag() / self._e

    def Gamma(self) -> float:
        b = self.Beta()
        return 1.0 / math.sqrt(1 - b * b)

    def Plus(self) -> float:
        return self._e + self.Z()

    def Minus(self) -> float:
        return self._e - self.Z()

    def Rapidity(self) -> float:
        return 0.5 * math.log((self._e + self.Z()) / (self._e - self.Z()))

    def PseudoRapidity(self) -> float:
        return self._p.PseudoRapidity()

    Eta = PseudoRapidity

    def DeltaPhi(self, v: TLorentzVector) -> float:
        return TVector2.Phi_mpi_pi(self.Phi() - v.Phi())

    def DeltaR(self, v: TLorentzVector, useRapidity: bool = False) -> float:
        """``DeltaR``: in eta and phi, or with ``useRapidity`` in rapidity and phi."""
        along = self.Rapidity() - v.Rapidity() if useRapidity else self.Eta() - v.Eta()
        return math.hypot(along, self.DeltaPhi(v))

    def DrEtaPhi(self, v: TLorentzVector) -> float:
        return self.DeltaR(v)

    def DrRapidityPhi(self, v: TLorentzVector) -> float:
        return self.DeltaR(v, True)

    def EtaPhiVector(self) -> TVector2:
        return TVector2(self.Eta(), self.Phi())

    def Angle(self, v: TVector3) -> float:
        return self._p.Angle(v)

    def Dot(self, q: TLorentzVector) -> float:
        return self._e * q._e - self._p.Dot(q._p)

    def BoostVector(self) -> TVector3:
        return TVector3(self.X() / self._e, self.Y() / self._e, self.Z() / self._e)

    # -- transformations ---------------------------------------------------------------------

    def Boost(self, bx: Any, by: float = 0.0, bz: float = 0.0) -> None:
        """``Boost(b)`` or ``Boost(bx, by, bz)``: from the frame moving at ``b`` to this one."""
        x, y, z, t = boosted(_three(bx, by, bz), (self.X(), self.Y(), self.Z(), self._e))
        self._p.SetXYZ(x, y, z)
        self._e = t

    def RotateX(self, angle: float) -> None:
        self._p.RotateX(angle)

    def RotateY(self, angle: float) -> None:
        self._p.RotateY(angle)

    def RotateZ(self, angle: float) -> None:
        self._p.RotateZ(angle)

    def RotateUz(self, uz: TVector3) -> None:
        self._p.RotateUz(uz)

    def Rotate(self, angle: float, axis: TVector3) -> None:
        self._p.Rotate(angle, axis)

    def Transform(self, m: TRotation) -> TLorentzVector:
        self._p.Transform(m)
        return self

    # -- printing and operators ----------------------------------------------------------

    def Print(self, option: str = "") -> None:
        print(c_format("(x,y,z,t)=(%f,%f,%f,%f) (P,eta,phi,E)=(%f,%f,%f,%f)", self.X(), self.Y(),
                       self.Z(), self._e, self.P(), self.Eta(), self.Phi(), self._e))  # fmt: skip

    def __add__(self, q: TLorentzVector) -> TLorentzVector:
        return TLorentzVector(self._p + q._p, self._e + q._e)

    def __sub__(self, q: TLorentzVector) -> TLorentzVector:
        return TLorentzVector(self._p - q._p, self._e - q._e)

    def __neg__(self) -> TLorentzVector:
        return TLorentzVector(-self.X(), -self.Y(), -self.Z(), -self._e)

    def __mul__(self, other: Any) -> Any:
        """``v * a`` scales; ``v * w`` is the four-vector product."""
        if isinstance(other, TLorentzVector):
            return self.Dot(other)
        return TLorentzVector(self._p * other, self._e * other)

    def __rmul__(self, other: Any) -> Any:
        return self * other

    def __iadd__(self, q: TLorentzVector) -> TLorentzVector:
        self._p += q._p
        self._e += q._e
        return self

    def __isub__(self, q: TLorentzVector) -> TLorentzVector:
        self._p -= q._p
        self._e -= q._e
        return self

    def __imul__(self, a: Any) -> TLorentzVector:
        if isinstance(a, TRotation):
            return self.Transform(a)
        self._p *= a
        self._e *= a
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TLorentzVector):
            return NotImplemented
        return self._p == other._p and self._e == other._e

    def __hash__(self) -> int:
        return hash((self.X(), self.Y(), self.Z(), self._e))

    def __getitem__(self, i: int) -> float:
        if not 0 <= i < 4:
            self.Error("operator()()", "bad index (%d) returning 0", i)
            return 0.0
        return (self.X(), self.Y(), self.Z(), self._e)[i]

    def __setitem__(self, i: int, value: float) -> None:
        if i == 3:
            self._e = float(value)
        else:
            self._p[i] = value

    def __call__(self, i: int) -> float:
        return self[i]

    def __iter__(self) -> Any:
        return iter((self.X(), self.Y(), self.Z(), self._e))
