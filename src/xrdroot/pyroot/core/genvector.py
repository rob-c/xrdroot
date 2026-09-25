"""``ROOT::Math``'s vectors: ``PtEtaPhiMVector``, ``PxPyPzEVector``, ``XYZVector`` and kin.

GenVector keeps a vector in the coordinates it was made in - a
``PtEtaPhiMVector`` holds pt, eta, phi and the mass - so ``M()`` of one is
the mass it was given, not one worked back from a sum of squares, and it
prints in those coordinates, ``(pt,eta,phi,m)``, as ``operator<<`` does.
Here each vector keeps its own four (or three, or two) numbers the same way;
anything asked of it in another system is worked out through Cartesian
components, and a sum of two takes the left one's coordinates, as C++'s
templates do. ROOT's ``Eta`` of a vector along the beam is ``z ± 22756``.
"""

from __future__ import annotations

import math
from typing import Any, Callable, ClassVar

__all__ = [
    "LorentzVector",
    "PxPyPzEVector",
    "PxPyPzMVector",
    "PtEtaPhiEVector",
    "PtEtaPhiMVector",
    "XYZTVector",
    "XYZTVectorF",
    "DisplacementVector3D",
    "PositionVector3D",
    "XYZVector",
    "XYZVectorF",
    "XYZPoint",
    "Polar3DVector",
    "RhoEtaPhiVector",
    "RhoZPhiVector",
    "XYVector",
    "Polar2DVector",
    "VectorUtil",
]

#: GenVector's ``etaMax``: what ``Eta`` adds to ``z`` along the beam.
ETA_MAX = 22756.0

Quad = tuple[float, float, float, float]


def eta_from(rho: float, z: float) -> float:
    """GenVector's ``Eta_FromRhoZ``."""
    if rho > 0:
        return math.asinh(z / rho)
    if z == 0:
        return 0.0
    return z + ETA_MAX if z > 0 else z - ETA_MAX


def _signed_root(squared: float) -> float:
    return -math.sqrt(-squared) if squared < 0 else math.sqrt(squared)


def _energy(p2: float, m: float) -> float:
    """The energy of momentum squared ``p2`` and mass ``m``; a negative mass is space-like."""
    return math.sqrt(p2 + m * m) if m >= 0 else _signed_root(p2 - m * m)


def _transverse(pt: float, eta: float, phi: float) -> tuple[float, float, float]:
    return pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)


def _from_pt_eta_phi_e(c: Quad) -> Quad:
    return (*_transverse(c[0], c[1], c[2]), c[3])


def _from_pt_eta_phi_m(c: Quad) -> Quad:
    x, y, z = _transverse(c[0], c[1], c[2])
    return x, y, z, _energy(x * x + y * y + z * z, c[3])


def _from_px_py_pz_m(c: Quad) -> Quad:
    return c[0], c[1], c[2], _energy(c[0] ** 2 + c[1] ** 2 + c[2] ** 2, c[3])


def _mass(x: Quad) -> float:
    return _signed_root(x[3] ** 2 - x[0] ** 2 - x[1] ** 2 - x[2] ** 2)


def _pt(x: Quad) -> float:
    return math.hypot(x[0], x[1])


def _phi(x: Quad) -> float:
    return 0.0 if x[0] == 0 and x[1] == 0 else math.atan2(x[1], x[0])


#: Each four-dimensional system: its coordinates' names, to Cartesian, and back.
SYSTEMS_4D: dict[str, tuple[tuple[str, ...], Callable[[Quad], Quad], Callable[[Quad], Quad]]] = {
    "PxPyPzE4D": (("Px", "Py", "Pz", "E"), lambda c: c, lambda x: x),
    "PxPyPzM4D": (("Px", "Py", "Pz", "M"), _from_px_py_pz_m, lambda x: (*x[:3], _mass(x))),
    "PtEtaPhiE4D": (
        ("Pt", "Eta", "Phi", "E"),
        _from_pt_eta_phi_e,
        lambda x: (_pt(x), eta_from(_pt(x), x[2]), _phi(x), x[3]),
    ),
    "PtEtaPhiM4D": (
        ("Pt", "Eta", "Phi", "M"),
        _from_pt_eta_phi_m,
        lambda x: (_pt(x), eta_from(_pt(x), x[2]), _phi(x), _mass(x)),
    ),
}


def _text(values: Any) -> str:
    """How ``operator<<`` writes a vector: its coordinates at six digits, in brackets."""
    return "(" + ",".join(f"{value:g}" for value in values) + ")"


#: The other names each coordinate is asked for by.
ALIASES = {"Pt": ("Rho", "Perp"), "E": ("T", "e", "t"), "M": ("Mag",),
           "Px": ("X", "x"), "Py": ("Y", "y"), "Pz": ("Z", "z")}  # fmt: skip


def _p2(x: Quad) -> float:
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2


#: What a four-vector is asked, from its Cartesian components.
DERIVED_4D: dict[str, Callable[[Quad], float]] = {
    "Px": lambda x: x[0],
    "Py": lambda x: x[1],
    "Pz": lambda x: x[2],
    "E": lambda x: x[3],
    "M": _mass,
    "M2": lambda x: x[3] ** 2 - _p2(x),
    "Mag2": lambda x: x[3] ** 2 - _p2(x),
    "Pt": _pt,
    "Perp2": lambda x: x[0] ** 2 + x[1] ** 2,
    "Pt2": lambda x: x[0] ** 2 + x[1] ** 2,
    "Eta": lambda x: eta_from(_pt(x), x[2]),
    "Phi": _phi,
    "P": lambda x: math.sqrt(_p2(x)),
    "R": lambda x: math.sqrt(_p2(x)),
    "P2": _p2,
    "Theta": lambda x: 0.0 if _p2(x) == 0 else math.atan2(_pt(x), x[2]),
    "Rapidity": lambda x: 0.5 * math.log((x[3] + x[2]) / (x[3] - x[2])),
    "Mt2": lambda x: x[3] ** 2 - x[2] ** 2,
    "Mt": lambda x: _signed_root(x[3] ** 2 - x[2] ** 2),
    "Et2": lambda x: 0.0 if _p2(x) == 0 else x[3] ** 2 * _pt(x) ** 2 / _p2(x),
    "Et": lambda x: math.copysign(x[3] * _pt(x) / math.sqrt(_p2(x)), x[3]) if _p2(x) else 0.0,
    "Beta": lambda x: math.sqrt(_p2(x)) / x[3],
    "Gamma": lambda x: 1.0 / math.sqrt(1.0 - _p2(x) / x[3] ** 2),
}
for _name, _others in ALIASES.items():
    for _other in _others:
        DERIVED_4D[_other] = DERIVED_4D[_name]


def _getter(name: str) -> Any:
    def get(self: Any) -> float:
        return float(self._get(name))

    get.__doc__ = f"``{name}()``."
    return get


def _setter(name: str) -> Any:
    def set_(self: Any, value: float) -> None:
        self._set(name, float(value))

    set_.__doc__ = f"``Set{name}(value)``."
    return set_


class _Vector:
    """What every GenVector class shares: its own coordinates, and questions answered from them."""

    SYSTEM = ""
    NAMES: tuple[str, ...] = ()
    DERIVED: ClassVar[dict[str, Callable[[Any], float]]] = {}
    SYSTEMS: ClassVar[dict[str, Any]] = {}
    SETTABLE: tuple[str, ...] = ()

    def __init__(self, *args: Any) -> None:
        if len(args) == 1 and isinstance(args[0], _Vector):
            self._set_cartesian(args[0]._cartesian())
        else:
            given = [float(value) for value in args] or [0.0] * len(self.NAMES)
            self._c: Any = tuple(given)

    def _cartesian(self) -> Any:
        """The vector's Cartesian components, worked out from its own."""
        return self.SYSTEMS[self.SYSTEM][1](self._c)

    def _set_cartesian(self, x: Any) -> None:
        """Make the vector the one with Cartesian components ``x``, in its own coordinates."""
        self._c = self.SYSTEMS[self.SYSTEM][2](tuple(float(v) for v in x))

    def _get(self, name: str) -> float:
        own = [coord for coord in self.NAMES if coord == name or name in ALIASES.get(coord, ())]
        if own:
            return float(self._c[self.NAMES.index(own[0])])
        return float(self.DERIVED[name](self._cartesian()))

    def _set(self, name: str, value: float) -> None:
        """One coordinate changed, in whichever system has it, the others there kept."""
        system = self.SYSTEM if name in self.NAMES else self._system_with(name)
        names, to_cartesian, from_cartesian = self.SYSTEMS[system]
        values = list(from_cartesian(self._cartesian()) if system != self.SYSTEM else self._c)
        values[names.index(name)] = value
        if system == self.SYSTEM:
            self._c = tuple(values)
        else:
            self._set_cartesian(to_cartesian(tuple(values)))

    def _system_with(self, name: str) -> str:
        return next(key for key, (names, _, _) in self.SYSTEMS.items() if name in names)

    def Coordinates(self) -> Any:
        return self

    def SetCoordinates(self, *values: float) -> Any:
        self._c = tuple(float(value) for value in values)
        return self

    def GetCoordinates(self, dest: Any = None) -> Any:
        if dest is not None:
            for index, value in enumerate(self._c):
                dest[index] = value
        return self._c

    def __str__(self) -> str:
        return _text(self._c)

    def __repr__(self) -> str:
        return f"<ROOT.Math.{type(self).__name__} {_text(self._c)}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Vector) and self._c == other._c

    def __hash__(self) -> int:
        return hash(self._c)

    def __iter__(self) -> Any:
        return iter(self._cartesian())

    def __getitem__(self, index: int) -> float:
        return float(self._cartesian()[index])

    def _made(self, x: Any) -> Any:
        made = type(self).__new__(type(self))
        made._set_cartesian(tuple(x))
        return made

    def __add__(self, other: _Vector) -> Any:
        return self._made(a + b for a, b in zip(self._cartesian(), other._cartesian()))

    def __sub__(self, other: _Vector) -> Any:
        return self._made(a - b for a, b in zip(self._cartesian(), other._cartesian()))

    def __neg__(self) -> Any:
        return self._made(-a for a in self._cartesian())

    def __mul__(self, a: Any) -> Any:
        if isinstance(a, _Vector):
            return self.Dot(a)
        return self._made(value * float(a) for value in self._cartesian())

    def __rmul__(self, a: Any) -> Any:
        return self * a

    def __truediv__(self, a: float) -> Any:
        return self._made(value / float(a) for value in self._cartesian())

    def __iadd__(self, other: _Vector) -> Any:
        self._set_cartesian(tuple(a + b for a, b in zip(self._cartesian(), other._cartesian())))
        return self

    def __isub__(self, other: _Vector) -> Any:
        self._set_cartesian(tuple(a - b for a, b in zip(self._cartesian(), other._cartesian())))
        return self

    def __imul__(self, a: Any) -> Any:
        self._set_cartesian(tuple(value * float(a) for value in self._cartesian()))
        return self

    def Dot(self, other: _Vector) -> float:
        """The Euclidean dot product; a four-vector's is Minkowski's."""
        return float(sum(a * b for a, b in zip(self._cartesian(), other._cartesian())))


def _dress(cls: Any) -> Any:
    """A vector class given a getter for every question and a setter per own coordinate."""
    for name in cls.DERIVED:
        if name not in cls.__dict__:
            setattr(cls, name, _getter(name))
    for coord in cls.SETTABLE:
        setattr(cls, f"Set{coord}", _setter(coord))
    return cls


class LorentzVector(_Vector):
    """``ROOT::Math::LorentzVector``: a four-vector in the coordinates it was made in.

    ``LorentzVector["ROOT::Math::PtEtaPhiM4D<double>"]`` is the class of that
    system, as PyROOT spells a template; the named ones below are those.
    """

    SYSTEM = "PxPyPzE4D"
    NAMES = SYSTEMS_4D["PxPyPzE4D"][0]
    DERIVED = DERIVED_4D
    SYSTEMS = SYSTEMS_4D
    SETTABLE = ("Px", "Py", "Pz", "E", "M", "Pt", "Eta", "Phi")

    def __class_getitem__(cls, system: Any) -> Any:
        wanted = str(getattr(system, "__name__", system))
        for made in (PxPyPzEVector, PxPyPzMVector, PtEtaPhiEVector, PtEtaPhiMVector):
            if made.SYSTEM in wanted:
                return made
        raise TypeError(f"ROOT::Math has no four-vector coordinates {wanted!r}")

    def SetPxPyPzE(self, px: float, py: float, pz: float, e: float) -> Any:
        self._set_cartesian((px, py, pz, e))
        return self

    SetXYZT = SetPxPyPzE

    def Dot(self, other: _Vector) -> float:
        a, b = self._cartesian(), other._cartesian()
        return float(a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2])

    def Vect(self) -> XYZVector:
        x = self._cartesian()
        return XYZVector(x[0], x[1], x[2])

    def BoostToCM(self) -> XYZVector:
        x = self._cartesian()
        return XYZVector(-x[0] / x[3], -x[1] / x[3], -x[2] / x[3])

    def isLightlike(self, tolerance: float = 100 * 2.220446049250313e-16) -> bool:
        x = self._cartesian()
        return bool(abs(x[3] ** 2 - _p2(x)) < tolerance * x[3] ** 2)

    def isTimelike(self) -> bool:
        return self._get("M2") > 0

    def isSpacelike(self) -> bool:
        return self._get("M2") < 0

    def ColinearRapidity(self) -> float:
        x = self._cartesian()
        p = math.sqrt(_p2(x))
        return 0.5 * math.log((x[3] + p) / (x[3] - p))


class PxPyPzEVector(LorentzVector):
    """``PxPyPzEVector``, also ``XYZTVector``: Cartesian momentum and energy."""


class PxPyPzMVector(LorentzVector):
    """``PxPyPzMVector``: Cartesian momentum, and the mass."""

    SYSTEM = "PxPyPzM4D"
    NAMES = SYSTEMS_4D["PxPyPzM4D"][0]


class PtEtaPhiEVector(LorentzVector):
    """``PtEtaPhiEVector``: transverse momentum, pseudorapidity, azimuth and energy."""

    SYSTEM = "PtEtaPhiE4D"
    NAMES = SYSTEMS_4D["PtEtaPhiE4D"][0]


class PtEtaPhiMVector(LorentzVector):
    """``PtEtaPhiMVector``: transverse momentum, pseudorapidity, azimuth and mass."""

    SYSTEM = "PtEtaPhiM4D"
    NAMES = SYSTEMS_4D["PtEtaPhiM4D"][0]


for _made in (LorentzVector, PxPyPzEVector, PxPyPzMVector, PtEtaPhiEVector, PtEtaPhiMVector):
    _dress(_made)

XYZTVector = PxPyPzEVector
XYZTVectorF = PxPyPzEVector


# -- three and two dimensions ------------------------------------------------------------------


def _r3(x: Any) -> float:
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


def _from_polar(c: Any) -> Any:
    r, theta, phi = c
    return (
        r * math.sin(theta) * math.cos(phi),
        r * math.sin(theta) * math.sin(phi),
        r * math.cos(theta),
    )


def _to_polar(x: Any) -> Any:
    r = _r3(x)
    return r, (0.0 if r == 0 else math.atan2(math.hypot(x[0], x[1]), x[2])), _phi(x)


def _from_rho_eta_phi(c: Any) -> Any:
    return _transverse(c[0], c[1], c[2])


def _to_rho_eta_phi(x: Any) -> Any:
    rho = math.hypot(x[0], x[1])
    return rho, eta_from(rho, x[2]), _phi(x)


#: Each three-dimensional system: its coordinates' names, to Cartesian, and back.
SYSTEMS_3D: dict[str, Any] = {
    "Cartesian3D": (("X", "Y", "Z"), lambda c: c, lambda x: x),
    "Polar3D": (("R", "Theta", "Phi"), _from_polar, _to_polar),
    "CylindricalEta3D": (("Rho", "Eta", "Phi"), _from_rho_eta_phi, _to_rho_eta_phi),
    "Cylindrical3D": (
        ("Rho", "Z", "Phi"),
        lambda c: (c[0] * math.cos(c[2]), c[0] * math.sin(c[2]), c[1]),
        lambda x: (math.hypot(x[0], x[1]), x[2], _phi(x)),
    ),
}

#: What a three-vector is asked, from its Cartesian components.
DERIVED_3D: dict[str, Callable[[Any], float]] = {
    "X": lambda x: x[0],
    "Y": lambda x: x[1],
    "Z": lambda x: x[2],
    "R": _r3,
    "Mag2": lambda x: _r3(x) ** 2,
    "Rho": lambda x: math.hypot(x[0], x[1]),
    "Perp2": lambda x: x[0] ** 2 + x[1] ** 2,
    "Eta": lambda x: eta_from(math.hypot(x[0], x[1]), x[2]),
    "Phi": _phi,
    "Theta": lambda x: _to_polar(x)[1],
}
for _name, _others in {"X": ("x",), "Y": ("y",), "Z": ("z",), "R": ("Mag", "r"),
                       "Rho": ("Perp", "rho")}.items():  # fmt: skip
    for _other in _others:
        DERIVED_3D[_other] = DERIVED_3D[_name]


class DisplacementVector3D(_Vector):
    """``ROOT::Math::DisplacementVector3D``: a vector in space, in its own coordinates."""

    SYSTEM = "Cartesian3D"
    NAMES = SYSTEMS_3D["Cartesian3D"][0]
    DERIVED = DERIVED_3D
    SYSTEMS = SYSTEMS_3D
    SETTABLE = ("X", "Y", "Z", "R", "Theta", "Phi", "Rho", "Eta")

    def __class_getitem__(cls, system: Any) -> Any:
        wanted = str(getattr(system, "__name__", system))
        found = {"Polar3D": Polar3DVector, "CylindricalEta3D": RhoEtaPhiVector,
                 "Cylindrical3D": RhoZPhiVector, "Cartesian3D": XYZVector}  # fmt: skip
        for key in ("CylindricalEta3D", "Cylindrical3D", "Polar3D", "Cartesian3D"):
            if key in wanted:
                return found[key]
        raise TypeError(f"ROOT::Math has no three-vector coordinates {wanted!r}")

    def Unit(self) -> Any:
        length = _r3(self._cartesian())
        return self / length if length > 0 else self._made(self._cartesian())

    def Cross(self, other: _Vector) -> Any:
        a, b = self._cartesian(), other._cartesian()
        return self._made((a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                           a[0] * b[1] - a[1] * b[0]))  # fmt: skip

    def SetXYZ(self, x: float, y: float, z: float) -> Any:
        self._set_cartesian((x, y, z))
        return self


class XYZVector(DisplacementVector3D):
    """``XYZVector``: Cartesian components."""


class PositionVector3D(DisplacementVector3D):
    """``ROOT::Math::PositionVector3D``: a point in space, which here is a vector."""


class XYZPoint(PositionVector3D):
    """``XYZPoint``: a point, in Cartesian components."""


class Polar3DVector(DisplacementVector3D):
    """``Polar3DVector``: length, polar angle and azimuth."""

    SYSTEM = "Polar3D"
    NAMES = SYSTEMS_3D["Polar3D"][0]


class RhoEtaPhiVector(DisplacementVector3D):
    """``RhoEtaPhiVector``: transverse length, pseudorapidity and azimuth."""

    SYSTEM = "CylindricalEta3D"
    NAMES = SYSTEMS_3D["CylindricalEta3D"][0]


class RhoZPhiVector(DisplacementVector3D):
    """``RhoZPhiVector``: transverse length, z and azimuth."""

    SYSTEM = "Cylindrical3D"
    NAMES = SYSTEMS_3D["Cylindrical3D"][0]


XYZVectorF = XYZVector

#: The two-dimensional systems.
SYSTEMS_2D: dict[str, Any] = {
    "Cartesian2D": (("X", "Y"), lambda c: c, lambda x: x),
    "Polar2D": (
        ("R", "Phi"),
        lambda c: (c[0] * math.cos(c[1]), c[0] * math.sin(c[1])),
        lambda x: (math.hypot(x[0], x[1]), _phi(x)),
    ),
}
DERIVED_2D: dict[str, Callable[[Any], float]] = {
    "X": lambda x: x[0],
    "Y": lambda x: x[1],
    "R": lambda x: math.hypot(x[0], x[1]),
    "Mag2": lambda x: x[0] ** 2 + x[1] ** 2,
    "Phi": _phi,
}


class XYVector(_Vector):
    """``XYVector``: a vector in the plane, Cartesian."""

    SYSTEM = "Cartesian2D"
    NAMES = SYSTEMS_2D["Cartesian2D"][0]
    DERIVED = DERIVED_2D
    SYSTEMS = SYSTEMS_2D
    SETTABLE = ("X", "Y", "R", "Phi")

    def Unit(self) -> Any:
        length = math.hypot(*self._cartesian())
        return self / length if length > 0 else self._made(self._cartesian())


class Polar2DVector(XYVector):
    """``Polar2DVector``: length and azimuth."""

    SYSTEM = "Polar2D"
    NAMES = SYSTEMS_2D["Polar2D"][0]


for _kind in (DisplacementVector3D, XYZVector, PositionVector3D, XYZPoint, Polar3DVector,
              RhoEtaPhiVector, RhoZPhiVector, XYVector, Polar2DVector):  # fmt: skip
    _dress(_kind)


def _phi_mpi_pi(x: float) -> float:
    """An angle brought into ``[-pi, pi]``, as ``VectorUtil::Phi_mpi_pi`` brings it."""
    if abs(x) <= math.pi:
        return x
    return x - 2 * math.pi * math.floor((x + math.pi) / (2 * math.pi))


def _boosted(v: LorentzVector, bx: float, by: float, bz: float) -> Any:
    """``VectorUtil::boost``: ``v`` boosted by the velocity ``b``, as ``TLorentzVector`` is."""
    from .vectors import boosted

    return v._made(boosted((bx, by, bz), tuple(v._cartesian())))


class VectorUtil:
    """``ROOT::Math::VectorUtil``: what is asked of two vectors at once."""

    @staticmethod
    def DeltaPhi(v1: Any, v2: Any) -> float:
        return _phi_mpi_pi(v2.Phi() - v1.Phi())

    @staticmethod
    def DeltaR2(v1: Any, v2: Any) -> float:
        return float(VectorUtil.DeltaPhi(v1, v2) ** 2 + (v2.Eta() - v1.Eta()) ** 2)

    @staticmethod
    def DeltaR(v1: Any, v2: Any) -> float:
        return math.sqrt(VectorUtil.DeltaR2(v1, v2))

    @staticmethod
    def DeltaRapidityPhi(v1: Any, v2: Any) -> float:
        return math.hypot(VectorUtil.DeltaPhi(v1, v2), v2.Rapidity() - v1.Rapidity())

    @staticmethod
    def CosTheta(v1: Any, v2: Any) -> float:
        a, b = v1._cartesian()[:3], v2._cartesian()[:3]
        product = _r3(a) * _r3(b)
        return (
            1.0 if product <= 0 else max(-1.0, min(1.0, sum(p * q for p, q in zip(a, b)) / product))
        )

    @staticmethod
    def Angle(v1: Any, v2: Any) -> float:
        return math.acos(VectorUtil.CosTheta(v1, v2))

    @staticmethod
    def InvariantMass(v1: Any, v2: Any) -> float:
        return float((v1 + v2).M())

    @staticmethod
    def InvariantMass2(v1: Any, v2: Any) -> float:
        return float((v1 + v2).M2())

    @staticmethod
    def boost(v: Any, b: Any) -> Any:
        """``boost(v, b)``: ``v`` boosted by the velocity ``b``, a three-vector."""
        x = b._cartesian()
        return _boosted(v, x[0], x[1], x[2])

    @staticmethod
    def Perp(v: Any, u: Any) -> float:
        a, b = v._cartesian()[:3], u._cartesian()[:3]
        along = sum(p * q for p, q in zip(a, b)) / _r3(b)
        return math.sqrt(max(_r3(a) ** 2 - along * along, 0.0))
