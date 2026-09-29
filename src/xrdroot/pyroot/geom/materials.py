"""``TGeoElement``, ``TGeoMaterial``, ``TGeoMixture`` and ``TGeoMedium``: what a volume is made of.

A material is an atomic mass, an atomic number and a density; a mixture is
several elements by weight; a medium is a material with a number and the
tracking parameters a transport code reads. Each is listed with the
current geometry as it is made - making the first one makes a default
``TGeoManager``, as ROOT does - and indexed in the order it was listed.
The radiation and interaction lengths are those given; ROOT's own
formulae for them, which only a transport code reads, are not worked out.
"""

from __future__ import annotations

from typing import Any

from ..core.objects import TAttFill, TNamed

__all__ = ["TGeoElement", "TGeoMaterial", "TGeoMixture", "TGeoMedium"]

#: The value ROOT gives the lengths of a vacuum.
BIG = 1.0e30


def _manager() -> Any:
    from .manager import current_manager

    return current_manager()


class TGeoElement(TNamed):
    """``TGeoElement(name, title, z, a)``: an element of the periodic table."""

    def __init__(self, name: Any = "", title: Any = "", z: int = 0, a: float = 0.0,
                 *_: Any) -> None:  # fmt: skip
        super().__init__(name, title)
        self._z, self._a = int(z), float(a)

    def Z(self) -> int:
        return self._z

    def A(self) -> float:
        return self._a

    def Print(self, option: str = "") -> None:
        name, z, n, a = self.GetName(), self._z, float(self.N()), self._a
        print(f"Element: {name}      Z={z}   N={n:f}   A={a:f} [g/mole]")

    def N(self) -> int:
        """The number of nucleons, ``Int_t(a)``, as ROOT's constructor keeps it."""
        return int(self._a)


def _numbers(args: tuple[Any, ...]) -> list[float]:
    """``a, z, rho, radlen, intlen`` from a material's arguments, an element's if it was given."""
    if args and isinstance(args[0], TGeoElement):
        element, *rest = args
        args = (element.A(), element.Z(), *rest)
    return [float(v) for v in args] + [0.0] * (5 - len(args))


def _length(given: float, vacuum: bool) -> float:
    """A radiation or interaction length as ROOT keeps it: huge for a vacuum, else as given."""
    return BIG if vacuum and given >= 0 else abs(given)


class TGeoMaterial(TNamed, TAttFill):
    """``TGeoMaterial(name, a, z, rho[, radlen, intlen])``, or ``(name, element, rho)``."""

    def __init__(self, name: Any = "", *args: Any) -> None:
        super().__init__(str(name).strip(), "")
        self._a, self._z, self._density, radlen, intlen = _numbers(args)[:5]
        vacuum = self._a < 0.9 or self._z < 0.9
        self._radlen, self._intlen = _length(radlen, vacuum), _length(intlen, vacuum)
        self._index = _manager().AddMaterial(self)

    def GetA(self) -> float:
        return self._a

    def GetZ(self) -> float:
        return self._z

    def GetDensity(self) -> float:
        return self._density

    def SetDensity(self, density: float) -> None:
        self._density = float(density)

    def GetRadLen(self) -> float:
        return self._radlen

    def GetIntLen(self) -> float:
        return self._intlen

    def GetIndex(self) -> int:
        return int(self._index)

    def IsMixture(self) -> bool:
        return False

    def SetTransparency(self, transparency: int = 0) -> None:
        self._transparency = int(transparency)

    def GetTransparency(self) -> int:
        return int(getattr(self, "_transparency", 0))


class TGeoMixture(TGeoMaterial):
    """``TGeoMixture(name, nelements, rho)``: elements by weight, added one by one."""

    def __init__(self, name: Any = "", nelements: int = 0, rho: float = 0.0) -> None:
        self._elements: list[tuple[float, float, float]] = []
        super().__init__(name, 0.0, 0.0, rho)

    def AddElement(self, *args: Any) -> None:
        """``AddElement(a, z, weight)``, or ``(element, weight)``."""
        if isinstance(args[0], TGeoElement):
            args = (args[0].A(), args[0].Z(), args[1])
        a, z, weight = (float(v) for v in args[:3])
        self._elements.append((a, z, weight))
        total = sum(w for *_, w in self._elements) or 1.0
        self._a = sum(a * w for a, _, w in self._elements) / total
        self._z = sum(z * w for _, z, w in self._elements) / total

    def DefineElement(self, index: int, a: float, z: float, weight: float) -> None:
        self.AddElement(a, z, weight)

    def GetNelements(self) -> int:
        return len(self._elements)

    def IsMixture(self) -> bool:
        return True


class TGeoMedium(TNamed):
    """``TGeoMedium(name, number, material[, params])``: a material with its tracking numbers."""

    def __init__(self, name: Any = "", numed: int = 0, material: Any = None,
                 params: Any = None) -> None:  # fmt: skip
        super().__init__(str(name).strip(), "")
        self._id = int(numed)
        self._material = material
        self._params = [float(v) for v in (params if params is not None else [])][:10]
        _manager().AddMedium(self)

    def GetId(self) -> int:
        return self._id

    def GetMaterial(self) -> Any:
        return self._material

    def GetParam(self, index: int) -> float:
        return self._params[index] if index < len(self._params) else 0.0
