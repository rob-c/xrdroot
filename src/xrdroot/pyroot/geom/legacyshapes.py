"""The shapes of ROOT's first geometry package: ``TBRIK``, ``TTUBE``, ``TPCON`` and the rest.

Each is ``(name, title, material, parameters...)`` - GEANT3's names and
numbers - listed with the current ``TGeometry`` and found by name when a
``TNode`` is made of it. Their surfaces are the same solids the ``TGeo``
shapes are (:mod:`xrdroot.geom.shapes`), tessellated in 20 steps round a
circle, as the old package's ``fNdiv`` is by default.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ...geom import shapes as build
from ...geom.mesh import Mesh
from ..core.objects import TAttFill, TAttLine, TNamed

__all__ = [
    "TShape", "TBRIK", "TTRD1", "TTRD2", "TTRAP", "TPARA", "TGTRA", "TTUBE", "TTUBS", "TCONE",
    "TCONS", "TSPHE", "TELTU", "TCTUB", "TPCON", "TPGON", "TXTRU",
]  # fmt: skip

#: The steps round a circle the old shapes are drawn in: ``fNdiv``'s default.
DIVISIONS = 20


class TShape(TNamed, TAttLine, TAttFill):
    """``TShape(name, title, material)``: a solid of the old package, with its numbers."""

    #: The constructor's numbers after the material, in order.
    PARAMETERS: ClassVar[tuple[str, ...]] = ()
    #: The values of those left out of a constructor call.
    DEFAULTS: ClassVar[tuple[float, ...]] = ()
    #: What makes the surface from those numbers.
    BUILD: ClassVar[Callable[..., Mesh] | None] = None

    def __init__(self, name: Any = "", title: Any = "", material: Any = "", *values: Any) -> None:
        super().__init__(str(name), str(title))
        count = len(self.PARAMETERS)
        given = [float(v) for v in values[:count]]
        given += list(self.DEFAULTS[len(given) - count:]) if len(given) < count else []
        self._values, self._material, self._visibility = given, material, 1
        from .legacy import current_geometry

        self._number = current_geometry().add("shapes", self)

    def __getattr__(self, name: str) -> Any:
        wanted = [f"Get{p}" for p in type(self).PARAMETERS]
        if name in wanted:
            at = wanted.index(name)
            return lambda: self.__dict__["_values"][at]
        raise AttributeError(f"ROOT's {type(self).__name__} has {name}; "
                             f"xrdroot.pyroot's does not yet")  # fmt: skip

    def _mesh(self) -> Mesh:
        builder = type(self).BUILD
        assert builder is not None  # every shape class names its own
        return builder(*self._values, **self._keywords())

    def _keywords(self) -> dict[str, Any]:
        return {"steps": DIVISIONS}

    def mesh(self) -> Mesh:
        return self._mesh()

    def GetVisibility(self) -> int:
        return self._visibility

    def SetVisibility(self, visibility: int = 1) -> None:
        self._visibility = int(visibility)

    def GetNumber(self) -> int:
        return self._number

    def GetMaterial(self) -> Any:
        return self._material


class _Flat(TShape):
    """A shape of flat faces: its builder takes no steps round a circle."""

    def _keywords(self) -> dict[str, Any]:
        return {}


class TBRIK(_Flat):
    """``TBRIK(name, title, material, dx, dy, dz)``: a box."""

    PARAMETERS = ("Dx", "Dy", "Dz")
    BUILD = build.box


class TTRD1(_Flat):
    PARAMETERS = ("Dx1", "Dx2", "Dy", "Dz")
    BUILD = build.trd1


class TTRD2(_Flat):
    PARAMETERS = ("Dx1", "Dx2", "Dy1", "Dy2", "Dz")
    BUILD = build.trd2


class TTRAP(_Flat):
    PARAMETERS = ("Dz", "Theta", "Phi", "H1", "Bl1", "Tl1", "Alpha1", "H2", "Bl2", "Tl2", "Alpha2")
    BUILD = build.trap


class TPARA(_Flat):
    PARAMETERS = ("Dx", "Dy", "Dz", "Alpha", "Theta", "Phi")
    BUILD = build.para


class TGTRA(_Flat):
    PARAMETERS = ("Dz", "Theta", "Phi", "Twist", "H1", "Bl1", "Tl1", "Alpha1", "H2", "Bl2", "Tl2",
                  "Alpha2")  # fmt: skip
    BUILD = build.old_gtra


class TTUBE(TShape):
    """``TTUBE(name, title, material, rmin, rmax, dz[, aspect])``: a tube, elliptic by aspect."""

    PARAMETERS: ClassVar[tuple[str, ...]] = ("Rmin", "Rmax", "Dz", "AspectRatio")
    DEFAULTS: ClassVar[tuple[float, ...]] = (1.0,)

    def _mesh(self) -> Mesh:
        rmin, rmax, dz, aspect = self._values
        mesh = build.tube(rmin, rmax, dz, steps=DIVISIONS)
        mesh.points[:, 1] *= aspect
        return mesh


class TTUBS(TShape):
    PARAMETERS = ("Rmin", "Rmax", "Dz", "Phi1", "Phi2")
    BUILD = build.tube


class TCONE(TShape):
    PARAMETERS: ClassVar[tuple[str, ...]] = ("Dz", "Rmin", "Rmax", "Rmin2", "Rmax2")
    BUILD = build.cone


class TCONS(TCONE):
    PARAMETERS = ("Dz", "Rmin", "Rmax", "Rmin2", "Rmax2", "Phi1", "Phi2")


class TSPHE(TShape):
    PARAMETERS = ("Rmin", "Rmax", "ThemMin", "ThemMax", "PhiMin", "PhiMax")
    BUILD = build.sphere


class TELTU(TShape):
    PARAMETERS = ("Rx", "Ry", "Dz")
    BUILD = build.eltu


class TCTUB(TShape):
    PARAMETERS = ("Rmin", "Rmax", "Dz", "Phi1", "Phi2", "CosLowX", "CosLowY", "CosLowZ",
                  "CosHighX", "CosHighY", "CosHighZ")  # fmt: skip

    def _mesh(self) -> Mesh:
        v = self._values
        return build.ctub(v[0], v[1], v[2], v[3], v[4], v[5:8], v[8:11], DIVISIONS)


class TPCON(TShape):
    """``TPCON(name, title, material, phi1, dphi, nz)``: a polycone, by ``DefineSection``."""

    PARAMETERS: ClassVar[tuple[str, ...]] = ("Phi1", "Dhi1", "Nz")

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._sections = [[0.0, 0.0, 0.0] for _ in range(int(self._values[-1]))]

    def DefineSection(self, index: int, z: float, rmin: float, rmax: float) -> None:
        self._sections[int(index)] = [float(z), float(rmin), float(rmax)]

    def _mesh(self) -> Mesh:
        return build.pcon(self._values[0], self._values[1], self._sections, DIVISIONS)


class TPGON(TPCON):
    """``TPGON(name, title, material, phi1, dphi, ndiv, nz)``: a polycone of ``ndiv`` sides."""

    PARAMETERS = ("Phi1", "Dhi1", "Ndiv", "Nz")

    def _mesh(self) -> Mesh:
        return build.pgon(self._values[0], self._values[1], int(self._values[2]), self._sections)


class TXTRU(TShape):
    """``TXTRU(name, title, material, nxy, nz)``: a polygon extruded through ``nz`` sections."""

    PARAMETERS = ("NXY", "NZ")

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._outline = [(0.0, 0.0)] * int(self._values[0])
        self._sections = [[0.0, 0.0, 0.0, 1.0] for _ in range(int(self._values[1]))]

    def DefineVertex(self, index: int, x: float, y: float) -> None:
        """Vertex ``index`` of the outline, the outline grown to hold it as ROOT grows it."""
        at = int(index)
        self._outline += [(0.0, 0.0)] * (at + 1 - len(self._outline))
        self._outline[at] = (float(x), float(y))

    def DefineSection(self, index: int, z: float, scale: float = 1.0, x0: float = 0.0,
                      y0: float = 0.0) -> None:  # fmt: skip
        """``DefineSection(i, z, scale, x0, y0)`` - the old order, scale before the offset."""
        at = int(index)
        self._sections += [[0.0, 0.0, 0.0, 1.0] for _ in range(at + 1 - len(self._sections))]
        self._sections[at] = [float(z), float(x0), float(y0), float(scale)]

    def _mesh(self) -> Mesh:
        return build.xtru(self._outline, self._sections)
