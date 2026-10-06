"""The ``TGeo`` solids: each ROOT's parameters, its box, and its surface to draw.

Every shape keeps the numbers ROOT's constructor takes - a name first if
one is given - answers ROOT's getters for them, and hands
:mod:`xrdroot.geom.shapes` those numbers for its surface, tessellated with
the current geometry's ``GetNsegments()``. Its bounding box (``GetDX``,
``GetDY``, ``GetDZ``, ``GetOrigin``) is that surface's extent, as
``ComputeBBox`` would find it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from ...geom import shapes as build
from ...geom.mesh import Mesh
from ..core.objects import TNamed

__all__ = [
    "TGeoShape", "TGeoBBox", "TGeoArb8", "TGeoTrd1", "TGeoTrd2", "TGeoPara", "TGeoTrap",
    "TGeoGtra", "TGeoTube", "TGeoTubeSeg", "TGeoCtub", "TGeoCone", "TGeoConeSeg",
    "TGeoSphere", "TGeoTorus", "TGeoEltu", "TGeoParaboloid", "TGeoHype",
]  # fmt: skip


def segments() -> int:
    """``gGeoManager->GetNsegments()``: the steps a circle is drawn in."""
    from .manager import current_manager

    return int(current_manager().GetNsegments())


class TGeoShape(TNamed):
    """``TGeoShape``: a solid, its parameters in ``PARAMETERS`` order, named or not."""

    #: The constructor's numbers, in its order.
    PARAMETERS: ClassVar[tuple[str, ...]] = ()
    #: The values of those left out of a constructor call.
    DEFAULTS: ClassVar[tuple[float, ...]] = ()

    def __init__(self, *args: Any) -> None:
        named = bool(args) and isinstance(args[0], str)
        super().__init__(args[0] if named else "", "")
        values = list(args[1:] if named else args)
        count = len(self.PARAMETERS)
        values += list(self.DEFAULTS[len(values) - count:]) if len(values) < count else []
        self._values = [float(v) for v in values[:count]]
        self._extra = values[count:]
        from .manager import current_manager

        current_manager().AddShape(self)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("Get") and name[3:].lower() in _lowered(type(self).PARAMETERS):
            at = _lowered(type(self).PARAMETERS).index(name[3:].lower())
            return lambda: self.__dict__["_values"][at]
        raise AttributeError(f"ROOT's {type(self).__name__} has {name}; "
                             f"xrdroot.pyroot's does not yet")  # fmt: skip

    def _mesh(self, steps: int) -> Mesh:
        raise NotImplementedError

    def mesh(self) -> Mesh:
        """The surface, tessellated as the current geometry says."""
        return self._mesh(segments())

    def _box(self) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        low, high = self._mesh(max(segments(), 360)).extent()
        return 0.5 * (high - low), 0.5 * (high + low)

    def GetDX(self) -> float:
        return float(self._box()[0][0])

    def GetDY(self) -> float:
        return float(self._box()[0][1])

    def GetDZ(self) -> float:
        return float(self._box()[0][2])

    def GetOrigin(self) -> np.ndarray[Any, Any]:
        return self._box()[1]

    def ComputeBBox(self) -> None:
        """``ComputeBBox``: the box is found from the surface whenever it is asked for."""

    def GetNsegments(self) -> int:
        return segments()

    def IsComposite(self) -> bool:
        return False

    def Draw(self, option: str = "") -> None:
        """``Draw``: the shape alone, in a volume of its own, as ``TGeoPainter::DrawShape``."""
        from .volumes import TGeoVolume

        TGeoVolume(self.GetName() or type(self).__name__, self).Draw(option)


def _built(builder: Callable[..., Mesh], values: list[float], **keywords: Any) -> Mesh:
    """``builder``'s surface from a shape's numbers, in its constructor's order."""
    return builder(*values, **keywords)


def _lowered(names: tuple[str, ...]) -> list[str]:
    return [name.lower() for name in names]


class TGeoBBox(TGeoShape):
    """``TGeoBBox(dx, dy, dz[, origin])``: a box of half-sides ``dx, dy, dz``."""

    PARAMETERS = ("DX", "DY", "DZ")

    def _mesh(self, steps: int) -> Mesh:
        origin = self._extra[0] if self._extra and self._extra[0] is not None else (0, 0, 0)
        return _built(build.box, self._values, origin=[float(v) for v in origin[:3]])

    def SetBoxDimensions(self, dx: float, dy: float, dz: float, origin: Any = None) -> None:
        self._values = [float(dx), float(dy), float(dz)]
        self._extra = [origin]


class TGeoArb8(TGeoShape):
    """``TGeoArb8(dz[, vertices])``: eight corners, set with ``SetVertex(i, x, y)``."""

    PARAMETERS = ("Dz",)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        given = self._extra[0] if self._extra else None
        self._corners = np.zeros((8, 2)) if given is None else (
            np.asarray(given, dtype=np.float64)[:16].reshape(8, 2))  # fmt: skip

    def SetVertex(self, index: int, x: float, y: float) -> None:
        self._corners[int(index)] = (float(x), float(y))

    def GetVertices(self) -> np.ndarray[Any, Any]:
        return self._corners.reshape(16)

    def _mesh(self, steps: int) -> Mesh:
        return build.arb8(self._values[0], self._corners)


class TGeoTrd1(TGeoShape):
    """``TGeoTrd1(dx1, dx2, dy, dz)``: half-width in x from ``dx1`` at ``-dz`` to ``dx2``."""

    PARAMETERS = ("Dx1", "Dx2", "Dy", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.trd1, self._values)


class TGeoTrd2(TGeoShape):
    """``TGeoTrd2(dx1, dx2, dy1, dy2, dz)``: half-widths in x and y changing along z."""

    PARAMETERS = ("Dx1", "Dx2", "Dy1", "Dy2", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.trd2, self._values)


class TGeoPara(TGeoShape):
    """``TGeoPara(dx, dy, dz, alpha, theta, phi)``: a parallelepiped."""

    PARAMETERS = ("X", "Y", "Z", "Alpha", "Theta", "Phi")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.para, self._values)


class TGeoTrap(TGeoShape):
    """``TGeoTrap(dz, theta, phi, h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2)``."""

    PARAMETERS = ("Dz", "Theta", "Phi", "H1", "Bl1", "Tl1", "Alpha1", "H2", "Bl2", "Tl2",
                  "Alpha2")  # fmt: skip

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.trap, self._values)


class TGeoGtra(TGeoShape):
    """``TGeoGtra(dz, theta, phi, twist, h1, bl1, tl1, alpha1, h2, bl2, tl2, alpha2)``."""

    PARAMETERS = ("Dz", "Theta", "Phi", "TwistAngle", "H1", "Bl1", "Tl1", "Alpha1", "H2", "Bl2",
                  "Tl2", "Alpha2")  # fmt: skip

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.gtra, self._values)


class TGeoTube(TGeoShape):
    """``TGeoTube(rmin, rmax, dz)``: a cylinder, hollow inside ``rmin``."""

    PARAMETERS: ClassVar[tuple[str, ...]] = ("Rmin", "Rmax", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.tube, self._values, steps=steps)

    def SetTubeDimensions(self, *values: float) -> None:
        self._values[: len(values)] = [float(v) for v in values]


class TGeoTubeSeg(TGeoTube):
    """``TGeoTubeSeg(rmin, rmax, dz, phi1, phi2)``: a tube over ``phi1..phi2`` degrees."""

    PARAMETERS = ("Rmin", "Rmax", "Dz", "Phi1", "Phi2")


class TGeoCtub(TGeoShape):
    """``TGeoCtub(rmin, rmax, dz, phi1, phi2, lx, ly, lz, tx, ty, tz)``: a tube cut by planes."""

    PARAMETERS = ("Rmin", "Rmax", "Dz", "Phi1", "Phi2", "Nlowx", "Nlowy", "Nlowz", "Nhighx",
                  "Nhighy", "Nhighz")  # fmt: skip

    def _mesh(self, steps: int) -> Mesh:
        v = self._values
        return build.ctub(v[0], v[1], v[2], v[3], v[4], v[5:8], v[8:11], steps)


class TGeoCone(TGeoShape):
    """``TGeoCone(dz, rmin1, rmax1, rmin2, rmax2)``: radii changing from ``-dz`` to ``+dz``."""

    PARAMETERS: ClassVar[tuple[str, ...]] = ("Dz", "Rmin1", "Rmax1", "Rmin2", "Rmax2")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.cone, self._values, steps=steps)


class TGeoConeSeg(TGeoCone):
    """``TGeoConeSeg(dz, rmin1, rmax1, rmin2, rmax2, phi1, phi2)``."""

    PARAMETERS = ("Dz", "Rmin1", "Rmax1", "Rmin2", "Rmax2", "Phi1", "Phi2")


class TGeoSphere(TGeoShape):
    """``TGeoSphere(rmin, rmax[, theta1, theta2, phi1, phi2])``: a shell, perhaps cut."""

    PARAMETERS = ("Rmin", "Rmax", "Theta1", "Theta2", "Phi1", "Phi2")
    DEFAULTS = (0.0, 180.0, 0.0, 360.0)

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.sphere, self._values, steps=steps)


class TGeoTorus(TGeoShape):
    """``TGeoTorus(r, rmin, rmax[, phi1, dphi])``: a tube bent into a ring."""

    PARAMETERS = ("R", "Rmin", "Rmax", "Phi1", "Dphi")
    DEFAULTS = (0.0, 360.0)

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.torus, self._values, steps=steps)


class TGeoEltu(TGeoShape):
    """``TGeoEltu(a, b, dz)``: a cylinder of elliptic section."""

    PARAMETERS = ("A", "B", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.eltu, self._values, steps=steps)


class TGeoParaboloid(TGeoShape):
    """``TGeoParaboloid(rlo, rhi, dz)``: a paraboloid of revolution, cut at ``+-dz``."""

    PARAMETERS = ("Rlo", "Rhi", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.paraboloid, self._values, steps=steps)


class TGeoHype(TGeoShape):
    """``TGeoHype(rin, stin, rout, stout, dz)``: a tube with hyperbolic walls."""

    PARAMETERS = ("Rmin", "StIn", "Rmax", "StOut", "Dz")

    def _mesh(self, steps: int) -> Mesh:
        return _built(build.hype, self._values, steps=steps)
