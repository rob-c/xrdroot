"""``TView``/``TView3D``, ``TPolyLine3D`` and ``TPolyMarker3D``: 3-D in a 2-D pad.

A pad draws 3-D through its view: ``TView::CreateView(1)`` makes a
parallel one, ``CreateView(11)`` - what drawing a geometry makes - a
perspective one, each from the pad's angles (``fLongitude = -90 - phi``,
``fLatitude = 90 - theta``) and each the pad's from then on
(``gPad->GetView()``), until the pad is cleared. A view given no range
(``SetRange``) takes the box round everything 3-D in the pad when it is
painted. The points and lines drawn through it are painted by
:mod:`xrdroot.geom.paint`.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..core.objects import TObject
from .drawn import Drawn

__all__ = ["TView", "TView3D", "TPolyLine3D", "TPolyMarker3D"]

#: ``TView3D``'s coordinate systems: kind 11 is the perspective a geometry is drawn in.
PERSPECTIVE = 11


class TView3D(TObject):
    """``TView3D(system, rmin, rmax)``: how 3-D in the current pad is laid flat."""

    def __init__(self, system: int = 1, rmin: Any = None, rmax: Any = None) -> None:
        from .canvas import default_canvas
        from .pads import current

        super().__init__()
        pad = current() or default_canvas()
        self._system = int(system)
        self._range: list[list[float]] | None = None
        if np.ndim(rmin) > 0 and np.ndim(rmax) > 0:
            self._range = [[float(v) for v in rmin[:3]], [float(v) for v in rmax[:3]]]
        self._angles = [-90.0 - float(pad.members["fPhi"]), 90.0 - float(pad.members["fTheta"]),
                        0.0]  # fmt: skip
        self._perspective = self._system == PERSPECTIVE
        pad.members.update({"fX1": -1.0, "fY1": -1.0, "fX2": 1.0, "fY2": 1.0})
        self._pad = pad
        pad.SetView(self)

    @staticmethod
    def CreateView(system: int = 1, rmin: Any = None, rmax: Any = None) -> TView3D:
        """``TView::CreateView``: a view of kind ``system``, the current pad's."""
        return TView3D(system, rmin, rmax)

    def spec(self) -> dict[str, Any]:
        """The view as :func:`xrdroot.geom.view.make_view` takes it."""
        low, high = self._range if self._range is not None else (None, None)
        longitude, latitude, psi = self._angles
        return {"perspective": self._perspective, "rmin": low, "rmax": high,
                "longitude": longitude, "latitude": latitude, "psi": psi}  # fmt: skip

    def SetRange(self, *args: Any) -> None:
        """``SetRange(x0, y0, z0, x1, y1, z1[, flag])``, or ``SetRange(min, max)``."""
        if len(args) >= 6:
            self._range = [[float(v) for v in args[:3]], [float(v) for v in args[3:6]]]
        else:
            self._range = [[float(v) for v in args[0][:3]], [float(v) for v in args[1][:3]]]

    def _found_range(self) -> list[list[float]]:
        """The range set, or - with none - the box round everything 3-D in the pad."""
        if self._range is not None:
            return self._range
        from ...canvas import Primitive
        from ...geom.paint import points_of

        prims = [Primitive(obj.classname, dict(obj.members)) if isinstance(obj, Drawn)
                 else getattr(obj, "_xrd", None) for obj, _ in self._pad.primitives]  # fmt: skip
        points = [points_of(prim) for prim in prims]
        every = np.concatenate(points) if points else np.zeros((0, 3))
        if not len(every):
            return [[0.0] * 3, [1.0] * 3]
        return [list(every.min(axis=0)), list(every.max(axis=0))]

    def GetRange(self, low: Any, high: Any) -> None:
        """``GetRange(min, max)``: the range, into the two arrays given."""
        found = self._found_range()
        for i in range(3):
            low[i], high[i] = found[0][i], found[1][i]

    def GetRmin(self) -> np.ndarray[Any, Any]:
        return np.array(self._found_range()[0])

    def GetRmax(self) -> np.ndarray[Any, Any]:
        return np.array(self._found_range()[1])

    def SetAutoRange(self, autorange: bool = True) -> None:
        if autorange:
            self._range = None

    def SetView(self, longitude: float, latitude: float, psi: float, irep: Any = None) -> None:
        self._angles = [float(longitude), float(latitude), float(psi)]
        if irep is not None and hasattr(irep, "value"):
            irep.value = 0

    def RotateView(self, phi: float, theta: float, pad: Any = None) -> None:
        """``RotateView``: look from longitude ``phi`` and latitude ``theta``, the pad's too."""
        from .pads import current

        self.SetView(phi, theta, 0.0)
        target = pad if pad is not None else current()
        if target is not None:
            target.members["fPhi"], target.members["fTheta"] = -90.0 - phi, 90.0 - theta

    def SideView(self, pad: Any = None) -> None:
        self.RotateView(0.0, 90.0, pad)

    def FrontView(self, pad: Any = None) -> None:
        self.RotateView(270.0, 90.0, pad)

    def TopView(self, pad: Any = None) -> None:
        self.RotateView(270.0, 0.0, pad)

    Side, Front, Top = SideView, FrontView, TopView

    def GetLongitude(self) -> float:
        return self._angles[0]

    def GetLatitude(self) -> float:
        return self._angles[1]

    def GetPsi(self) -> float:
        return self._angles[2]

    def SetParallel(self) -> None:
        self._perspective = False

    def SetPerspective(self) -> None:
        self._perspective = True

    def IsPerspective(self) -> bool:
        return self._perspective

    def GetSystem(self) -> int:
        return self._system

    def ShowAxis(self) -> None:
        """``ShowAxis``: the 3-D rulers are not drawn here."""


TView = TView3D


def _points(n: int, given: tuple[Any, ...]) -> np.ndarray[Any, Any]:
    """``n`` points, flat ``x, y, z, x, y, z...``: from ``p[3n]``, from ``x, y, z``, or zeros."""
    arrays = [value for value in given if np.ndim(value) > 0]
    if len(arrays) >= 3:
        xyz = [np.asarray(a, dtype=np.float64)[:n] for a in arrays[:3]]
        return np.column_stack(xyz).reshape(-1)
    if arrays:
        return np.asarray(arrays[0], dtype=np.float64)[: 3 * n].copy()
    return np.zeros(3 * n)


class _Points3D(Drawn):
    """Points in space, kept flat in ``fP`` as ROOT keeps them."""

    def _start(self, n: int, given: tuple[Any, ...], option: str) -> None:
        p = _points(int(n), given)
        self.members.update(fN=len(p) // 3, fP=p, fOption=str(option), fLastPoint=len(p) // 3 - 1)

    def SetPoint(self, i: int, x: float, y: float, z: float) -> None:
        """Point ``i``, the list grown to hold it if it is past the end."""
        index = int(i)
        p = self.members["fP"]
        if 3 * index + 3 > len(p):
            p = np.append(p, np.zeros(3 * index + 3 - len(p)))
        p[3 * index : 3 * index + 3] = (float(x), float(y), float(z))
        self.members.update(fP=p, fN=max(int(self.members["fN"]), index + 1),
                            fLastPoint=max(int(self.members["fLastPoint"]), index))  # fmt: skip

    def SetNextPoint(self, x: float, y: float, z: float) -> int:
        self.SetPoint(int(self.members["fLastPoint"]) + 1, x, y, z)
        return int(self.members["fLastPoint"])

    def GetN(self) -> int:
        return int(self.members["fN"])

    def GetP(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.members["fP"])

    def GetLastPoint(self) -> int:
        return int(self.members["fLastPoint"])


class TPolyLine3D(_Points3D):
    """``TPolyLine3D(n[, p | x, y, z][, option])``: points in space, joined."""

    classname = "TPolyLine3D"
    groups: ClassVar[tuple[str, ...]] = ("line",)

    def __init__(self, n: int = 0, *given: Any) -> None:
        super().__init__()
        option = next((value for value in given if isinstance(value, str)), "")
        self._start(n, given, option)

    def SetPolyLine(self, n: int, *given: Any) -> None:
        self._start(n, given, str(self.members["fOption"]))


class TPolyMarker3D(_Points3D):
    """``TPolyMarker3D(n[, p | x, y, z][, marker][, option])``: points in space, each marked."""

    classname = "TPolyMarker3D"
    groups: ClassVar[tuple[str, ...]] = ("marker",)

    def __init__(self, n: int = 0, *given: Any) -> None:
        super().__init__()
        option = next((value for value in given if isinstance(value, str)), "")
        self._start(n, given, option)
        marker = [value for value in given if isinstance(value, (int, np.integer))]
        if marker:
            self.SetMarkerStyle(int(marker[0]))

    def SetPolyMarker(self, n: int, *given: Any) -> None:
        self._start(n, given, str(self.members["fOption"]))
