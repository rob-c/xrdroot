"""``TGraph2D`` and ``TGraph2DErrors``: points in space, by ROOT's methods.

The points live in the :class:`xrdroot.Graph2D` this stands for, grown a
point at a time as ``SetPoint`` past the end grows ROOT's; what it draws and
interpolates is that graph's Delaunay surface, and its fits are
``TGraph2D::Fit``'s - the fitted function hung on the graph afterwards.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np

from ...graph2d import GRAPHS2D, UNSET, Graph2D, members
from .objects import TAttFill, TAttLine, TAttMarker, TNamed
from .refs import store
from .wrapping import adopt, register, wrap

__all__ = ["TGraph2D", "TGraph2DErrors"]


def _grown(values: Any, count: int) -> np.ndarray[Any, Any]:
    """``values`` padded with zeros to ``count``."""
    held = np.asarray(values, dtype=np.float64)
    return np.concatenate([held, np.zeros(max(count - len(held), 0))])


def _double(value: Any) -> float:
    """A number as C++ takes it - a ``ctypes.c_double`` by its value, as PyROOT passes one."""
    return float(np.asarray(value, dtype=np.float64))


class TGraph2D(TNamed, TAttLine, TAttFill, TAttMarker):
    """``TGraph2D``: points in space, and the surface through them."""

    CLASS_TITLE = "Graph 2D graphics class"

    def __init__(self, *args: Any) -> None:
        classname = type(self).__name__ if type(self).__name__ in GRAPHS2D else "TGraph2D"
        self._xrd = Graph2D(classname, members(classname=classname))
        TNamed.__init__(self, "Graph2D", "Graph2D")
        self._construct(args)

    def _adopted(self, xrd: Any) -> None:
        self._xrd = xrd
        TNamed.__init__(self, xrd.name, xrd.title)

    def _construct(self, args: tuple[Any, ...]) -> None:
        """``()``, ``(n)``, ``(n, x, y, z[, ex, ey, ez])``, and the name and title first."""
        if args and isinstance(args[0], str):
            self.SetNameTitle(args[0], args[1] if len(args) > 1 else "")
            args = args[2:]
        if not args:
            return
        if np.ndim(args[0]) > 0:
            args = (len(args[0]), *args)
        count = int(args[0])
        given = [np.asarray(a, dtype=np.float64)[:count] if a is not None else np.zeros(count)
                 for a in args[1:]]  # fmt: skip
        while len(given) < 3 + len(GRAPHS2D[self._xrd.classname]):
            given.append(np.zeros(count))
        self._xrd.set_points(*given[:3], tuple(given[3:]) or None)

    def _attribute_holder(self) -> Any:
        return self._xrd.members

    def ClassName(self) -> str:
        return self._xrd.classname

    def SetName(self, name: Any) -> None:
        TNamed.SetName(self, name)
        self._xrd.members["TNamed"]["fName"] = str(name)

    def SetTitle(self, title: Any = "") -> None:
        TNamed.SetTitle(self, title)
        self._xrd.members["TNamed"]["fTitle"] = str(title)

    # -- points ------------------------------------------------------------------------------

    def GetN(self) -> int:
        return len(self._xrd)

    def Set(self, n: int) -> None:
        """``Set(n)``: ``n`` points, those past the old end at zero."""
        row = self._xrd.members
        for key in ("fX", "fY", "fZ", *GRAPHS2D[self._xrd.classname]):
            row[key] = _grown(np.asarray(row[key])[: int(n)], int(n))
        row["fNpoints"] = int(n)
        self._xrd.changed()

    def SetPoint(self, i: int, x: float, y: float, z: float) -> None:
        """``SetPoint(i, x, y, z)``: point ``i``, the graph grown to hold it."""
        index = int(i)
        if index < 0:
            return
        if index >= self.GetN():
            self._grow(index + 1)
        row = self._xrd.members
        row["fX"][index], row["fY"][index], row["fZ"][index] = _double(x), _double(y), _double(z)
        self._xrd.changed()

    def _grow(self, count: int) -> None:
        row = self._xrd.members
        for key in ("fX", "fY", "fZ", *GRAPHS2D[self._xrd.classname]):
            held = np.asarray(row[key])[: self.GetN()]
            row[key] = _grown(held, max(count, 2 * len(held)))
        row["fNpoints"] = count

    def AddPoint(self, x: float, y: float, z: float) -> None:
        self.SetPoint(self.GetN(), x, y, z)

    def GetPoint(self, i: int, x: Any = None, y: Any = None, z: Any = None) -> int:
        """``GetPoint(i, x, y, z)``: the point, into ``x``, ``y`` and ``z``; ``-1`` past the end."""
        if not 0 <= int(i) < self.GetN():
            return -1
        for into, values in ((x, self._xrd.x), (y, self._xrd.y), (z, self._xrd.z)):
            store(into, float(values[int(i)]))
        return int(i)

    def GetX(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fX"])

    def GetY(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fY"])

    def GetZ(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fZ"])

    def GetPointX(self, i: int) -> float:
        return float(self._xrd.x[int(i)])

    def GetPointY(self, i: int) -> float:
        return float(self._xrd.y[int(i)])

    def GetPointZ(self, i: int) -> float:
        return float(self._xrd.z[int(i)])

    def GetXmin(self) -> float:
        return self._xrd.extent(0)[0]

    def GetXmax(self) -> float:
        return self._xrd.extent(0)[1]

    def GetYmin(self) -> float:
        return self._xrd.extent(1)[0]

    def GetYmax(self) -> float:
        return self._xrd.extent(1)[1]

    def GetZmin(self) -> float:
        return self._xrd.extent(2)[0]

    def GetZmax(self) -> float:
        return self._xrd.extent(2)[1]

    def GetXminE(self) -> float:
        return self._xrd.extent(0, bars=True)[0]

    def GetXmaxE(self) -> float:
        return self._xrd.extent(0, bars=True)[1]

    def GetYminE(self) -> float:
        return self._xrd.extent(1, bars=True)[0]

    def GetYmaxE(self) -> float:
        return self._xrd.extent(1, bars=True)[1]

    def GetZminE(self) -> float:
        return self._xrd.extent(2, bars=True)[0]

    def GetZmaxE(self) -> float:
        return self._xrd.extent(2, bars=True)[1]

    # -- the surface and its histogram ----------------------------------------------------------

    def _set(self, key: str, value: Any) -> None:
        self._xrd.members[key] = value
        self._xrd.changed()

    def SetNpx(self, npx: int = 40) -> None:
        """``SetNpx``: the histogram's bins across - at least four, as ROOT keeps them."""
        self._set("fNpx", max(int(npx), 4))

    def SetNpy(self, npy: int = 40) -> None:
        self._set("fNpy", max(int(npy), 4))

    def GetNpx(self) -> int:
        return int(self._xrd.members["fNpx"])

    def GetNpy(self) -> int:
        return int(self._xrd.members["fNpy"])

    def SetMargin(self, margin: float = 0.1) -> None:
        """``SetMargin``: how much of the points' range the histogram reaches past them."""
        self._set("fMargin", min(max(float(margin), 0.0), 1.0))

    def GetMargin(self) -> float:
        return float(self._xrd.members["fMargin"])

    def SetMaxIter(self, n: int = 100000) -> None:
        self._xrd.members["fMaxIter"] = int(n)

    def SetMinimum(self, minimum: float = UNSET) -> None:
        self._set("fMinimum", float(minimum))

    def SetMaximum(self, maximum: float = UNSET) -> None:
        self._set("fMaximum", float(maximum))

    def GetMinimum(self) -> float:
        return float(self._xrd.members["fMinimum"])

    def GetMaximum(self) -> float:
        return float(self._xrd.members["fMaximum"])

    def Interpolate(self, x: float, y: float) -> float:
        """``Interpolate``: the Delaunay surface's height at ``(x, y)``, zero beyond the points."""
        if self.GetN() <= 2:
            self.Error("Interpolate", "You need at least 3 points to interpolate")
            return 0.0
        return float(self._xrd.interpolate(float(x), float(y)))

    def GetHistogram(self, option: str = "") -> Any:
        """``GetHistogram``: the ``TH2D`` the graph draws with - ``"empty"``, just its bins."""
        return wrap(self._xrd.histogram(empty="empty" in str(option).lower()))

    def GetXaxis(self) -> Any:
        return self.GetHistogram().GetXaxis()

    def GetYaxis(self) -> Any:
        return self.GetHistogram().GetYaxis()

    def GetZaxis(self) -> Any:
        return self.GetHistogram().GetZaxis()

    # -- fits ----------------------------------------------------------------------------------

    def Fit(self, f2: Any, option: str = "", goption: str = "") -> Any:
        """``Fit``: as ``TGraph2D::Fit``, the function hung on the graph afterwards."""
        from .fits import fit

        return fit(self, f2, option, goption, (0.0, 0.0))

    def GetListOfFunctions(self) -> Any:
        from .collections import FunctionList

        return FunctionList(self._xrd.functions, self.__dict__.setdefault("_extras", []))

    def FindObject(self, name: Any) -> Any:
        """``FindObject``: a function hung on the graph by its name."""
        return self.GetListOfFunctions().FindObject(name)

    def GetFunction(self, name: Any) -> Any:
        return self.FindObject(name)

    def Clone(self, newname: str = "") -> Any:
        import copy

        made = adopt(type(self), copy.deepcopy(self._xrd))
        if newname:
            made.SetName(newname)
        return made


class TGraph2DErrors(TGraph2D):
    """``TGraph2DErrors``: points in space with an error each way."""

    CLASS_TITLE = "Graph 2D graphics class with errors"

    def SetPointError(self, i: int, ex: float, ey: float, ez: float) -> None:
        """``SetPointError(i, ex, ey, ez)``: point ``i``'s errors, the graph grown to hold it."""
        index = int(i)
        if index < 0:
            return
        if index >= self.GetN():
            self._grow(index + 1)
        row = self._xrd.members
        row["fEX"][index], row["fEY"][index], row["fEZ"][index] = (_double(ex), _double(ey),
                                                                   _double(ez))  # fmt: skip
        self._xrd.changed()

    def GetEX(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fEX"])

    def GetEY(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fEY"])

    def GetEZ(self) -> np.ndarray[Any, Any]:
        return np.asarray(self._xrd.members["fEZ"])

    def _error(self, key: str, i: int) -> float:
        return float(self._xrd.members[key][int(i)]) if 0 <= int(i) < self.GetN() else -1.0

    def GetErrorX(self, i: int) -> float:
        return self._error("fEX", i)

    def GetErrorY(self, i: int) -> float:
        return self._error("fEY", i)

    def GetErrorZ(self, i: int) -> float:
        return self._error("fEZ", i)


for _cls in (TGraph2D, TGraph2DErrors):
    register(_cls.__name__, factory=partial(adopt, _cls))
