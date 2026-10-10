"""``TGraph2D`` and ``TGraph2DErrors``: points in space, and the surface through them.

A ``TGraph2D`` is three arrays - where each point is across, up and how
high - and with errors, three more. What it draws, and what ``Interpolate``
says between its points, is the plane over each triangle of the points'
Delaunay triangulation (:mod:`xrdroot.delaunay`); what it draws with is
``GetHistogram()``, a ``TH2D`` of ``fNpx`` by ``fNpy`` bins over the points'
range - widened by ``fMargin`` of it either side - each bin filled at its
centre with the height there, and its z range the points' own.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .delaunay import Delaunay
from .function.attached import listed
from .hist import FILL, LINE, MARKER, Histogram

__all__ = ["GRAPHS2D", "Graph2D"]

#: The classes of points in space this holds, and the error arrays each keeps.
GRAPHS2D: dict[str, tuple[str, ...]] = {"TGraph2D": (), "TGraph2DErrors": ("fEX", "fEY", "fEZ")}
#: ``TGraph2D``'s defaults: the histogram's bins each way, and ``fMinimum``'s "not set".
NPX, UNSET = 40, -1111.0
#: ``TH1::kNoStats``: the histogram a graph draws with has no stats box.
NO_STATS = 1 << 9


def members(name: str = "Graph2D", title: str = "Graph2D", classname: str = "TGraph2D",
            points: int = 0) -> dict[str, Any]:  # fmt: skip
    """A new graph's members, as ROOT's constructors leave them: ``points`` points at zero."""
    made: dict[str, Any] = {
        "TNamed": {"fName": name, "fTitle": title},
        "TAttLine": dict(LINE), "TAttFill": dict(FILL), "TAttMarker": dict(MARKER),
        "fNpoints": points, "fNpx": NPX, "fNpy": NPX, "fMaxIter": 100000,
        "fX": np.zeros(points), "fY": np.zeros(points), "fZ": np.zeros(points),
        "fMinimum": UNSET, "fMaximum": UNSET, "fMargin": 0.0, "fZout": 0.0,
        "fFunctions": [], "fHistogram": None,
    }  # fmt: skip
    for key in GRAPHS2D[classname]:
        made[key] = np.zeros(points)
    return made


class Graph2D:
    """Points in space: ``x``, ``y`` and ``z``, and with errors ``ex``, ``ey`` and ``ez``.

        >>> g = Graph2D.new("g", [0, 1, 0, 1], [0, 0, 1, 1], [0, 1, 1, 2])
        >>> float(g.interpolate(0.5, 0.25))
        0.75
    """

    __slots__ = ("classname", "members", "_delaunay")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        #: ``TGraph2D`` or ``TGraph2DErrors``.
        self.classname = classname
        #: Every member, by ROOT's names.
        self.members = members
        self._delaunay: Delaunay | None = None

    @classmethod
    def new(cls, name: str, x: Any, y: Any, z: Any, *, title: str = "", errors: Any = None) -> Graph2D:
        """A graph of the points given, with ``errors`` - ``(ex, ey, ez)`` - a ``TGraph2DErrors``."""
        classname = "TGraph2D" if errors is None else "TGraph2DErrors"
        made = cls(classname, members(name, title, classname, len(np.reshape(x, -1))))
        made.set_points(x, y, z, errors)
        return made

    def set_points(self, x: Any, y: Any, z: Any, errors: Any = None) -> None:
        """Every point at once - and every error, for a graph that has them."""
        columns = [np.array(v, dtype=np.float64).reshape(-1) for v in (x, y, z)]
        self.members.update(fX=columns[0], fY=columns[1], fZ=columns[2], fNpoints=len(columns[0]))
        for key, values in zip(GRAPHS2D[self.classname], errors or (), strict=False):
            self.members[key] = np.array(values, dtype=np.float64).reshape(-1)
        self.changed()

    def changed(self) -> None:
        """Forget the triangles and the histogram: the points have moved."""
        self._delaunay = None
        self.members["fHistogram"] = None

    @property
    def name(self) -> str:
        return str(self.members["TNamed"]["fName"])

    @property
    def title(self) -> str:
        return str(self.members["TNamed"]["fTitle"])

    def _column(self, key: str) -> Any:
        return np.asarray(self.members[key])[: int(self.members["fNpoints"])]

    @property
    def x(self) -> Any:
        return self._column("fX")

    @property
    def y(self) -> Any:
        return self._column("fY")

    @property
    def z(self) -> Any:
        return self._column("fZ")

    @property
    def errors(self) -> tuple[Any, Any, Any] | None:
        """``(ex, ey, ez)`` for a ``TGraph2DErrors``, ``None`` for a graph without them."""
        if self.classname == "TGraph2D":
            return None
        ex, ey, ez = (self._column(key) for key in GRAPHS2D[self.classname])
        return ex, ey, ez

    def __len__(self) -> int:
        return int(self.members["fNpoints"])

    @property
    def functions(self) -> list[Any]:
        """``GetListOfFunctions``: the fits hung on the graph."""
        return listed(self.members)

    # -- ranges ---------------------------------------------------------------------------

    def extent(self, axis: int, bars: bool = False) -> tuple[float, float]:
        """``GetXmin``/``GetXmax`` and their kin; with ``bars``, ``GetXminE``, reaching the bars."""
        values = (self.x, self.y, self.z)[axis]
        if not len(values):
            return 0.0, 0.0
        found = self.errors
        spread = found[axis] if bars and found is not None else np.zeros(len(values))
        return float(np.min(values - spread)), float(np.max(values + spread))

    # -- the surface ----------------------------------------------------------------------------

    @property
    def delaunay(self) -> Delaunay:
        """The triangles through the points, made when first asked for."""
        if self._delaunay is None:
            self._delaunay = Delaunay(self.x, self.y, self.z, float(self.members["fZout"]))
        return self._delaunay

    def interpolate(self, x: Any, y: Any) -> Any:
        """``Interpolate``: the height of the surface at ``(x, y)`` - zero beyond the points."""
        if len(self) <= 2:
            raise ValueError(
                f"{self.name!r} has {len(self)} points, and a surface is laid over three or more"
            )
        return self.delaunay.interpolate(x, y)

    def _frame_axes(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """The histogram's x and y ranges: the points' and their bars', ``fMargin`` wider."""
        margin = float(self.members["fMargin"])
        found = []
        for axis in (0, 1):
            low, high = self.extent(axis, bars=True)
            width = high - low
            low, high = low - margin * width, high + margin * width
            if abs(high - low) < 1e-9:
                low, high = low - 1.0, high + 1.0
            found.append((low, high))
        return found[0], found[1]

    def histogram(self, empty: bool = False) -> Histogram:
        """``GetHistogram``: the surface sampled at each bin's centre - or, ``empty``, just
        the bins - with the points' z range, or the one set, as its own."""
        held = self.members.get("fHistogram")
        if held is not None and held[0] == empty:
            return held[1]  # type: ignore[no-any-return]
        (x0, x1), (y0, y1) = self._frame_axes()
        npx, npy = int(self.members["fNpx"]), int(self.members["fNpy"])
        made = Histogram.book(self.name, (npx, x0, x1), (npy, y0, y1), title=self.title)
        named = made._core["TNamed"]
        named["fBits"] = int(named.get("fBits", 0)) | NO_STATS
        if not empty:
            xs = x0 + (np.arange(1, npx + 1) - 0.5) * ((x1 - x0) / npx)
            ys = y0 + (np.arange(1, npy + 1) - 0.5) * ((y1 - y0) / npy)
            gx, gy = (grid.ravel() for grid in np.meshgrid(xs, ys, indexing="ij"))
            made.fill(gx, gy, weight=self.interpolate(gx, gy))
        self._limits(made)
        self.members["fHistogram"] = (empty, made)
        return made

    def _limits(self, made: Histogram) -> None:
        """The histogram's z range: the points', the bars' reach, or what was set."""
        low, high = self.extent(2, bars=True)
        for key, value, own in (("fMinimum", low, self.members["fMinimum"]),
                                ("fMaximum", high, self.members["fMaximum"])):  # fmt: skip
            made._core[key] = float(own) if own != UNSET else value

    # -- fitting ----------------------------------------------------------------------------------

    def fit(self, model: Any, option: str = "", range: Any = None, **keywords: Any) -> Any:
        """``TGraph2D::Fit``: a function of two variables fitted to the heights, their z errors -
        and, for a ``TGraph2DErrors``, x and y errors by the effective variance - weighing them."""
        from .fit import fit_object

        return fit_object(self, model, option, range, **keywords)
