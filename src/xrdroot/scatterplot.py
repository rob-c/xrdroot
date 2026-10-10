"""``TScatter`` and ``TScatter2D``: points in a plane or in space, each with a colour and a size.

A scatter plot is a graph of positions with two more numbers a point: one
coloured by the palette between the smallest and the largest of them - or
between the ends of the z axis' range, when one is set - and one sized
between ``fMinMarkerSize`` and ``fMaxMarkerSize``. Its frame is the ``TH2F``
``GetHistogram`` makes: the points' range, ``fMargin`` of it wider either
side, and the axes titled from a title of ``title;x;y;z``. ``logc`` and
``logs`` colour and size the points by the logarithms of their values.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .graph import Graph
from .graph2d import Graph2D
from .hist import FILL, LINE, MARKER, Histogram

__all__ = ["ScatterPlot"]

#: The frame's bins each way, as ``TScatter::GetHistogram`` books them.
FRAME_BINS = 100
#: ROOT's "not set" for a histogram's minimum and maximum, and ``TH1::kNoStats``.
UNSET, NO_STATS = -1111.0, 1 << 9
#: The least a value may be before its logarithm is taken.
TINY = 1e-300


def _span(values: Any, margin: float) -> tuple[float, float]:
    """The points' range along one axis, ``margin`` of it wider at each end."""
    low, high = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
    width = (high - low) or 1.0
    return low - margin * width, high + margin * width


def _column(values: Any) -> Any:
    """A value per point as doubles, or ``None`` where none were given."""
    return None if values is None else np.array(values, dtype=np.float64).reshape(-1)


def _axis_titles(title: str) -> list[str]:
    """``title;x;y;z``: the title and up to three axis titles, blank where not given."""
    parts = [*str(title).split(";"), "", "", ""]
    return parts[:4]


def _logged(values: Any, low: float, high: float) -> tuple[Any, float, float]:
    """The values and their ends as logarithms, each kept above zero first."""
    floor = np.log10(np.maximum([low, high], TINY))
    return np.log10(np.maximum(values, TINY)), float(floor[0]), float(floor[1])


class ScatterPlot:
    """Points with a colour and a size each: ``TScatter`` in a plane, ``TScatter2D`` in space."""

    __slots__ = ("classname", "members")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        self.classname = classname
        self.members = members

    @classmethod
    def new(cls, name: str, x: Any, y: Any, z: Any = None, *, colors: Any = None,
            sizes: Any = None, title: str = "") -> ScatterPlot:  # fmt: skip
        """A scatter plot of the points, in space when ``z`` is given."""
        graph = Graph2D.new(name, x, y, z) if z is not None else Graph.new(name, x, y)
        members: dict[str, Any] = {
            "TNamed": {"fName": name, "fTitle": title}, "TAttLine": dict(LINE),
            "TAttFill": dict(FILL), "TAttMarker": dict(MARKER), "fNpoints": len(graph),
            "fGraph": graph, "fColor": _column(colors), "fSize": _column(sizes),
            "fMaxMarkerSize": 5.0, "fMinMarkerSize": 1.0, "fMargin": 0.1, "fHistogram": None,
        }  # fmt: skip
        return cls("TScatter2D" if z is not None else "TScatter", members)

    @property
    def name(self) -> str:
        return str(self.members["TNamed"]["fName"])

    @property
    def title(self) -> str:
        return str(self.members["TNamed"]["fTitle"])

    @property
    def graph(self) -> Any:
        return self.members["fGraph"]

    @property
    def x(self) -> Any:
        return self.graph.x

    @property
    def y(self) -> Any:
        return self.graph.y

    @property
    def z(self) -> Any:
        """The points' heights, for a scatter plot in space; ``None`` in a plane."""
        return self.graph.z if self.classname == "TScatter2D" else None

    @property
    def colors(self) -> Any:
        return self.members.get("fColor")

    @property
    def sizes(self) -> Any:
        return self.members.get("fSize")

    def __len__(self) -> int:
        return int(self.members["fNpoints"])

    # -- the frame, the colours and the sizes ---------------------------------------------------

    def frame(self) -> Histogram:
        """``GetHistogram``: the ``TH2F`` the axes are drawn with, made once over the points'
        range widened by ``fMargin`` each way, titled from ``title;x;y;z``."""
        held = self.members.get("fHistogram")
        if isinstance(held, Histogram):
            return held
        margin = float(self.members.get("fMargin", 0.1))
        name, xlabel, ylabel, zlabel = _axis_titles(self.title)
        made = Histogram.book(
            f"{self.name}_h", (FRAME_BINS, *_span(self.x, margin)),
            (FRAME_BINS, *_span(self.y, margin)), title=f"{name};{xlabel};{ylabel};{zlabel}",
        )  # fmt: skip
        made._core["TNamed"]["fBits"] = int(made._core["TNamed"].get("fBits", 0)) | NO_STATS
        made._core["fZaxis"]["TNamed"]["fTitle"] = zlabel
        self.members["fHistogram"] = made
        return made

    def frame3d(self) -> Histogram:
        """The box a scatter plot in space stands in: the frame, with the z range the heights'."""
        made = self.frame()
        low, high = _span(self.z, float(self.members.get("fMargin", 0.1)))
        made._core["fMinimum"], made._core["fMaximum"] = low, high
        return made

    def colour_scale(self) -> tuple[float, float]:
        """The ends the colours run between: the z axis' range when one was set on the frame
        - its minimum and maximum, as ``SetRangeUser`` on a ``TH2``'s z axis sets them - else
        the smallest and largest colour values."""
        colors = self.colors
        if colors is None or not len(colors):
            return (0.0, 1.0)
        low, high = UNSET, UNSET
        if self.z is None:  # in space the frame's z range is the heights', not the colours'
            core = self.frame()._core
            low, high = float(core.get("fMinimum", UNSET)), float(core.get("fMaximum", UNSET))
        return (float(colors.min()) if low == UNSET else low,
                float(colors.max()) if high == UNSET else high)  # fmt: skip

    def colour_fractions(self) -> Any:
        """Where each point's colour value lies along the scale, 0 to 1, or ``None`` for
        points coloured all alike; the logarithm's way along for ``logc``."""
        colors = self.colors
        if colors is None:
            return None
        values, (low, high) = np.asarray(colors, dtype=np.float64), self.colour_scale()
        if self.members.get("fLogC"):
            values, low, high = _logged(values, low, high)
        return np.clip((values - low) / ((high - low) or 1.0), 0.0, 1.0)

    def marker_sizes(self) -> Any:
        """Each point's marker size, from ``fMinMarkerSize`` to ``fMaxMarkerSize`` as its size
        value lies between the smallest and the largest; the marker's own without sizes."""
        sizes = self.sizes
        if sizes is None:
            return np.full(len(self), float(self.members["TAttMarker"]["fMarkerSize"]))
        values = np.asarray(sizes, dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        if self.members.get("fLogS"):
            values, low, high = _logged(values, low, high)
        least, most = float(self.members["fMinMarkerSize"]), float(self.members["fMaxMarkerSize"])
        return least + (most - least) * (values - low) / ((high - low) or 1.0)
