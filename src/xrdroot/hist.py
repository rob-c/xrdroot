"""Histograms, as objects rather than as the dictionaries they are written as.

A histogram in a ROOT file is a class like any other: a name, a set of drawing
attributes nobody analysing data wants, three axes whether it uses them or not,
and one flat run of numbers holding every bin including the two the axis keeps
for what fell off each end. This turns that into the thing it is - bins, edges
and what is in them - while leaving every member it was written with in reach
under :attr:`Histogram.members`, so nothing is hidden by being tidied away.
"""

from __future__ import annotations

import array
import math
from typing import Any, NamedTuple

import numpy as np

from .draw import axes, bar, missing_picture, shade
from .errors import FormatError, UnsupportedFeatureError

__all__ = ["HISTOGRAMS", "Axis", "Histogram", "Traits"]

#: The histogram classes this reads: one, two and three dimensions, in each of
#: the types ROOT keeps bin contents in.
HISTOGRAMS = tuple(f"TH{dimension}{kind}" for dimension in (1, 2, 3) for kind in "CSILFD")

#: What a freshly made object draws like: ROOT's own defaults, spelled out.
LINE = {"fLineColor": 1, "fLineStyle": 1, "fLineWidth": 1}
FILL = {"fFillColor": 0, "fFillStyle": 1001}
MARKER = {"fMarkerColor": 1, "fMarkerStyle": 1, "fMarkerSize": 1.0}
AXIS_STYLE = {
    "fNdivisions": 510,
    "fAxisColor": 1,
    "fLabelColor": 1,
    "fLabelFont": 42,
    "fLabelOffset": 0.005,
    "fLabelSize": 0.035,
    "fTickLength": 0.03,
    "fTitleOffset": 1.0,
    "fTitleSize": 0.035,
    "fTitleColor": 1,
    "fTitleFont": 42,
}


def _axis(name: str, nbins: int, low: float, high: float, edges: Any) -> dict[str, Any]:
    """The members of one freshly made ``TAxis``."""
    return {
        "TNamed": {"fName": name, "fTitle": ""},
        "TAttAxis": dict(AXIS_STYLE),
        "fNbins": nbins,
        "fXmin": low,
        "fXmax": high,
        "fXbins": np.asarray(edges, dtype=np.float64),
        "fFirst": 0,
        "fLast": 0,
        "fBits2": 0,
        "fTimeDisplay": False,
        "fTimeFormat": "",
        "fLabels": None,
        "fModLabs": None,
    }


class Traits(NamedTuple):
    """What kind of axis this is, in the words the plotting libraries ask in.

    A ROOT axis is neither: it does not wrap round, and its bins are ranges
    rather than labels, even on the axis of a histogram of categories.
    """

    circular: bool = False
    discrete: bool = False


class Axis:
    """One axis of a histogram: how it is binned, and what it is called.

    It is also a sequence of ``(low, high)`` bins, which is the axis half of
    the plotting protocol ``hist``, ``boost-histogram`` and ``mplhep`` all
    share, so any of them takes a :class:`Histogram` read here as it is.
    """

    __slots__ = ("name", "title", "nbins", "low", "high", "_edges")
    #: Neither circular nor discrete, whatever the histogram holds.
    traits = Traits()

    def __init__(self, row: dict[str, Any]) -> None:
        named = row["TNamed"]
        #: What ROOT calls the axis, which is ``xaxis`` unless it was renamed.
        self.name: str = named["fName"]
        #: The label the axis is drawn with, which is where units usually are.
        self.title: str = named["fTitle"]
        #: How many bins there are, not counting the two for what fell off.
        self.nbins: int = row["fNbins"]
        #: The low end of the first bin, and the high end of the last.
        self.low: float = row["fXmin"]
        self.high: float = row["fXmax"]
        self._edges = row["fXbins"]

    def __len__(self) -> int:
        return self.nbins

    def __getitem__(self, index: int) -> tuple[float, float]:
        """The low and high edge of one bin, counting from zero."""
        if index < 0:
            index += self.nbins
        if not 0 <= index < self.nbins:
            raise IndexError(f"bin {index} of an axis of {self.nbins}")
        edges = self.edges()
        return float(edges[index]), float(edges[index + 1])

    def __iter__(self) -> Any:
        edges = self.edges().tolist()
        return iter(zip(edges[:-1], edges[1:]))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Axis):
            return NotImplemented
        return bool(np.array_equal(self.edges(), other.edges()))

    __hash__ = None  # type: ignore[assignment]

    @property
    def label(self) -> str:
        """What to write along the axis: its title, or its name without one."""
        return self.title or self.name

    def edges(self) -> np.ndarray[Any, Any]:
        """The ``nbins + 1`` edges, whether they were written or are worked out.

        An axis binned unevenly keeps every edge; an evenly binned one keeps
        none of them, because the two ends and the count say where they are.
        The inner edges are worked out the way ROOT works them out, one width
        at a time from the low end, so they are the same doubles ROOT has.
        """
        if len(self._edges) == self.nbins + 1:
            return np.array(self._edges, dtype=np.float64)
        width = (self.high - self.low) / self.nbins
        return np.append(self.low + width * np.arange(self.nbins), self.high)

    def centers(self) -> np.ndarray[Any, Any]:
        """The middle of each bin, which is what a point is usually drawn at."""
        edges = self.edges()
        return (edges[:-1] + edges[1:]) / 2

    def widths(self) -> np.ndarray[Any, Any]:
        """How wide each bin is, which a density divides by."""
        return np.diff(self.edges())

    def __repr__(self) -> str:
        return f"<Axis {self.name!r} of {self.nbins} bins from {self.low:g} to {self.high:g}>"


def _core(row: dict[str, Any]) -> dict[str, Any] | None:
    """The ``TH1`` part of a histogram, however far down it is inherited."""
    if "fNcells" in row:
        return row
    for value in row.values():
        if isinstance(value, dict):
            found = _core(value)
            if found is not None:
                return found
    return None


def _contents(row: dict[str, Any]) -> np.ndarray[Any, Any] | None:
    """The bins themselves, which are the array base the class inherits.

    Read from a file they are a NumPy array; members put together by hand may
    hold them as an :class:`array.array`, which is taken the same way. A class
    built on a histogram - a profile is a ``TH1D`` with more on top - keeps
    them one base further down, and they are looked for there too.
    """
    for name, value in row.items():
        if name.startswith("TArray") and isinstance(value, (np.ndarray, array.array)):
            return np.asarray(value)
    for value in row.values():
        if isinstance(value, dict) and "fNcells" not in value:
            found = _contents(value)
            if found is not None:
                return found
    return None


class Histogram:
    """A histogram: its bins, what is in them, and the axes they lie on.

    Bins are counted the way Python counts, from zero and without the two
    ROOT keeps at each end for what fell off it - ``values()[0]`` is the first
    bin of the axis, not the underflow. Pass ``flow=True`` to get those two
    back, at the ends where ROOT keeps them.

    Everything comes back as a NumPy array shaped the way the axes are, x
    first: ``values()[ix, iy]`` is the bin ROOT would call
    ``GetBinContent(ix+1, iy+1)``. The class speaks the plotting protocol
    that ``hist``, ``boost-histogram`` and ``mplhep`` share - ``kind``,
    ``values``, ``variances``, ``counts`` and ``axes`` - so any of those takes
    one as it stands, and :meth:`to_hist` and :meth:`to_numpy` hand it over
    outright.
    """

    __slots__ = ("classname", "members", "axes", "_core", "_bins", "_widths")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        core, bins = _core(members), _contents(members)
        if core is None or bins is None:
            raise FormatError(f"a {classname} was written without its bins or its axes")
        #: The class the file says this is, such as ``TH1D``.
        self.classname = classname
        #: Every member, as it was written, for whatever is not here by name.
        self.members = members
        self._core, self._bins = core, bins
        #: One :class:`Axis` per dimension, x first.
        self.axes = tuple(Axis(core[f"f{letter}axis"]) for letter in "XYZ"[: self._dimensions()])
        self._widths = [len(axis) + 2 for axis in self.axes]
        cells = math.prod(self._widths)
        if len(bins) < cells:
            raise FormatError(
                f"a {classname} of {cells} bins counting the ends holds only {len(bins)} values"
            )

    def _dimensions(self) -> int:
        """How many axes this class has, which a ``TH2F`` says in its name."""
        return int(self.classname[2])

    @property
    def name(self) -> str:
        """What the histogram is called, which is the key it was written under."""
        return str(self._core["TNamed"]["fName"])

    @property
    def title(self) -> str:
        """The title it is drawn with, which is usually a sentence about it."""
        return str(self._core["TNamed"]["fTitle"])

    @property
    def entries(self) -> float:
        """How many times it was filled, which weights make a fraction of."""
        return float(self._core["fEntries"])

    @property
    def shape(self) -> tuple[int, ...]:
        """How many bins along each axis, not counting the two at the ends."""
        return tuple(len(axis) for axis in self.axes)

    def __len__(self) -> int:
        return math.prod(self.shape)

    def edges(self, axis: int = 0) -> np.ndarray[Any, Any]:
        """The edges of one axis, x by default: one more than it has bins."""
        return self.axes[axis].edges()

    #: What the bins hold, in the plotting protocol's words: counts of
    #: things, rather than the means of something a profile keeps.
    kind = "COUNT"

    def values(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """What is in each bin, shaped the way the axes are."""
        return self._shaped(self._bins, flow)

    @property
    def weighted(self) -> bool:
        """Was it filled with weights, so that it keeps their squares too?"""
        return len(self._core["fSumw2"]) == len(self._bins)

    def variances(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The variance of each bin: the sum of the squared weights in it.

        One filled without weights keeps no such sum, and then the variance
        of a count of *n* is *n* itself, which is what ROOT would give back.
        """
        source = self._core["fSumw2"] if self.weighted else self._bins
        return self._shaped(np.asarray(source, dtype=np.float64), flow)

    def errors(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The uncertainty on each bin, the root of its variance.

        A variance below zero is only ever a count gone negative by being
        subtracted from, and ROOT's own answer for it is the root of its size.
        """
        return np.sqrt(np.abs(self.variances(flow)))

    def counts(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """How many fills each bin is worth: the effective number of entries.

        For a histogram filled without weights that is what is in it. With
        weights it is ``values**2 / variances`` - the count that would have
        the same relative uncertainty - which is what the plotting protocol
        means by the word, and zero wherever there is no variance to divide.
        """
        values = self.values(flow).astype(np.float64)
        if not self.weighted:
            return values
        variances = self.variances(flow)
        return np.divide(
            values * values, variances, out=np.zeros_like(values), where=variances > 0
        )

    def sum(self, flow: bool = False) -> float:
        """Everything in the bins added up, which weights make a float of."""
        return math.fsum(self.values(flow).ravel().tolist())

    def density(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The values divided by bin size and by the total, so they integrate to one."""
        if flow:
            raise ValueError("a density of the flow bins is not a thing: they have no width")
        volume = self.axes[0].widths()
        for axis in self.axes[1:]:
            volume = np.multiply.outer(volume, axis.widths())
        total = self.sum()
        return self.values() / volume / total if total else np.zeros(self.shape)

    def _shaped(self, flat: Any, flow: bool) -> np.ndarray[Any, Any]:
        """One flat run of bins cut into the shape its axes give it.

        ROOT writes the bins with x running fastest, which is NumPy's Fortran
        order, so a reshape the other way round and a transpose is the lot.
        """
        cells = math.prod(self._widths)
        full = np.asarray(flat)[:cells].reshape(self._widths[::-1]).T
        if flow:
            return full.copy()
        return full[(slice(1, -1),) * len(self._widths)].copy()

    def to_numpy(self, flow: bool = False) -> tuple[np.ndarray[Any, Any], ...]:
        """The values and each axis's edges, the way :func:`numpy.histogram` gives them.

        With ``flow`` the edges gain an infinite one at each end, so the two
        flow bins still have a place on the axis.
        """
        edges = [axis.edges() for axis in self.axes]
        if flow:
            edges = [np.concatenate(([-np.inf], edge, [np.inf])) for edge in edges]
        return (self.values(flow), *edges)

    def to_hist(self) -> Any:
        """The same histogram as a :class:`hist.Hist`, flow bins and all.

        Weighted histograms become ``Weight`` storage, so the variances go
        across as well as the values; unweighted ones become ``Double``.
        """
        try:
            import hist
        except ImportError:
            raise UnsupportedFeatureError(
                "turning this into a hist.Hist needs the hist package: pip install hist "
                "- or use .to_numpy(), which needs nothing more"
            ) from None
        made = [
            hist.axis.Variable(axis.edges(), name=axis.name, label=axis.title)
            for axis in self.axes
        ]
        storage = hist.storage.Weight() if self.weighted else hist.storage.Double()
        out = hist.Hist(*made, storage=storage, name=self.name, label=self.title)
        if self.weighted:
            out.view(flow=True)["value"] = self.values(flow=True)
            out.view(flow=True)["variance"] = self.variances(flow=True)
        else:
            out.view(flow=True)[...] = self.values(flow=True)
        return out

    @classmethod
    def new(
        cls,
        name: str,
        edges: Any,
        values: Any,
        *,
        title: str = "",
        errors: Any = None,
        variances: Any = None,
        entries: float | None = None,
        labels: Any = None,
    ) -> Histogram:
        """A histogram of one, two or three dimensions built from numbers, ready to write.

            >>> h = Histogram.new("counts", [0, 1, 2, 4], [5, 3, 1])
            >>> h.values()
            array([5., 3., 1.])
            >>> Histogram.new("map", ([0, 1, 2], [0, 5, 10]), [[1, 2], [3, 4]]).classname
            'TH2D'

        ``values`` is what is in each bin, and its dimensions say how many
        axes there are: indexed x first, ``values[ix, iy]``. ``edges`` is every
        bin edge along the one axis, or a sequence of those, one per axis;
        each is one longer than the axis has bins and increasing, and evenly
        spaced ones are stored the compact way ROOT stores an even axis. Give
        two extra values along an axis to fill its flow bins, which are
        otherwise zero.

        ``errors`` is the uncertainty per bin, or ``variances`` its square,
        shaped like ``values``; without either a bin's error is the square
        root of its count, as ROOT gives for a histogram filled without
        weights. ``entries`` is how many fills the histogram represents, the
        sum of the values unless said otherwise, and ``labels`` the titles to
        write along each axis.
        """
        values = np.asarray(values, dtype=np.float64)
        per_axis = _edge_sets(edges, values.ndim)
        shape = tuple(len(edge) - 1 for edge in per_axis)
        full = _flowed(values, shape, "values")
        squares = _squares(errors, variances, shape)
        titles = [str(label) for label in labels] if labels is not None else [""] * len(shape)
        members = _members(name, title, per_axis, titles, full, squares, entries)
        return cls(f"TH{len(shape)}D", members)

    @classmethod
    def of(cls, obj: Any, name: str | None = None) -> Histogram:
        """Any histogram Python has, as one this library reads and writes.

        That is a :class:`Histogram` already, anything speaking the plotting
        protocol - a ``hist.Hist``, a ``boost_histogram.Histogram`` - or what
        :func:`numpy.histogram`, ``histogram2d`` or ``histogramdd`` give back.
        ``name`` is what to call it, when the object does not say.
        """
        if isinstance(obj, Histogram):
            return obj
        found = _numpy_parts(obj)
        if found is not None:
            values, edges = found
            return cls.new(name or "", edges[0] if len(edges) == 1 else edges, values)
        if not all(hasattr(obj, part) for part in ("kind", "values", "variances", "axes")):
            raise TypeError(
                f"a {type(obj).__name__} is not a histogram: it is neither one that "
                f"speaks the plotting protocol nor what numpy.histogram gives back"
            )
        return _from_plottable(cls, obj, name)

    @staticmethod
    def recognises(obj: Any) -> bool:
        """Whether :meth:`of` would take ``obj``, without making anything of it."""
        return (
            isinstance(obj, Histogram)
            or _numpy_parts(obj) is not None
            or all(hasattr(obj, part) for part in ("kind", "values", "variances", "axes"))
        )

    def plot(self, ax: Any = None, **options: Any) -> Any:
        """Draw onto matplotlib axes, made fresh unless ``ax`` brings some.

        One dimension draws as steps, two as a shaded mesh; three have no
        flat picture and refuse rather than pretending. The axes come back,
        so styling and saving carry on where this left off - and matplotlib
        not being installed refuses with the two ways out by name.
        """
        if len(self.axes) > 2:
            raise missing_picture("histogram", len(self.axes))
        if ax is None:
            ax = axes()
        if len(self.axes) == 1:
            ax.stairs(self.values(), self.edges(), **options)
        else:
            ax.pcolormesh(self.edges(0), self.edges(1), self.values().T, **options)
        if self.title:
            ax.set_title(self.title)
        if self.axes[0].title:
            ax.set_xlabel(self.axes[0].title)
        if len(self.axes) > 1 and self.axes[1].title:
            ax.set_ylabel(self.axes[1].title)
        return ax

    def text(self, width: int = 60) -> str:
        """The histogram drawn with characters, for a terminal or a log file.

        One line per bin - its edges, a bar and the value - for one
        dimension; a shaded grid with y upward for two; a refusal for three,
        which have no flat picture.
        """
        if len(self.axes) == 1:
            return _text_1d(self, width)
        if len(self.axes) == 2:
            return _text_2d(self)
        raise missing_picture("histogram", len(self.axes))

    def __repr__(self) -> str:
        shape = " x ".join(str(count) for count in self.shape)
        return f"<{self.classname} {self.name!r} of {shape} bins, {self.entries:g} entries>"


def _shape_text(shape: tuple[int, ...]) -> str:
    return " x ".join(str(count) for count in shape)


def _flowed(values: Any, shape: tuple[int, ...], what: str) -> np.ndarray[Any, Any]:
    """Values for every bin as doubles, the flow bins zeroed unless given."""
    given = np.asarray(values, dtype=np.float64)
    if given.shape == shape:
        return np.pad(given, 1)
    if given.shape == tuple(count + 2 for count in shape):
        return given.copy()
    raise ValueError(
        f"{_shape_text(given.shape)} {what} for {_shape_text(shape)} bins: give one per "
        f"bin, or two more counting the flow at each end"
    )


def _validated_edges(edges: Any) -> np.ndarray[Any, Any]:
    made = np.asarray(edges, dtype=np.float64).reshape(-1)
    if len(made) < 2:
        raise ValueError("a histogram needs at least two edges to have a bin")
    if np.any(np.diff(made) <= 0):
        raise ValueError("edges must increase: each bin has to be wider than nothing")
    return made


def _edge_sets(edges: Any, dimensions: int) -> list[np.ndarray[Any, Any]]:
    """The edges of every axis, for values of ``dimensions`` dimensions."""
    if not 1 <= dimensions <= 3:
        raise ValueError(
            f"values of {dimensions} dimensions are not a histogram ROOT has: it has "
            f"one, two and three"
        )
    if dimensions == 1:
        return [_validated_edges(edges)]
    sets = list(edges)
    if len(sets) != dimensions:
        raise ValueError(
            f"{dimensions}-dimensional values need {dimensions} sets of edges, one per "
            f"axis, and {len(sets)} were given"
        )
    return [_validated_edges(edge) for edge in sets]


def _squares(errors: Any, variances: Any, shape: tuple[int, ...]) -> np.ndarray[Any, Any] | None:
    """The sum of squared weights per bin, from whichever of the two was given."""
    if errors is not None and variances is not None:
        raise ValueError("give errors or variances, not both: one is the root of the other")
    if errors is not None:
        return _flowed(errors, shape, "errors") ** 2
    if variances is not None:
        return _flowed(variances, shape, "variances")
    return None


def _stored(edges: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The edges an axis keeps: none when they are even, since the ends say them."""
    nbins = len(edges) - 1
    width = (edges[-1] - edges[0]) / nbins
    even = np.append(edges[0] + width * np.arange(nbins), edges[-1])
    return np.zeros(0) if np.array_equal(edges, even) else edges


def _fsum(values: Any) -> float:
    return math.fsum(np.asarray(values, dtype=np.float64).ravel().tolist())


def _moments(inner: np.ndarray[Any, Any], centers: list[np.ndarray[Any, Any]]) -> dict[str, float]:
    """The sums ROOT keeps to give a mean and a spread without the bins.

    For each axis the weighted sum of the bin centres and of their squares,
    and for each pair of axes the weighted sum of their products - which is
    what ``GetMean``, ``GetStdDev`` and ``GetCorrelationFactor`` are made of.
    """
    letters = "xyz"[: inner.ndim]
    sums: dict[str, float] = {}
    for axis, (letter, center) in enumerate(zip(letters, centers)):
        others = tuple(other for other in range(inner.ndim) if other != axis)
        along = inner.sum(axis=others) if others else inner
        sums[f"fTsumw{letter}"] = _fsum(along * center)
        sums[f"fTsumw{letter}2"] = _fsum(along * center * center)
    for first in range(inner.ndim):
        for second in range(first + 1, inner.ndim):
            shape = [1] * inner.ndim
            shape[first], shape[second] = len(centers[first]), len(centers[second])
            product = np.multiply.outer(centers[first], centers[second]).reshape(shape)
            sums[f"fTsumw{letters[first]}{letters[second]}"] = _fsum(inner * product)
    return sums


def _members(
    name: str,
    title: str,
    per_axis: list[np.ndarray[Any, Any]],
    titles: list[str],
    full: np.ndarray[Any, Any],
    squares: np.ndarray[Any, Any] | None,
    entries: float | None,
) -> dict[str, Any]:
    """Every member of a freshly made ``TH1D``, ``TH2D`` or ``TH3D``.

    The bins go in with x running fastest, as ROOT keeps them; the sums of
    moments are worked out from the bins, which is what ROOT has too for a
    histogram it did not fill itself, entry by entry.
    """
    inner = full[(slice(1, -1),) * full.ndim]
    centers = [(edge[:-1] + edge[1:]) / 2 for edge in per_axis]
    moments = _moments(inner, centers)
    core = _histogram_core(name, title, per_axis, titles, full, inner, squares, entries)
    core["fTsumwx"], core["fTsumwx2"] = moments.pop("fTsumwx"), moments.pop("fTsumwx2")
    bins = {"TArrayD": full.ravel(order="F")}
    if full.ndim == 1:
        return {"TH1": core, **bins}
    if full.ndim == 2:
        return {"TH2": {"TH1": core, "fScalefactor": 1.0, **moments}, **bins}
    return {"TH3": {"TH1": core, "TAtt3D": {}, **moments}, **bins}


def _histogram_core(
    name: str,
    title: str,
    per_axis: list[np.ndarray[Any, Any]],
    titles: list[str],
    full: np.ndarray[Any, Any],
    inner: np.ndarray[Any, Any],
    squares: np.ndarray[Any, Any] | None,
    entries: float | None,
) -> dict[str, Any]:
    """The ``TH1`` every histogram is built on, whatever its dimensions."""
    made = [
        _axis(f"{letter}axis", len(edge) - 1, float(edge[0]), float(edge[-1]), _stored(edge))
        for letter, edge in zip("xyz", per_axis)
    ]
    for axis, label in zip(made, titles):
        axis["TNamed"]["fTitle"] = label
    while len(made) < 3:
        made.append(_axis("xyz"[len(made)] + "axis", 1, 0.0, 1.0, []))
    return {
        "TNamed": {"fName": str(name), "fTitle": str(title)},
        "TAttLine": dict(LINE),
        "TAttFill": dict(FILL),
        "TAttMarker": dict(MARKER),
        "fNcells": full.size,
        "fXaxis": made[0],
        "fYaxis": made[1],
        "fZaxis": made[2],
        "fBarOffset": 0,
        "fBarWidth": 1000,
        "fEntries": float(entries) if entries is not None else _fsum(full),
        "fTsumw": _fsum(inner),
        "fTsumw2": _fsum(squares[(slice(1, -1),) * full.ndim] if squares is not None else inner),
        "fMaximum": -1111.0,
        "fMinimum": -1111.0,
        "fNormFactor": 0.0,
        "fContour": np.zeros(0),
        "fSumw2": squares.ravel(order="F") if squares is not None else np.zeros(0),
        "fOption": "",
        "fFunctions": [],
        "fBufferSize": 0,
        "fBuffer": np.zeros(0),
        "fBinStatErrOpt": 0,
    }


def _numpy_parts(obj: Any) -> tuple[np.ndarray[Any, Any], list[Any]] | None:
    """The values and the edges, if ``obj`` is what numpy's histograms give back.

    :func:`numpy.histogram` gives ``(values, edges)``, ``histogram2d`` gives
    ``(values, xedges, yedges)``, and ``histogramdd`` gives ``(values,
    [edges, ...])``; anything else is not one of them.
    """
    if not isinstance(obj, tuple) or len(obj) < 2 or not isinstance(obj[0], np.ndarray):
        return None
    edges = list(obj[1]) if len(obj) == 2 and isinstance(obj[1], (list, tuple)) else list(obj[1:])
    if obj[0].ndim != len(edges) or not all(np.ndim(edge) == 1 for edge in edges):
        return None
    return obj[0], edges


def _plottable_edges(axis: Any) -> np.ndarray[Any, Any]:
    """The edges of one axis of a protocol histogram, which gives its bins as pairs."""
    traits = getattr(axis, "traits", None)
    if getattr(traits, "discrete", False) or getattr(traits, "circular", False):
        raise UnsupportedFeatureError(
            "an axis of categories, or one that wraps round, has no ROOT spelling "
            "that this writer makes yet; rebin it onto a regular or variable axis"
        )
    bins = list(axis)
    return np.asarray([low for low, _high in bins] + [bins[-1][1]], dtype=np.float64)


def _plottable_bins(obj: Any, shape: tuple[int, ...]) -> tuple[Any, Any]:
    """Values and variances of a protocol histogram, with the flow when it keeps any."""
    flowed = tuple(count + 2 for count in shape)
    try:
        values, variances = obj.values(flow=True), obj.variances(flow=True)
    except TypeError:
        values, variances = obj.values(), obj.variances()
    if np.shape(values) not in (shape, flowed):
        values, variances = obj.values(), obj.variances()
    return values, variances


def _from_plottable(cls: type[Histogram], obj: Any, name: str | None) -> Histogram:
    kind = str(getattr(obj.kind, "value", obj.kind))
    if kind != "COUNT":
        raise UnsupportedFeatureError(
            f"a histogram of kind {kind} keeps means rather than counts, which is a "
            f"TProfile, and this writer does not make those yet"
        )
    edges = [_plottable_edges(axis) for axis in obj.axes]
    shape = tuple(len(edge) - 1 for edge in edges)
    values, variances = _plottable_bins(obj, shape)
    return cls.new(
        name or _said(obj, "name"),
        edges[0] if len(edges) == 1 else edges,
        values,
        title=_said(obj, "label"),
        variances=variances,
        labels=[_said(axis, "label") for axis in obj.axes],
    )


def _said(obj: Any, what: str) -> str:
    """A name or label a protocol object may carry, or nothing if it carries none."""
    return str(getattr(obj, what, None) or "")


def _text_1d(histogram: Histogram, width: int) -> str:
    values = histogram.values()
    edges = histogram.edges()
    top = max((value for value in values if value > 0), default=0.0)
    return "\n".join(
        f"[{edges[step]:g}, {edges[step + 1]:g})".rjust(24)
        + f" {bar(value / top if top else 0.0, width):<{width}} {value:g}"
        for step, value in enumerate(values)
    )


def _text_2d(histogram: Histogram) -> str:
    rows = histogram.values()
    top = max((value for row in rows for value in row if value > 0), default=0.0)
    return "\n".join(
        "".join(shade(row[step] / top if top else 0.0) for row in rows)
        for step in reversed(range(len(histogram.axes[1])))
    )
