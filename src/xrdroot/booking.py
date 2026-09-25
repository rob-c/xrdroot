"""Booking: the members of a histogram, profile or efficiency before anything is in it.

``TH1D h("h", "title", 100, 0, 1)`` in ROOT makes an object whose every member
has a value - the axes, the drawing attributes, a zero in each moment - and
that is what is written when nothing has been filled yet. This makes the same
members, in the same shapes the reader gives back for a file ROOT wrote, so a
booked histogram and one read from a file are the same kind of thing and the
writer, the statistics and the filling never have to tell them apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

__all__ = [
    "AXIS_STYLE",
    "FILL",
    "LINE",
    "MARKER",
    "STORAGE",
    "Binning",
    "axis_members",
    "binning",
    "histogram_members",
    "profile_members",
    "split_title",
    "th1_members",
]

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

#: The bin storage each histogram class letter keeps: the ``TArray`` it is
#: built on, and the NumPy type of one bin.
STORAGE = {
    "C": ("TArrayC", np.int8),
    "S": ("TArrayS", np.int16),
    "I": ("TArrayI", np.int32),
    "F": ("TArrayF", np.float32),
    "D": ("TArrayD", np.float64),
}

#: ROOT's ``EStatOverflows::kNeutral``: follow the global setting, which
#: leaves what fell off the axes out of the moments.
NEUTRAL = 2

#: The letters ``SetErrorOption`` takes, against the ``fErrorMode`` each sets.
ERROR_OPTIONS = {"": 0, "s": 1, "i": 2, "g": 3}


class Binning(NamedTuple):
    """One axis as booked: how many bins, where it runs, and the edges it keeps.

    ``stored`` is empty for an evenly binned axis, whose two ends and count say
    where every edge is, and every edge for one binned unevenly - exactly
    what ROOT keeps in ``fXbins``.
    """

    nbins: int
    low: float
    high: float
    stored: np.ndarray[Any, Any]


def axis_members(name: str, nbins: int, low: float, high: float, edges: Any) -> dict[str, Any]:
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


def _integral(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def binning(spec: Any) -> Binning:
    """One axis from how it is asked for: ``(nbins, low, high)``, or its edges.

    A tuple of three whose first is a whole number is ROOT's evenly binned
    axis; anything else is every edge in order, as ROOT's variable binning
    takes them. So edges are given as a list or an array, never as a tuple of
    three - which would be read as a count and two ends.
    """
    if isinstance(spec, tuple) and len(spec) == 3 and _integral(spec[0]):
        return _even(int(spec[0]), float(spec[1]), float(spec[2]))
    edges = np.asarray(spec, dtype=np.float64).reshape(-1)
    if len(edges) < 2:
        raise ValueError(
            "an axis needs a count and two ends, (nbins, low, high), or at least two edges"
        )
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("an axis's edges must be finite and increase, each bin wider than none")
    return Binning(len(edges) - 1, float(edges[0]), float(edges[-1]), edges)


def _even(nbins: int, low: float, high: float) -> Binning:
    if nbins < 1:
        raise ValueError(
            f"an axis of {nbins} bins has nowhere to put anything: ask for one or more"
        )
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(
            f"an axis from {low} to {high} is not a range: its low end must be below its high "
            f"one, and both finite"
        )
    return Binning(nbins, low, high, np.zeros(0))


def split_title(title: str) -> tuple[str, list[str]]:
    """A title and the axis titles ROOT's ``"title;x;y;z"`` spelling carries.

    ``TH1::SetTitle`` splits at the semicolons: what comes first is the title,
    and the rest, in order, the titles of the x, y and z axes.
    """
    parts = str(title).split(";", 3)
    return parts[0], parts[1:]


def _made_axes(
    axes: Sequence[Binning], titles: Sequence[str]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """All three ``TAxis`` members, the unused ones the one-bin axis ROOT keeps."""
    made = [
        axis_members(f"{letter}axis", axis.nbins, axis.low, axis.high, axis.stored)
        for letter, axis in zip("xyz", axes)
    ]
    for axis, label in zip(made, titles):
        axis["TNamed"]["fTitle"] = str(label)
    while len(made) < 3:
        made.append(axis_members("xyz"[len(made)] + "axis", 1, 0.0, 1.0, []))
    return made[0], made[1], made[2]


def th1_members(
    name: str, title: str, axes: Sequence[Binning], titles: Sequence[str]
) -> dict[str, Any]:
    """The ``TH1`` every histogram is built on, empty: what a ROOT constructor leaves."""
    x, y, z = _made_axes(axes, titles)
    cells = math.prod(axis.nbins + 2 for axis in axes)
    return {
        "TNamed": {"fName": str(name), "fTitle": str(title)},
        "TAttLine": dict(LINE),
        "TAttFill": dict(FILL),
        "TAttMarker": dict(MARKER),
        "fNcells": cells,
        "fXaxis": x,
        "fYaxis": y,
        "fZaxis": z,
        "fBarOffset": 0,
        "fBarWidth": 1000,
        "fEntries": 0.0,
        "fTsumw": 0.0,
        "fTsumw2": 0.0,
        "fTsumwx": 0.0,
        "fTsumwx2": 0.0,
        "fMaximum": -1111.0,
        "fMinimum": -1111.0,
        "fNormFactor": 0.0,
        "fContour": np.zeros(0),
        "fSumw2": np.zeros(0),
        "fOption": "",
        "fFunctions": [],
        "fBufferSize": 0,
        "fBuffer": np.zeros(0),
        "fBinStatErrOpt": 0,
        "fStatOverflows": NEUTRAL,
    }


def _dimensioned(core: dict[str, Any], dimensions: int) -> dict[str, Any]:
    """The ``TH1`` wrapped in the ``TH2`` or ``TH3`` that adds the other moments."""
    if dimensions == 1:
        return core
    if dimensions == 2:
        return {"TH1": core, "fScalefactor": 1.0, **dict.fromkeys(_MOMENTS[2], 0.0)}
    return {"TH1": core, "TAtt3D": {}, **dict.fromkeys(_MOMENTS[3], 0.0)}


#: The moments ``TH2`` and ``TH3`` add to what ``TH1`` keeps.
_MOMENTS = {
    2: ("fTsumwy", "fTsumwy2", "fTsumwxy"),
    3: ("fTsumwy", "fTsumwy2", "fTsumwxy", "fTsumwz", "fTsumwz2", "fTsumwxz", "fTsumwyz"),
}


def _parsed(specs: Sequence[Any], labels: Any, title: str) -> tuple[list[Binning], str, list[str]]:
    if not 1 <= len(specs) <= 3:
        raise ValueError(
            f"{len(specs)} axes is not a histogram ROOT has: it has one, two and three"
        )
    axes = [binning(spec) for spec in specs]
    title, titles = split_title(title)
    if labels is not None:
        titles = [str(label) for label in labels]
    return axes, title, titles


def histogram_members(
    name: str, specs: Sequence[Any], title: str, kind: str, labels: Any
) -> tuple[str, dict[str, Any]]:
    """The class and members of an empty ``TH1C`` to ``TH3D``."""
    if kind not in STORAGE:
        raise ValueError(
            f"kind={kind!r} is not a storage ROOT's histograms have: 'C', 'S' and 'I' for "
            f"8, 16 and 32-bit integers, 'F' for floats and 'D' for doubles"
        )
    axes, title, titles = _parsed(specs, labels, title)
    core = th1_members(name, title, axes, titles)
    base, dtype = STORAGE[kind]
    members = {
        (f"TH{len(axes)}" if len(axes) > 1 else "TH1"): _dimensioned(core, len(axes)),
        base: np.zeros(core["fNcells"], dtype=dtype),
    }
    return f"TH{len(axes)}{kind}", members


#: What each profile adds to the histogram of doubles it is built on: the
#: name of the value's range and of its moments, which follow the letters
#: of the axes it is binned along.
_PROFILE_VALUE = {1: "y", 2: "z", 3: "t"}
_PROFILE_CLASS = {1: "TProfile", 2: "TProfile2D", 3: "TProfile3D"}


def profile_members(
    name: str,
    specs: Sequence[Any],
    title: str,
    error_option: str,
    value_range: Any,
    labels: Any,
) -> tuple[str, dict[str, Any]]:
    """The class and members of an empty ``TProfile``, ``TProfile2D`` or ``TProfile3D``.

    It is the ``TH1D``, ``TH2D`` or ``TH3D`` it derives from, with its sums of
    squares allocated from the start as ROOT allocates them, and the sum of
    weights per bin beside it.
    """
    option = str(error_option).lower()
    if option not in ERROR_OPTIONS:
        raise ValueError(
            f"error_option={error_option!r} is not one of ROOT's: '' for the error on the "
            f"mean, 's' for the spread, 'i' for integers and 'g' for Gaussian weights"
        )
    low, high = (0.0, 0.0) if value_range is None else value_range
    _classname, base = histogram_members(name, specs, title, "D", labels)
    dimensions = len(specs)
    core = base["TH1"] if dimensions == 1 else base[f"TH{dimensions}"]["TH1"]
    core["fSumw2"] = np.zeros(core["fNcells"])
    value = _PROFILE_VALUE[dimensions]
    bound = value.upper()
    members = {
        f"TH{dimensions}D": base,
        "fBinEntries": np.zeros(core["fNcells"]),
        "fErrorMode": ERROR_OPTIONS[option],
        f"f{bound}min": float(low),
        f"f{bound}max": float(high),
        f"fTsumw{value}": 0.0,
        f"fTsumw{value}2": 0.0,
        "fBinSumw2": np.zeros(0),
    }
    return _PROFILE_CLASS[dimensions], members
