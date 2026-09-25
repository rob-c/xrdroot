"""Reshaping: ROOT's ``Rebin``, ``Rebin2D``, projections and profiles of a histogram.

Merging bins, summing a histogram along an axis and averaging it along one
are each a few lines of arithmetic and a page of bookkeeping: where what fell
off an axis goes when the axis changes, when the running sums of the old
histogram still describe the new one and when they have to be made again, and
how many entries the result says it has. Each function here keeps ROOT's
bookkeeping, and makes a new histogram rather than changing the one it was
given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from .arithmetic import copied, put_statistics, reset_statistics
from .booking import Binning, axis_members, binning
from .errors import UnsupportedFeatureError
from .filling import running, store_cells
from .moments import effective_entries, statistics

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Axis, Histogram
    from .profile import Profile

__all__ = ["profiled", "projection", "rebinned"]

#: How near a projection's total must come to the histogram's own total
#: weight for ROOT to keep the running sums: closer for doubles than floats.
MATCH = {"TArrayF": 1e-6}
MATCH_DOUBLE = 1e-12


def _counting(histogram: Histogram, what: str) -> None:
    if histogram.kind == "MEAN":
        raise UnsupportedFeatureError(
            f"{what} {histogram.name!r} is not offered: it is a profile, whose bins are means, "
            f"and ROOT does that for a profile a way of its own"
        )


def _new_axis(histogram: Histogram, letter: str, axis: Binning) -> None:
    """Rebuild one ``TAxis`` of ``histogram``'s members for a new binning, title kept."""
    old = histogram._core[f"f{letter}axis"]
    made = axis_members(old["TNamed"]["fName"], axis.nbins, axis.low, axis.high, axis.stored)
    made["TNamed"]["fTitle"] = old["TNamed"]["fTitle"]
    made["TAttAxis"] = dict(old["TAttAxis"])
    histogram._core[f"f{letter}axis"] = made


def _rebuilt(histogram: Histogram, axes: Sequence[Binning], name: str | None) -> Histogram:
    """A copy of ``histogram`` on new axes, its bins and squares resized and empty."""
    made = copied(histogram, name)
    for letter, axis in zip("XYZ", axes):
        _new_axis(made, letter, axis)
    cells = math.prod(axis.nbins + 2 for axis in axes)
    made._core["fNcells"] = cells
    made._home[made._key] = np.zeros(cells, dtype=made._cells().dtype)
    if histogram._sumw2() is not None:
        made._core["fSumw2"] = np.zeros(cells)
    return type(histogram)(made.classname, made.members)


# -- Rebin ------------------------------------------------------------------


def _grouped(axis: Axis, group: int, uneven: bool = False) -> tuple[Binning, bool]:
    """The axis ``Rebin(group)`` makes, and whether bins fell off its end into the overflow.

    An uneven axis keeps every edge it lands on, and so does an even one when
    ``uneven`` says so - as ``Rebin2D`` makes both axes uneven when either is.
    """
    if not isinstance(group, (int, np.integer)) or not 1 <= group <= axis.nbins:
        raise ValueError(
            f"a group of {group!r} bins does not rebin an axis of {axis.nbins}: give a whole "
            f"number from 1 to {axis.nbins}"
        )
    count = axis.nbins // int(group)
    dropped = count * group != axis.nbins
    if uneven or not axis.even:
        edges = np.array([_low_edge(axis, 1 + step * int(group)) for step in range(count + 1)])
        return Binning(count, float(edges[0]), float(edges[-1]), edges), dropped
    high = _low_edge(axis, count * int(group) + 1) if dropped else axis.high
    return Binning(count, axis.low, high, np.zeros(0)), dropped


def _low_edge(axis: Axis, bin: int) -> float:
    """``TAxis::GetBinLowEdge``, which works an edge past the last one out as if even."""
    if not axis.even and 0 < bin <= axis.nbins:
        return float(axis.edges()[bin - 1])
    return axis.low + (bin - 1) * ((axis.high - axis.low) / axis.nbins)


def _subset(axis: Axis, edges: Any) -> Binning:
    """The axis ``Rebin`` given edges makes, refusing any edge the old axis lacks."""
    made = binning(np.asarray(edges, dtype=np.float64))
    old = axis.edges()
    stray = [float(edge) for edge in made.stored if edge not in old]
    if stray:
        raise ValueError(
            f"rebinning onto new edges can only merge bins, never split one: "
            f"{stray[0]:g} is not an edge of the axis, which runs from {axis.low:g} to "
            f"{axis.high:g} in {axis.nbins} bins"
        )
    return made


def rebinned(histogram: Histogram, groups: Sequence[Any], name: str | None) -> Histogram:
    """``TH1::Rebin`` and ``TH2::Rebin2D``: neighbouring bins merged into one."""
    _counting(histogram, "rebinning")
    count = len(histogram.axes)
    if count == 1 and len(groups) == 1 and np.ndim(groups[0]) == 1:
        made = _subset(histogram.axes[0], groups[0])
        return _rebin_1d(histogram, made, histogram.axes[0].nbins, False, name)
    if len(groups) != count or count == 3:
        raise ValueError(
            f"{histogram.name!r} has {count} axes, and rebinning takes one group per axis for "
            f"one or two - or, for one axis, the new edges"
        )
    if count == 1:
        axis, dropped = _grouped(histogram.axes[0], groups[0])
        return _rebin_1d(histogram, axis, int(groups[0]), dropped, name)
    return _rebin_2d(histogram, [int(group) for group in groups], name)


def _rebin_1d(
    histogram: Histogram, axis: Binning, group: int, dropped: bool, name: str | None
) -> Histogram:
    """``TH1::Rebin``, bin by bin as ROOT walks it.

    A new bin takes old bins in turn until one's centre passes its upper
    edge; the underflow takes every old bin before the new axis starts, and
    the overflow every one after it ends.
    """
    errors = histogram._bin_errors() if histogram._sumw2() is not None else None
    found, entries = statistics(histogram), histogram.entries
    made = _rebuilt(histogram, [axis], name)
    parts = _parts(histogram.axes[0], made.axes[0].edges(), group)
    contents = histogram._bins.astype(np.float64)
    gathered = [_gathered(contents, errors, part) for part in parts]
    sums, squares = (np.array(column) for column in zip(*gathered))
    _finish(made, sums, squares, errors is not None)
    _kept_statistics(made, found, entries, dropped)
    return made


def _parts(old: Axis, new: np.ndarray[Any, Any], group: int) -> list[slice]:
    """Which old bins each new bin takes, the underflow first and the overflow last.

    ROOT starts at the first old bin whose centre is past the new low edge;
    a new bin then takes up to ``group`` old ones in turn, stopping at one
    whose centre is past its upper edge. What is before the start goes to
    the underflow, and what is left at the end to the overflow.
    """
    centres = old.root_centers()
    start = 1
    while start <= old.nbins and centres[start] < new[0]:
        start += 1
    parts, at = [slice(0, start)], start
    for upper in new[1:]:
        taken = 0
        while taken < group and at + taken <= old.nbins and centres[at + taken] <= upper:
            taken += 1
        parts.append(slice(at, at + taken))
        at += taken
    return [*parts, slice(at, old.nbins + 2)]


def _kept_statistics(made: Histogram, found: list[float], entries: float, dropped: bool) -> None:
    """The entries and running sums a rebinned histogram keeps from the old one.

    When bins went to the overflow the old sums describe what is no longer on
    the axis, and ROOT leaves the total weight at zero, so they are made from
    the bins when next asked for.
    """
    made._core["fEntries"] = entries
    if dropped:
        made._moment_homes()["fTsumw"]["fTsumw"] = 0.0
    else:
        put_statistics(made, found)


def _gathered(contents: Any, errors: Any, part: slice) -> tuple[float, float]:
    """The sum of some old bins, and the sum of their errors squared, in turn."""
    total = running(0.0, contents[part])
    if errors is None:
        return total, 0.0
    return total, running(0.0, errors[part] * errors[part])


def _finish(made: Histogram, sums: Any, squares: Any, weighted: bool) -> None:
    """Set the new bins with ``SetBinContent`` and their errors with ``SetBinError``.

    An error set is squared back from its root, as ROOT's ``SetBinError``
    squares it, so the squares are ROOT's to the bit.
    """
    store_cells(made._cells(), sums)
    if weighted:
        made._ensure_sumw2()[:] = np.sqrt(squares) ** 2


def _rebin_2d(histogram: Histogram, groups: list[int], name: str | None) -> Histogram:
    """``TH2::Rebin2D``: blocks of bins merged, the flow and the leftovers kept in the flow."""
    uneven = not all(axis.even for axis in histogram.axes)
    made_axes, dropped = zip(
        *(_grouped(axis, group, uneven) for axis, group in zip(histogram.axes, groups))
    )
    shape = tuple(axis.nbins + 2 for axis in histogram.axes)
    maps = [
        _bin_map(axis.nbins, group, new.nbins)
        for axis, group, new in zip(histogram.axes, groups, made_axes)
    ]
    found, entries = statistics(histogram), histogram.entries
    made = _rebuilt(histogram, list(made_axes), name)
    target = maps[0][:, None] + (made_axes[0].nbins + 2) * maps[1][None, :]
    order = target.ravel(order="F")
    sums = np.zeros(math.prod(axis.nbins + 2 for axis in made_axes))
    np.add.at(sums, order, histogram._bins.astype(np.float64)[: math.prod(shape)])
    squares = np.zeros_like(sums)
    weighted = histogram._sumw2() is not None
    if weighted:
        np.add.at(squares, order, histogram._variance_cells()[: math.prod(shape)])
    store_cells(made._cells(), sums)
    if weighted:
        made._ensure_sumw2()[:] = squares
    _kept_statistics(made, found, entries, any(dropped))
    return made


def _bin_map(nbins: int, group: int, count: int) -> Any:
    """Which new bin each old one goes to: groups in order, the rest to the overflow."""
    old = np.arange(nbins + 2)
    mapped = np.where(old == 0, 0, 1 + (old - 1) // group)
    return np.where(old > count * group, count + 1, mapped)


# -- projections ------------------------------------------------------------


def _letters(histogram: Histogram, axes: str) -> list[int]:
    letters = "xyz"[: len(histogram.axes)]
    picked = [letters.find(letter) for letter in axes.lower()]
    if not 1 <= len(picked) < len(letters) or -1 in picked or len(set(picked)) != len(picked):
        raise ValueError(
            f"{axes!r} is not a projection of {histogram.name!r}, whose axes are "
            f"{', '.join(letters)}: name fewer of them than it has, each once"
        )
    return picked


def _sum_range(axis: Axis, given: Any) -> tuple[int, int]:
    """The bins of a summed-out axis, as ROOT's projections clamp them."""
    if given is None:
        return 0, axis.nbins + 1
    first, last = (int(bound) for bound in given)
    if last < first:
        return 0, axis.nbins + 1
    return max(first, 0), min(last, axis.nbins + 1)


def projection(
    histogram: Histogram, axes: str, name: str | None, ranges: Mapping[str, Any] | None
) -> Histogram:
    """``ProjectionX``, ``ProjectionY`` and ``Project3D``: a histogram summed over axes.

    ``axes`` names what is kept, in the order the result has them - ``"x"``,
    or ``"xy"`` for a ``TH2D`` of x against y - and every other axis is
    summed over, flow bins and all unless ``ranges`` gives it ROOT's first
    and last bin numbers. The running sums carry over when every bin on the
    summed axes was taken and none of their flow, or when the total comes
    out the same, and the entries only when all of it was; otherwise they
    are made as ROOT makes them.
    """
    _counting(histogram, "projecting")
    kept = _letters(histogram, axes)
    summed = [axis for axis in range(len(histogram.axes)) if axis not in kept]
    bounds = dict(ranges or {})
    cuts = {axis: _sum_range(histogram.axes[axis], bounds.get("xyz"[axis])) for axis in summed}
    cut = tuple(
        slice(cuts[axis][0], cuts[axis][1] + 1) if axis in cuts else slice(None)
        for axis in range(len(histogram.axes))
    )
    sums = _summed(histogram.values(flow=True).astype(np.float64)[cut], summed, kept)
    squares = _summed((histogram.errors(flow=True) ** 2)[cut], summed, kept)
    made = _projected(histogram, kept, axes, name)
    _finish(made, sums.ravel(order="F"), squares.ravel(order="F"), histogram._sumw2() is not None)
    total = running(0.0, sums.ravel())
    reuse = _spans(histogram, cuts, False) or _matches(histogram, total)
    _projected_statistics(histogram, made, kept, reuse, _spans(histogram, cuts, True), total)
    return made


def _spans(histogram: Histogram, cuts: dict[int, tuple[int, int]], flow: bool) -> bool:
    """Whether every summed axis was summed over all its bins - and its flow, or none of it."""
    extra = 1 if flow else 0
    return all(
        bounds == (1 - extra, histogram.axes[axis].nbins + extra) for axis, bounds in cuts.items()
    )


def _summed(values: Any, summed: list[int], kept: list[int]) -> Any:
    """Values summed over some axes in turn, what is left in the order ``kept`` asks for.

    ROOT sums a projection's bins one at a time, the summed axes nested in
    their own order, so this does too rather than let NumPy add pairwise.
    """
    moved = values.transpose([*kept, *summed])
    flat = moved.reshape(*moved.shape[: len(kept)], -1)
    return np.add.accumulate(flat, axis=-1)[..., -1]


def _matches(histogram: Histogram, total: float) -> bool:
    weight = float(histogram._moment_homes()["fTsumw"]["fTsumw"])
    near = MATCH.get(histogram._key, MATCH_DOUBLE)
    return weight != 0 and abs(weight - total) < abs(weight) * near


def _projected(histogram: Histogram, kept: list[int], axes: str, name: str | None) -> Histogram:
    """The empty ``TH1D`` or ``TH2D`` a projection fills, binned as the kept axes are."""
    from .hist import Histogram

    suffix = f"_p{axes}" if len(histogram.axes) == 2 else f"_{axes}"
    specs = [_spec(histogram.axes[axis]) for axis in kept]
    labels = [histogram.axes[axis].title for axis in kept]
    return Histogram.book(
        name or histogram.name + suffix, *specs, title=histogram.title, labels=labels
    )


def _spec(axis: Axis) -> Any:
    return (axis.nbins, axis.low, axis.high) if axis.even else axis.edges()


def _projected_statistics(
    histogram: Histogram, made: Histogram, kept: list[int], reuse: bool, whole: bool, total: float
) -> None:
    """The running sums and entries of a projection, by ROOT's ``DoProjection`` rules."""
    if reuse:
        found = statistics(histogram)
        letters = "xyz"
        names = ["fTsumw", "fTsumw2"]
        for axis in kept:
            names += [f"fTsumw{letters[axis]}", f"fTsumw{letters[axis]}2"]
        if len(kept) == 2:
            pair = "".join(sorted(letters[axis] for axis in kept))
            names.append(f"fTsumw{pair}")
        order = histogram._moment_names()
        put_statistics(made, [found[order.index(name)] for name in names])
    else:
        # What ROOT's SetBinContent leaves: one entry per bin set, no total
        # weight, and so running sums made from the bins when asked for.
        made._core["fEntries"] = float(len(made._bins))
        made._moment_homes()["fTsumw"]["fTsumw"] = 0.0
    if reuse and whole:
        made._core["fEntries"] = histogram.entries
    elif made._sumw2() is not None:
        made._core["fEntries"] = effective_entries(made)
    else:
        made._core["fEntries"] = math.floor(total + 0.5)


# -- profiles ---------------------------------------------------------------


def profiled(histogram: Histogram, along: int, name: str | None, bounds: Any) -> Profile:
    """``TH2::ProfileX`` and ``ProfileY``: the mean of one axis in each bin of the other.

    Every bin's content is filled into the profile as the weight of its
    centre, so the profile's mean is the histogram's own mean along that
    axis, bin by bin; the flow bins of the axis kept are filled too, and of
    the axis averaged only the range asked for, all of its bins by default.
    """
    from .profile import Profile

    if len(histogram.axes) != 2 or histogram.kind == "MEAN":
        raise ValueError(
            f"a profile is made from a two-dimensional histogram, and {histogram.name!r} is not one"
        )
    out, inner = histogram.axes[along], histogram.axes[1 - along]
    first, last = _profile_range(inner, bounds)
    suffix = "_pfx" if along == 0 else "_pfy"
    made = Profile.book(
        name or histogram.name + suffix, _spec(out), title=histogram.title, labels=[out.title]
    )
    grid = histogram.values(flow=True).astype(np.float64)
    if along == 1:
        grid = grid.T
    block = grid[:, first : last + 1]
    rows = np.repeat(np.arange(out.nbins + 2), block.shape[1])
    columns = np.tile(np.arange(first, last + 1), out.nbins + 2)
    weights = block.ravel()
    filled = weights != 0
    xs = out.root_centers()[rows[filled]]
    ys = inner.root_centers()[columns[filled]]
    weighted = histogram._sumw2() is not None
    if weighted:
        made._ensure_sumw2()
    made.fill(xs, ys, weight=weights[filled])
    if weighted:
        # ROOT fills with the content as the weight and then overwrites the
        # squares of weights with the histogram's own, bin by bin in turn.
        origin = _origin(histogram, along, rows[filled], columns[filled])
        target = made._ensure_sumw2()
        target[:] = 0.0
        np.add.at(target, made.axes[0].find_bin(xs), histogram._variance_cells()[origin])
    reset_statistics(made)
    made._core["fEntries"] = effective_entries(made)
    return made


def _profile_range(axis: Axis, bounds: Any) -> tuple[int, int]:
    """The bins a profile averages over, clamped the way ``DoProfile`` clamps them."""
    first, last = (1, -1) if bounds is None else (int(bound) for bound in bounds)
    if first < 0:
        first = 1
    if last < 0 or last > axis.nbins + 1:
        last = axis.nbins
    return first, last


def _origin(histogram: Histogram, along: int, rows: Any, columns: Any) -> Any:
    """The global bin in ``histogram`` of each (kept, averaged) pair of bin numbers."""
    width = histogram.axes[0].nbins + 2
    x, y = (rows, columns) if along == 0 else (columns, rows)
    return x + width * y
