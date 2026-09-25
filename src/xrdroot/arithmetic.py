"""Arithmetic: ROOT's ``Add``, ``Scale``, ``Multiply``, ``Divide`` and ``Merge``.

Adding two histograms is not adding two arrays. The errors propagate - in
quadrature for a sum, by the relative errors for a product or a ratio, and by
the binomial formula for an efficiency - the running sums behind the mean and
the spread are carried along where they still mean something and rebuilt from
the bins where they do not, and the number of entries follows rules of its
own. Every function here does what the ROOT method it is named after does, in
the same order, so the numbers that come out are ROOT's.

A profile's bins are means rather than counts, and ROOT's arithmetic on them
is a different arithmetic; only merging - adding up what went into each bin,
as ``hadd`` does - is offered for profiles, and the rest refuses them.
"""

from __future__ import annotations

import copy as _copy
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from .errors import UnsupportedFeatureError
from .filling import add_to_cells, store_cells
from .moments import integral, statistics

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Histogram

    H = TypeVar("H", bound=Histogram)

__all__ = [
    "add",
    "compatible",
    "copied",
    "divide",
    "merge",
    "multiply",
    "normalized",
    "put_statistics",
    "reset",
    "reset_statistics",
    "scale",
]

#: What ROOT keeps in ``fMaximum`` and ``fMinimum`` for a limit never set;
#: every operation that changes the bins forgets any that was.
UNSET = -1111.0


def _describe(histogram: Histogram) -> str:
    return " by ".join(
        f"{axis.nbins} bins from {axis.low:g} to {axis.high:g}" for axis in histogram.axes
    )


def compatible(one: Histogram, other: Histogram) -> None:
    """Refuse two histograms whose bins are not the same bins, naming both."""
    if one.axes != other.axes:
        raise ValueError(
            f"{one.name!r} and {other.name!r} are binned differently - {_describe(one)} "
            f"against {_describe(other)} - and ROOT's arithmetic goes bin by bin, so it "
            f"needs the same bins in both"
        )


def _counting(*histograms: Histogram) -> None:
    for histogram in histograms:
        if histogram.kind == "MEAN":
            raise UnsupportedFeatureError(
                f"{histogram.name!r} is a profile, whose bins are means rather than counts, "
                f"and this arithmetic is a histogram's; Profile.merge adds profiles up the "
                f"way hadd does"
            )


def copied(histogram: H, name: str | None = None) -> H:
    """``Clone``: the same histogram, members and all, sharing nothing with this one."""
    made = type(histogram)(histogram.classname, _copy.deepcopy(histogram.members))
    if name is not None:
        made._core["TNamed"]["fName"] = str(name)
    return made


def put_statistics(histogram: Histogram, found: list[float]) -> None:
    """``PutStats``: set the running sums, each where the class keeps it."""
    homes = histogram._moment_homes()
    for name, value in zip(histogram._moment_names(), found):
        homes[name][name] = float(value)


def reset_statistics(histogram: Histogram) -> None:
    """``ResetStats``: throw the running sums away and make them again from the bins.

    The entries go the same way: the total weight, or the effective number
    of entries when the histogram keeps the squares of its weights.
    """
    core = histogram._core
    histogram._moment_homes()["fTsumw"]["fTsumw"] = 0.0
    core["fEntries"] = 1.0
    found = statistics(histogram)
    put_statistics(histogram, found)
    core["fEntries"] = abs(found[0])
    if len(core["fSumw2"]) and found[0] > 0 and found[1] > 0:
        core["fEntries"] = found[0] * found[0] / found[1]


def _forget_limits(histogram: Histogram) -> None:
    histogram._core["fMaximum"] = histogram._core["fMinimum"] = UNSET


def _sum_of_weights(histogram: Histogram) -> float:
    return float(histogram.values().astype(np.float64).sum())


def add(histogram: Histogram, other: Histogram, factor: float) -> None:
    """``TH1::Add(other, factor)``: this plus ``factor`` times ``other``, in place.

    The errors add in quadrature. With a factor of zero or more the running
    sums add too, weighted as the bins are; a negative factor - a subtraction
    - could leave them describing a negative variance, and ROOT rebuilds them
    from the bins instead, as this does.
    """
    _counting(histogram, other)
    compatible(histogram, other)
    entries = abs(histogram.entries + factor * other.entries)
    mine, theirs = statistics(histogram), statistics(other)
    _add_cells(histogram, other, factor)
    if factor < 0:
        reset_statistics(histogram)
        return
    weights = [factor * factor if at == 1 else factor for at in range(len(mine))]
    put_statistics(histogram, [a + w * b for a, w, b in zip(mine, weights, theirs)])
    histogram._core["fEntries"] = entries


def _add_cells(histogram: Histogram, other: Histogram, factor: float) -> None:
    """The bins and squares of ``TH1::Add``: ``other``'s, weighted, added to this one's.

    ``other`` is taken at its normalisation, if it was given one with
    ``SetNormFactor``, as ROOT takes it.
    """
    if histogram._sumw2() is None and other._sumw2() is not None:
        histogram._ensure_sumw2()
    _forget_limits(histogram)
    norm = float(other._core["fNormFactor"])
    scaled = norm / _sum_of_weights(other) if norm != 0 else 1.0
    add_to_cells(histogram._cells(), factor * scaled * other._bins.astype(np.float64))
    squares = histogram._sumw2()
    if squares is not None:
        error = scaled * other._bin_errors()
        squares += factor * factor * error * error


def scale(histogram: Histogram, factor: float, width: bool) -> None:
    """``TH1::Scale(factor)``, in place; ``width`` divides each bin by its size as well.

    A histogram scaled by anything but one starts keeping the squares of its
    weights first, because its errors are no longer the roots of its counts.
    Scaling by bin size is ``Scale(factor, "width")``, after which ROOT - and
    this - make the running sums and the entries again from the bins.
    """
    _counting(histogram)
    if histogram._sumw2() is None and factor != 1.0:
        histogram._ensure_sumw2()
    if width:
        _scale_by_size(histogram, factor)
        return
    cells = histogram._cells()
    store_cells(cells, factor * cells.astype(np.float64))
    squares = histogram._sumw2()
    if squares is not None:
        squares *= factor * factor
    found = statistics(histogram)
    put_statistics(
        histogram, [(factor * factor if at == 1 else factor) * v for at, v in enumerate(found)]
    )
    _forget_limits(histogram)


def _volumes(histogram: Histogram) -> np.ndarray[Any, Any]:
    """The size of every cell, flow and all, in the order the bins are kept."""
    size = np.ones(())
    for axis in histogram.axes:
        size = np.multiply.outer(size, axis.root_widths())
    return size.ravel(order="F")


def _scale_by_size(histogram: Histogram, factor: float) -> None:
    size = _volumes(histogram)
    squares = histogram._sumw2()
    cells = histogram._cells()
    store_cells(cells, factor * cells.astype(np.float64) / size)
    if squares is not None:
        error = np.sqrt(squares) / size
        squares[:] = factor * factor * error * error
    reset_statistics(histogram)
    _forget_limits(histogram)


def multiply(histogram: Histogram, other: Histogram) -> None:
    """``TH1::Multiply(other)``: this times ``other`` bin by bin, in place.

    The relative errors add in quadrature; the running sums are made again
    from the bins, since a product of histograms has no fills behind it.
    """
    _counting(histogram, other)
    compatible(histogram, other)
    if histogram._sumw2() is None and other._sumw2() is not None:
        histogram._ensure_sumw2()
    mine, theirs = histogram._bins.astype(np.float64), other._bins.astype(np.float64)
    squares = histogram._sumw2()
    if squares is not None:
        squares[:] = squares * theirs * theirs + other._variance_cells() * mine * mine
    store_cells(histogram._cells(), mine * theirs)
    _forget_limits(histogram)
    reset_statistics(histogram)


def divide(histogram: Histogram, other: Histogram, binomial: bool) -> None:
    """``TH1::Divide``: this over ``other`` bin by bin, in place; zero where ``other`` is.

    Without ``binomial`` the errors are those of a ratio of independent
    numbers. With it they are ROOT's option ``"B"``: this is taken to be a
    subset of ``other`` - what passed, out of what was tried - and the error
    is the binomial one, zero where every one passed; the entries become
    ``other``'s, as ROOT has them.
    """
    _counting(histogram, other)
    compatible(histogram, other)
    if histogram._sumw2() is None and (binomial or other._sumw2() is not None):
        histogram._ensure_sumw2()
    top, bottom = histogram._bins.astype(np.float64), other._bins.astype(np.float64)
    squares = histogram._sumw2()
    with np.errstate(all="ignore"):
        store_cells(histogram._cells(), np.where(bottom != 0, top / bottom, 0.0))
        if squares is not None:
            spread = _binomial if binomial else _ratio
            found = spread(top, bottom, squares, other._variance_cells())
            squares[:] = np.where(bottom != 0, found, 0.0)
    _forget_limits(histogram)
    reset_statistics(histogram)
    if binomial:
        histogram._core["fEntries"] = other.entries


def _ratio(top: Any, bottom: Any, above: Any, below: Any) -> Any:
    square = bottom * bottom
    return (above * square + below * top * top) / (square * square)


def _binomial(top: Any, bottom: Any, above: Any, below: Any) -> Any:
    tops, bottoms = top * top, bottom * bottom
    found = np.abs(((1.0 - 2.0 * top / bottom) * above + tops * below / bottoms) / bottoms)
    return np.where(top != bottom, found, 0.0)


def reset(histogram: Histogram) -> None:
    """``TH1::Reset``: every bin, square and running sum back to zero, and no entries."""
    store_cells(histogram._cells(), np.zeros(len(histogram._bins)))
    for held in histogram._per_cell():
        held[:] = 0.0
    put_statistics(histogram, [0.0] * len(histogram._moment_names()))
    histogram._core["fEntries"] = 0.0


def normalized(histogram: H, width: bool) -> H:
    """A copy scaled so that its bins add to one - or, with ``width``, its density.

    With ``width`` each bin is also divided by its size, so the bins times
    their sizes add to one: ``h->Scale(1/h->Integral(), "width")``.
    """
    total = integral(histogram, None, None, False)[0]
    if total == 0:
        raise ValueError(
            f"{histogram.name!r} has nothing on its axes to normalise: its bins add to zero"
        )
    made = copied(histogram)
    scale(made, 1.0 / total, width)
    return made


def merge(histograms: Iterable[H]) -> H:
    """``TH1::Merge``: one histogram holding everything the others hold, as ``hadd`` makes.

    The bins, the squares of the weights, the running sums and the entries
    all add up, and so, for profiles, do the sums of weights per bin. All of
    them must be binned alike, and all histograms or all profiles.
    """
    given = list(histograms)
    if not given:
        raise ValueError("merging needs at least one histogram to start from")
    first = given[0]
    for other in given[1:]:
        if other.kind != first.kind:
            raise TypeError(
                f"{first.name!r} and {other.name!r} are a histogram and a profile, and a "
                f"merge adds up one kind of thing"
            )
        compatible(first, other)
    made = copied(first)
    totals, entries = statistics(first), first.entries
    for other in given[1:]:
        totals = [a + b for a, b in zip(totals, statistics(other))]
        entries += other.entries
        made._merge_cells(other)
    put_statistics(made, totals)
    made._core["fEntries"] = entries
    return made
