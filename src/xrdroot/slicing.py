"""Indexing a histogram the way ``hist`` and ``boost-histogram`` do: UHI's ``h[...]``.

    >>> h[3]                           # the fourth bin's content          # doctest: +SKIP
    >>> h[underflow], h[loc(1.5)]      # the underflow, the bin 1.5 is in  # doctest: +SKIP
    >>> h[2:8], h[loc(0.5):loc(2.0)]   # a cut, by bin or by coordinate    # doctest: +SKIP
    >>> h[::rebin(2)], h2[:, sum]      # bins merged, an axis summed away  # doctest: +SKIP
    >>> h2[{0: slice(2, 8), 1: sum}]   # the same, one axis at a time      # doctest: +SKIP
    >>> h[...] = values                # every bin set at once             # doctest: +SKIP

The rules are UHI's (https://uhi.readthedocs.io/en/latest/indexing.html). A
whole number is a bin, counted from zero, negative from the end; a locator -
:class:`loc`, :data:`underflow`, :data:`overflow`, or anything UHI calls one,
a callable taking the axis and giving back its index - is one too, and can
name a flow bin. A slice cuts the axis down, and what it cuts away is added
to the flow bins rather than lost; a :class:`rebin` step merges bins, the
ones a group leaves over going to the overflow; a ``sum`` step sums the axis
away, flow and all unless the slice says where to start and stop - so
``h[0:len:sum]`` leaves the flow out. A bin picked on an axis of a histogram
of more than one is that one bin summed over, and picking or summing every
axis gives back a number.

The bookkeeping is ROOT's wherever ROOT has the same operation. An axis
summed away is ROOT's projection - ``ProjectionX`` with a range - and keeps
its rules for the running sums and the entries. A cut or a rebinning is
``TH1::Rebin`` with bins dropping into the flow: the entries stay, as every
fill is still in some bin, and the running sums are kept when nothing moved
into the flow and made again from the bins when anything did. Setting bins
is ``SetBinContent`` once a bin: one more entry each, and the running sums
made again from the bins, the squares of the weights left as they were.
"""

from __future__ import annotations

import copy as _copy
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import numpy as np

from .arithmetic import copied
from .booking import Binning
from .errors import UnsupportedFeatureError
from .filling import running, store_cells
from .interp import ARRAYS
from .reshaping import _low_edge, _new_axis, projection

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from .hist import Axis, Histogram

__all__ = ["Locator", "loc", "overflow", "rebin", "underflow"]

#: The arrays of one number per bin a profile keeps beside its contents and squares.
PROFILE_CELLS = ("fBinEntries", "fBinSumw2")


class Locator:
    """Something that finds a bin on an axis: UHI's locator, with an offset in bins.

    ``loc(1.5) + 1`` is the bin after the one 1.5 is in. Calling one with an
    axis gives its index there: ``-1`` for the underflow, the number of bins
    for the overflow, which is how UHI numbers them.
    """

    __slots__ = ("offset",)

    def __init__(self, offset: int = 0) -> None:
        #: How many bins past what is located to go, backwards if negative.
        self.offset = int(offset)

    def __add__(self, other: int) -> Locator:
        made = _copy.copy(self)
        made.offset += int(other)
        return made

    def __sub__(self, other: int) -> Locator:
        return self + -int(other)

    def _located(self, axis: Any) -> int:
        raise NotImplementedError

    def __call__(self, axis: Any) -> int:
        return self._located(axis) + self.offset


class loc(Locator):
    """The bin a coordinate falls in: ``h[loc(1.5)]``, ``h[loc(0.0):loc(2.0)]``.

    It is spelled in lower case, as UHI, ``hist`` and ``boost-histogram``
    all spell it, and so is :class:`rebin`.
    """

    __slots__ = ("value",)

    def __init__(self, value: float, offset: int = 0) -> None:
        super().__init__(offset)
        #: The coordinate whose bin this is.
        self.value = float(value)

    def _located(self, axis: Any) -> int:
        return int(axis.index(self.value))

    def __repr__(self) -> str:
        return f"loc({self.value!r})" + (f" + {self.offset}" if self.offset else "")


class _Underflow(Locator):
    __slots__ = ()

    def _located(self, axis: Any) -> int:
        return -1

    def __repr__(self) -> str:
        return "underflow"


class _Overflow(Locator):
    __slots__ = ()

    def _located(self, axis: Any) -> int:
        return len(axis)

    def __repr__(self) -> str:
        return "overflow"


#: The bin below an axis, for what fell off its low end: ``h[underflow]``.
underflow = _Underflow()

#: The bin above an axis, for what fell off its high end: ``h[overflow]``.
overflow = _Overflow()


class rebin:
    """A slice's step that merges every ``factor`` bins into one: ``h[::rebin(2)]``."""

    __slots__ = ("factor",)

    def __init__(self, factor: int) -> None:
        if not _whole(factor) or factor < 1:
            raise ValueError(
                f"rebin({factor!r}) does not merge bins: give it a whole number of them, one "
                f"or more"
            )
        #: How many bins become one.
        self.factor = int(factor)

    def __repr__(self) -> str:
        return f"rebin({self.factor})"


def _whole(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


# -- reading an index ---------------------------------------------------------


class _Kept(NamedTuple):
    """An axis kept: bins ``start`` to ``stop``, from zero, merged ``factor`` at a time."""

    start: int
    stop: int
    factor: int


class _Summed(NamedTuple):
    """An axis summed away over ``first`` to ``last``, counted flow and all, ``last`` excluded.

    ``picked`` is a single bin asked for by itself, which is all an index
    that sets bins may hold besides slices.
    """

    first: int
    last: int
    picked: bool


_Spec = Union[_Kept, _Summed]


def _spread(histogram: Histogram, index: Any) -> list[Any]:
    """One item per axis: a dict by axis number, or a tuple with ``...`` for the rest."""
    count = len(histogram.axes)
    if isinstance(index, dict):
        return _by_axis(histogram, index)
    given = list(index) if isinstance(index, tuple) else [index]
    dots = [at for at, item in enumerate(given) if item is Ellipsis]
    if len(dots) > 1:
        raise IndexError("an index may hold one ... and this one holds more")
    if dots:
        at = dots[0]
        given[at : at + 1] = [slice(None)] * (count - len(given) + 1)
    if len(given) > count:
        raise IndexError(
            f"{histogram.name!r} has {count} axes, and an index of {len(given)} items has "
            f"more of them than that"
        )
    return given + [slice(None)] * (count - len(given))


def _by_axis(histogram: Histogram, index: dict[Any, Any]) -> list[Any]:
    spread: list[Any] = [slice(None)] * len(histogram.axes)
    for axis, item in index.items():
        if not _whole(axis) or not 0 <= axis < len(spread):
            raise IndexError(
                f"{axis!r} is not an axis of {histogram.name!r}, which has {len(spread)}, "
                f"numbered from zero"
            )
        spread[axis] = item
    return spread


def _position(axis: Axis, item: Any) -> int:
    """The one bin an item names, from ``-1`` for the underflow to ``nbins`` for the overflow."""
    if _whole(item):
        found = int(item) + (axis.nbins if item < 0 else 0)
        if not 0 <= found < axis.nbins:
            raise IndexError(f"bin {item} is not on an axis of {axis.nbins} bins")
        return found
    if callable(item):
        found = int(item(axis))
        if not -1 <= found <= axis.nbins:
            raise IndexError(
                f"{item!r} locates bin {found}, and an axis of {axis.nbins} bins has them from "
                f"-1, its underflow, to {axis.nbins}, its overflow"
            )
        return found
    raise TypeError(
        f"a {type(item).__name__} does not pick a bin: give a whole number, a locator such "
        f"as loc(x), underflow or overflow, a slice, or sum"
    )


def _endpoint(axis: Axis, value: Any, default: int) -> int:
    """A slice's start or stop, from zero: a number, ``len``, a locator, or nothing."""
    if value is None:
        return default
    if value is len:
        return axis.nbins
    if _whole(value):
        return int(value) + (axis.nbins if value < 0 else 0)
    if callable(value):
        return int(value(axis))
    raise TypeError(
        f"a {type(value).__name__} does not end a slice: give a whole number, len, or a "
        f"locator such as loc(x)"
    )


def _factor(step: Any) -> int:
    if step is None:
        return 1
    factor: Any = getattr(step, "factor", None)
    if _whole(factor) and factor >= 1:
        return int(factor)
    raise TypeError(
        f"{step!r} is not a step a histogram's slice takes: give rebin(n) to merge bins, "
        f"or sum to sum the axis away"
    )


def _clamped(value: int, low: int, high: int) -> int:
    return min(max(value, low), high)


def _sliced(axis: Axis, item: slice) -> _Spec:
    nbins = axis.nbins
    if item.step is sum:
        first = _clamped(_endpoint(axis, item.start, -1), -1, nbins + 1)
        last = _clamped(_endpoint(axis, item.stop, nbins + 1), -1, nbins + 1)
        if last <= first:
            raise ValueError(
                f"{item!r} sums no bins of {axis.name!r}: its stop is not past its start"
            )
        return _Summed(first + 1, last + 1, False)
    factor = _factor(item.step)
    start = _clamped(_endpoint(axis, item.start, 0), 0, nbins)
    stop = _clamped(_endpoint(axis, item.stop, nbins), 0, nbins)
    if (stop - start) // factor < 1:
        raise ValueError(
            f"{item!r} leaves no bins of {axis.name!r}: {max(stop - start, 0)} bins from "
            f"{start} make no group of {factor}"
        )
    return _Kept(start, stop, factor)


def _spec(axis: Axis, item: Any) -> _Spec:
    if item is sum:
        return _Summed(0, axis.nbins + 2, False)
    if isinstance(item, slice):
        return _sliced(axis, item)
    found = _position(axis, item)
    return _Summed(found + 1, found + 2, True)


def _specs(histogram: Histogram, index: Any) -> list[_Spec]:
    return [_spec(axis, item) for axis, item in zip(histogram.axes, _spread(histogram, index))]


# -- reading bins -------------------------------------------------------------


def get(histogram: Histogram, index: Any) -> Any:
    """``h[index]``: a bin's content, a sum of bins, or a new histogram cut to the index."""
    specs = _specs(histogram, index)
    kept = [spec for spec in specs if isinstance(spec, _Kept)]
    if not kept:
        return _number(histogram, specs)
    made = histogram
    if len(kept) < len(specs):
        made = _projected(histogram, specs)
    return _cut(made, kept)


def _number(histogram: Histogram, specs: Sequence[_Spec]) -> float:
    """What an index picking or summing every axis gives: one bin, or the sum of a block."""
    cut = tuple(slice(spec.first, spec.last) for spec in specs if isinstance(spec, _Summed))
    if all(isinstance(spec, _Summed) and spec.picked for spec in specs):
        return float(histogram.values(flow=True)[cut].ravel()[0])
    if histogram.kind == "MEAN":
        raise UnsupportedFeatureError(
            f"{histogram.name!r} is a profile, whose bins are means, and a sum of means is "
            f"not a thing: pick a bin, or slice without summing"
        )
    return running(0.0, histogram.values(flow=True).astype(np.float64)[cut].ravel())


def _projected(histogram: Histogram, specs: Sequence[_Spec]) -> Histogram:
    """The axes summed away, as ROOT's projection with a range sums them."""
    letters = "".join("xyz"[at] for at, spec in enumerate(specs) if isinstance(spec, _Kept))
    ranges = {
        "xyz"[at]: (spec.first, spec.last - 1)
        for at, spec in enumerate(specs)
        if isinstance(spec, _Summed)
    }
    return projection(histogram, letters, histogram.name, ranges)


def _binning(axis: Axis, spec: _Kept) -> Binning:
    """The axis a cut and a rebinning leave, its edges ROOT's where ROOT would work them out."""
    groups = (spec.stop - spec.start) // spec.factor
    end = spec.start + groups * spec.factor
    if not axis.even:
        edges = axis.edges()[spec.start : end + 1 : spec.factor]
        return Binning(groups, float(edges[0]), float(edges[-1]), edges)
    low = axis.low if spec.start == 0 else _low_edge(axis, spec.start + 1)
    high = axis.high if end == axis.nbins else _low_edge(axis, end + 1)
    return Binning(groups, low, high, np.zeros(0))


def _moves(axis: Axis, spec: _Kept) -> bool:
    """Whether a cut or rebinning sends any bin of the axis into its flow."""
    groups = (spec.stop - spec.start) // spec.factor
    return spec.start > 0 or spec.start + groups * spec.factor < axis.nbins


def _sequential(block: np.ndarray[Any, Any], axis: int) -> np.ndarray[Any, Any]:
    """A sum along one axis, taken in turn as ROOT's loops take it."""
    return np.take(np.add.accumulate(block, axis=axis), -1, axis=axis)


def _cut_axis(values: np.ndarray[Any, Any], axis: int, spec: _Kept) -> np.ndarray[Any, Any]:
    """One axis, flow and all, cut down and merged, what is cut away added to its flow."""
    moved = np.moveaxis(values, axis, 0)
    groups = (spec.stop - spec.start) // spec.factor
    end = spec.start + groups * spec.factor
    body = moved[1 + spec.start : 1 + end]
    merged = _sequential(body.reshape(groups, spec.factor, *body.shape[1:]), 1)
    under = _sequential(moved[: 1 + spec.start], 0)
    over = _sequential(moved[1 + end :], 0)
    return np.moveaxis(np.concatenate((under[None], merged, over[None])), 0, axis)


def _cells_of(histogram: Histogram) -> list[tuple[dict[str, Any], str]]:
    """Where every array of one number per bin is kept: contents first, then the rest."""
    found = [(histogram._home, histogram._key)]
    if histogram._sumw2() is not None:
        found.append((histogram._core, "fSumw2"))
    for name in PROFILE_CELLS:
        if len(histogram.members.get(name, ())) == len(histogram._bins):
            found.append((histogram.members, name))
    return found


def _cut(histogram: Histogram, kept: Sequence[_Kept]) -> Histogram:
    """A new histogram of every per-bin array cut and merged along each axis."""
    binnings = [_binning(axis, spec) for axis, spec in zip(histogram.axes, kept)]
    moved = any(_moves(axis, spec) for axis, spec in zip(histogram.axes, kept))
    made = copied(histogram)
    cells = math.prod(binning.nbins + 2 for binning in binnings)
    for position, (home, key) in enumerate(_cells_of(made)):
        dtype = np.dtype(ARRAYS[key].typename) if position == 0 else np.dtype(np.float64)
        home[key] = _cut_cells(made, home[key], kept, np.zeros(cells, dtype=dtype))
    for letter, binning in zip("XYZ", binnings):
        _new_axis(made, letter, binning)
    made._core["fNcells"] = cells
    if moved:
        made._moment_homes()["fTsumw"]["fTsumw"] = 0.0
    return type(made)(made.classname, made.members)


def _cut_cells(
    histogram: Histogram, held: Any, kept: Sequence[_Kept], into: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """One per-bin array cut along every axis, stored in ``into`` as its type stores it."""
    values = histogram._shaped(np.asarray(held, dtype=np.float64), True)
    for axis, spec in enumerate(kept):
        values = _cut_axis(values, axis, spec)
    store_cells(into, values.ravel(order="F"))
    return into


# -- setting bins -------------------------------------------------------------


def _target(histogram: Histogram, specs: Sequence[_Spec], given: Any) -> tuple[Any, ...]:
    """The block of bins, flow counted, a value set by ``h[index] = value`` goes into."""
    block: list[Any] = []
    for spec in specs:
        if isinstance(spec, _Summed) and not spec.picked:
            raise ValueError("setting bins takes bins and slices, and sum sets none")
        if isinstance(spec, _Kept) and spec.factor != 1:
            raise ValueError("setting bins takes bins and slices, and rebin sets none")
        kept = isinstance(spec, _Kept)
        block.append(slice(spec[0] + 1, spec[1] + 1) if kept else spec[0])
    return _with_flow(histogram, block, given)


def _with_flow(histogram: Histogram, block: list[Any], given: Any) -> tuple[Any, ...]:
    """A plain slice over a whole axis takes its flow too, when the value has two more there.

    That is UHI's rule for setting a histogram's flow bins along with the rest.
    """
    kept = [at for at, part in enumerate(block) if isinstance(part, slice)]
    if np.ndim(given) != len(kept):
        return tuple(block)
    for place, at in enumerate(kept):
        nbins = histogram.axes[at].nbins
        if block[at] == slice(1, nbins + 1) and np.shape(given)[place] == nbins + 2:
            block[at] = slice(0, nbins + 2)
    return tuple(block)


def put(histogram: Histogram, index: Any, value: Any) -> None:
    """``h[index] = value``: bins set as ``SetBinContent`` sets them, one entry more a bin."""
    if histogram.kind == "MEAN":
        raise UnsupportedFeatureError(
            f"{histogram.name!r} is a profile, whose bins are means of what was filled, and "
            f"a mean cannot be set without saying what it was the mean of; fill it instead"
        )
    given = np.asarray(value, dtype=np.float64)
    block = _target(histogram, _specs(histogram, index), given)
    full = histogram.values(flow=True).astype(np.float64)
    try:
        full[block] = given
    except ValueError:
        raise ValueError(
            f"values of shape {given.shape} do not fit the {np.shape(full[block])} bins they "
            f"are set into: give one per bin, or one for all of them"
        ) from None
    store_cells(histogram._cells(), full.ravel(order="F"))
    histogram._core["fEntries"] = histogram.entries + int(np.size(full[block]))
    histogram._moment_homes()["fTsumw"]["fTsumw"] = 0.0
