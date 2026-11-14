"""UHI's indexing of ROOT's histograms: ``h[i, j]``, ``h[a:b, ::sum]``, ``h[...] = values``.

An index is turned into ROOT's bin numbers axis by axis
(:func:`_compute_common_index`): a whole number counted from zero, the
first bin inside the flow, negative from the end; a tag of
:mod:`.tags`, or ``len``, the bin past the last; a slice, the bins from its
start up to its stop, the flow bins included where an end is left open. A
``dict`` names the axes it indexes and leaves the rest whole, and ``...``
stands for every axis not named.

Every axis a number gives back that bin's content. Anything else is a new
histogram cut to the slices as ROOT 6.40's ``ROOT::Internal::Slice`` cuts
one - what falls before the cut gathered into the first bin of the whole
histogram, what falls after it into the last, the statistics made again from
the bins - and then rebinned or summed by each slice's step: ``rebin(n)``
merges bins, ``sum`` sums the axis away. Of a 1-D histogram, a sum is its
``Integral`` over the slice, flow and all where the slice runs to the end.

Setting writes the bins named, as ``SetSliceContent`` does: a scalar into
each bin inside the flow, an array of exactly as many values as the slice
has bins counting its flow into those too, and anything else broadcast over
the bins inside the flow; then the statistics are made again.
"""

from __future__ import annotations

import builtins
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .plotting import _shape
from .tags import _get_axis, _get_axis_len, _overflow, _rebin, _rebinned, _sum, _underflow

__all__: list[str] = []


@contextmanager
def _temporarily_disable_add_directory() -> Iterator[None]:
    """Histograms made inside it go into no directory: ``TH1::AddDirectory(false)`` for a while."""
    from xrdroot.pyroot.core.histcore import Booked

    old_status = Booked.AddDirectoryStatus()
    Booked.AddDirectory(False)
    try:
        yield
    finally:
        Booked.AddDirectory(old_status)


# -- from UHI's index to ROOT's bin numbers ---------------------------------------------------


def _bin_number(self: Any, index: int, axis: int, is_slice_stop: bool) -> int:
    """ROOT's number for UHI's bin ``index``, which counts from the first bin inside the flow."""
    nbins = _get_axis_len(self, axis)
    found = index + nbins if index < 0 else index
    if not 0 <= found < nbins + (1 if is_slice_stop else 0):
        raise IndexError(
            f"Histogram index {index} out of range for axis {axis}. Valid range: (0,{nbins})"
        )
    return found + 1


def _process_index_for_axis(self: Any, index: Any, axis: int, is_slice_stop: bool = False) -> int:
    """ROOT's bin number for one axis' index: a number, a tag, or ``len``."""
    if index is len:
        return _get_axis_len(self, axis) + 1
    if callable(index):
        return int(index(self, axis))
    if isinstance(index, (int, np.integer)):
        return _bin_number(self, int(index), axis, is_slice_stop)
    raise TypeError(f"Unsupported index type: {type(index).__name__}")


def _resolve_slice_indices(
    self: Any, index: slice, axis: int, flow: bool = True
) -> tuple[int, int]:
    """A slice's first bin and the bin it stops before, in ROOT's numbers; an end left open
    reaches into the flow when ``flow`` is asked for."""
    start = (
        _underflow(self, axis) + (0 if flow else 1)
        if index.start is None
        else _process_index_for_axis(self, index.start, axis)
    )
    stop = (
        _overflow(self, axis) + (1 if flow else 0)
        if index.stop is None
        else _process_index_for_axis(self, index.stop, axis, is_slice_stop=True)
    )
    if start < _underflow(self, axis) or stop > _overflow(self, axis) + 1 or start > stop:
        raise IndexError(
            f"Slice indices {start, stop} out of range for axis {axis}. "
            f"Valid range: {_underflow(self, axis), _overflow(self, axis) + 1}"
        )
    return start, stop


def _is_action(index: Any) -> bool:
    """Whether the index is a step alone - ``rebin(n)`` or ``sum`` - meaning the whole axis."""
    return isinstance(index, _rebin) or index is _sum or index is builtins.sum


def _compute_uhi_index(self: Any, index: Any, axis: int, flow: bool = True) -> Any:
    """One axis' index as ROOT's bin number, or as a slice of them keeping its step."""
    if _is_action(index):
        index = slice(None, None, index)
    if isinstance(index, slice):
        start, stop = _resolve_slice_indices(self, index, axis, flow)
        return slice(start, stop, index.step)
    return _process_index_for_axis(self, index, axis)


def _expanded(self: Any, index: Any) -> tuple[Any, ...]:
    """One index per axis: a ``dict``'s axes filled in, and ``...`` spread over the rest."""
    dim = self.GetDimension()
    if isinstance(index, dict):
        index = tuple(index.get(axis, slice(None)) for axis in range(dim))
    if not isinstance(index, tuple):
        index = (index,)
    ellipses = [at for at, item in enumerate(index) if item is Ellipsis]
    if len(ellipses) > 1:
        raise IndexError("Only one ellipsis is allowed in the index.")
    if ellipses:
        at = ellipses[0]
        index = index[:at] + (slice(None),) * (dim - len(index) + 1) + index[at + 1 :]
    if len(index) != dim:
        raise IndexError(f"Expected {dim} indices, got {len(index)}")
    return tuple(index)


def _compute_common_index(self: Any, index: Any, flow: bool = True) -> list[Any]:
    """Every axis' index as ROOT's bin numbers: :func:`_compute_uhi_index` on each."""
    expanded = _expanded(self, index)
    return [_compute_uhi_index(self, item, axis, flow) for axis, item in enumerate(expanded)]


def _get_processed_slices(self: Any, index: Any) -> tuple[list[tuple[int, int]], list[Any]]:
    """Each axis' bins as ``(first, stop)`` and the step to take over them; a single bin is
    one bin summed over."""
    if len(index) != self.GetDimension():
        raise IndexError(f"Expected {self.GetDimension()} indices, got {len(index)}")
    processed_slices, actions = [], []
    for idx in index:
        if isinstance(idx, slice):
            processed_slices.append((idx.start, idx.stop))
            actions.append(idx.step)
        elif isinstance(idx, int):
            processed_slices.append((idx, idx + 1))
            actions.append(_sum)
        else:
            raise TypeError(f"Unsupported index type: {type(idx).__name__}")
    return processed_slices, actions


# -- ROOT::Internal::Slice --------------------------------------------------------------------


def _cell_arrays(hist: Any) -> list[np.ndarray[Any, Any]]:
    """Every array of one number per bin the histogram keeps, each the histogram's own."""
    engine = hist._xrd
    found = [engine._cells()]
    squares = engine._sumw2()
    if squares is not None:
        found.append(squares)
    for name in ("fBinEntries", "fBinSumw2"):  # a profile's entries and squared weights
        held = engine._writable(engine.members, name)
        if held is not None:
            found.append(held)
    return found


def _padded(hist: Any, bounds: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]]]:
    """The bins along x, y and z and the bounds cut along each, an axis the histogram lacks
    being one bin, all of it kept."""
    widths = [*_shape(hist), 1, 1][:3]
    return widths, [*bounds, (0, 1), (0, 1)][:3]


def _flows(
    flat: np.ndarray[Any, Any], widths: list[int], bounds: list[tuple[int, int]]
) -> tuple[Any, Any]:
    """What a cut leaves out, as ``SliceHistoInPlace`` adds it up: every bin before the first
    kept, in ROOT's order of the bins, with those before the cut along x on every later row
    kept; and every bin after the cut along x on each row kept, with all after the last."""
    full = flat.reshape(widths, order="F")
    (sx, ex), (sy, ey), (sz, ez) = bounds
    rows = full[:, sy:ey, sz:ez]
    wide = np.int64 if flat.dtype.kind in "iu" else np.float64
    first = (sz * widths[1] + sy) * widths[0] + sx
    after = ((ez - 1) * widths[1] + ey) * widths[0]
    under = flat[:first].sum(dtype=wide) + rows[:sx, 1:].sum(dtype=wide)
    under += rows[:sx, :1, 1:].sum(dtype=wide)
    over = rows[ex:].sum(dtype=wide) + flat[after:].sum(dtype=wide)
    return under, over


def _sliced_cells(
    flat: np.ndarray[Any, Any], widths: list[int], bounds: list[tuple[int, int]], dim: int
) -> np.ndarray[Any, Any]:
    """One per-bin array cut to ``bounds``, everything left out gathered into its first bin
    and its last, as ROOT 6.40 does (:func:`_flows`)."""
    flat = flat[: math.prod(widths)]
    kept = tuple(slice(start, stop) for start, stop in bounds)
    shape = [stop - start + (2 if axis < dim else 0) for axis, (start, stop) in enumerate(bounds)]
    made = np.zeros(shape, dtype=flat.dtype)
    inner = tuple(slice(1, -1) if axis < dim else slice(None) for axis in range(3))
    made[inner] = flat.reshape(widths, order="F")[kept]
    out = made.ravel(order="F")
    out[[0, -1]] = np.array(_flows(flat, widths, bounds)).astype(flat.dtype)
    return out


def _sliced_edges(hist: Any, bounds: list[tuple[int, int]]) -> list[Any]:
    """``SetBins``'s arguments for the axes cut to ``bounds``: each a count and its edges."""
    found: list[Any] = []
    for axis, (start, stop) in enumerate(bounds):
        tAxis = _get_axis(hist, axis)
        edges = [tAxis.GetBinLowEdge(b) for b in range(start, stop + 1)]
        found += [stop - start, np.array(edges, dtype=np.float64)]
    return found


def _slice(hist: Any, slices: list[tuple[int, int]]) -> Any:
    """``ROOT::Internal::Slice``: a copy of the histogram cut to the bins ``slices`` give, each
    ``(first, stop)`` kept inside the flow."""
    bounds = [
        (max(1, start), min(_overflow(hist, axis), stop))
        for axis, (start, stop) in enumerate(slices)
    ]
    widths, padded = _padded(hist, bounds)
    dim = hist.GetDimension()
    cells = [_sliced_cells(each, widths, padded, dim) for each in _cell_arrays(hist)]
    with _temporarily_disable_add_directory():
        made = hist.Clone()
    made.SetBins(*_sliced_edges(hist, bounds))
    for into, values in zip(_cell_arrays(made), cells, strict=True):
        into[:] = values
    made.ResetStats()
    return made


# -- the steps: rebin and sum ----------------------------------------------------------------


def _is_sum(action: Any) -> bool:
    return action is _sum or action is builtins.sum


def _check_actions(actions: list[Any]) -> None:
    unknown = [a for a in actions if not (a is None or _is_sum(a) or isinstance(a, _rebin))]
    if unknown:
        raise ValueError(f"Unsupported action detected in actions {actions}")


def _integral_bins(index: list[Any], expanded: tuple[Any, ...]) -> list[int]:
    """The bins a 1-D sum runs over: the slice's own, its last one the overflow when the
    slice was left open at the top."""
    start, stop = index[0].start, index[0].stop
    given = expanded[0]
    open_top = not isinstance(given, slice) or given.stop is None
    return [start, stop - (0 if open_top else 1)]


def _apply_actions(
    hist: Any, actions: list[Any], index: list[Any], expanded: tuple[Any, ...], original: Any
) -> Any:
    """The sliced histogram summed and rebinned as the slices' steps say."""
    _check_actions(actions)
    summed = tuple(axis for axis, action in enumerate(actions) if _is_sum(action))
    if summed and original.GetDimension() == 1:
        return _sum(original, summed, _integral_bins(index, expanded))
    if summed:
        hist = _sum(hist, summed)
    groups = _rebin_groups(actions)
    return _rebinned(hist, groups) if groups else hist


def _rebin_groups(actions: list[Any]) -> list[int]:
    """Each axis left after the sums merged by its ``rebin``'s group, or by one; none when no
    axis is rebinned."""
    if not any(isinstance(action, _rebin) for action in actions):
        return []
    return [a.ngroup if isinstance(a, _rebin) else 1 for a in actions if not _is_sum(a)]


def _slice_get(self: Any, index: list[Any], expanded: tuple[Any, ...]) -> Any:
    """``h[slices]``: the histogram cut to the slices, then summed or rebinned by their steps."""
    processed_slices, actions = _get_processed_slices(self, index)
    target = _slice(self, processed_slices)
    return _apply_actions(target, actions, index, expanded, self)


# -- setting ---------------------------------------------------------------------------------


def _slice_shape(self: Any, index: list[Any]) -> tuple[int, ...]:
    processed_slices, _ = _get_processed_slices(self, index)
    return tuple(stop - start for start, stop in processed_slices)


def _set_slice_content(self: Any, values: np.ndarray[Any, Any], index: list[Any]) -> None:
    """``SetSliceContent``: the bins of the slices set, then the statistics made again."""
    processed_slices, _ = _get_processed_slices(self, index)
    full = self.GetArray()[: self.GetNcells()].reshape(_shape(self), order="F")
    full[tuple(slice(start, stop) for start, stop in processed_slices)] = values
    self.ResetStats()


def _slice_set(self: Any, index: list[Any], expanded: tuple[Any, ...], value: Any) -> None:
    """``h[slices] = value``: a value per bin, or one broadcast over them all.

    An array with exactly as many values as the slices have bins counting
    the flow sets the flow too; anything else is set inside the flow alone.
    """
    given = np.asarray(value, dtype=np.float64)
    if np.ndim(value) == 0 or given.size != math.prod(_slice_shape(self, index)):
        index = [_compute_uhi_index(self, item, axis, False) for axis, item in enumerate(expanded)]
    shape = _slice_shape(self, index)
    try:
        values = given.reshape(shape) if given.size == math.prod(shape) else None
        values = np.broadcast_to(given, shape) if values is None else values
    except ValueError:
        raise ValueError(f"Expected {math.prod(shape)} bin values, got {given.size}") from None
    _set_slice_content(self, values, index)


# -- the protocol ----------------------------------------------------------------------------


def _getitem(self: Any, index: Any) -> Any:
    """``h[index]``: a bin's content, or a histogram cut, rebinned or summed (see above)."""
    expanded = _expanded(self, index)
    uhi_index = [_compute_uhi_index(self, item, axis) for axis, item in enumerate(expanded)]
    if all(isinstance(i, int) for i in uhi_index):
        return self.GetBinContent(*uhi_index)
    return _slice_get(self, uhi_index, expanded)


def _setitem(self: Any, index: Any, value: Any) -> None:
    """``h[index] = value``: one bin's ``SetBinContent``, or the bins of a slice set at once."""
    expanded = _expanded(self, index)
    uhi_index = [_compute_uhi_index(self, item, axis) for axis, item in enumerate(expanded)]
    if all(isinstance(i, int) for i in uhi_index):
        self.SetBinContent(self.GetBin(*uhi_index), value)
        return
    _slice_set(self, uhi_index, expanded, value)


def _iter(self: Any) -> Iterator[float]:
    """Every bin's content, the flow bins too, the last axis running fastest."""
    return iter(np.asarray(self.values(flow=True), dtype=np.float64).ravel().tolist())
