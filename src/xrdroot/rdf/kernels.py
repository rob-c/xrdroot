"""Row-wise arithmetic on collections kept flat: one array of values, one of offsets.

Everything ``ROOT::VecOps`` does to an ``RVec`` it does one entry at a time,
in a C++ loop. Here a whole batch of entries is one flat array of every
element and the ``offsets`` saying where each entry's elements start and
stop - the layout of :class:`~xrdroot.Jagged` - and every operation is a few
NumPy calls over all of them at once: a reduction is a ``reduceat`` over the
rows that have elements, a selection is a mask over the flat values and a
count of what each row kept, an index is an offset into the flat array.

The offsets here are always *compact* - they start at zero and end at the
number of values - which is what :func:`compact` makes of any ``Jagged``.
Both :mod:`.vecops`, for Python code, and the expression language, for
strings, are built on these, so the two give the same numbers.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from ..tree import Jagged

__all__ = [
    "compact",
    "offsets_of",
    "rows_of",
    "positions",
    "reduced",
    "summed",
    "means",
    "variances",
    "arg_extreme",
    "masked",
    "taken",
    "first_n",
    "nonzero",
    "argsorted",
    "reversed_index",
    "concatenated",
    "dropped",
    "counting",
    "combinations",
    "cartesian",
    "delta_phi",
    "invariant_mass",
    "pair_masses",
]

Array = Any


def compact(rows: Jagged) -> tuple[Array, Array]:
    """A ``Jagged``'s values and offsets, with the offsets starting at zero."""
    offsets = rows.offsets
    if len(offsets) and offsets[0] == 0 and offsets[-1] == len(rows.content):
        return rows.content, offsets
    return rows.flat, offsets - offsets[0]


def offsets_of(lengths: Array) -> Array:
    """The offsets of rows of these lengths."""
    offsets = np.zeros(len(lengths) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def rows_of(offsets: Array) -> Array:
    """The row each element belongs to."""
    count = len(offsets) - 1
    return np.repeat(np.arange(count, dtype=np.int64), np.diff(offsets))


def positions(offsets: Array) -> Array:
    """Where each element sits within its own row, counting from zero."""
    lengths = np.diff(offsets)
    return np.arange(offsets[-1], dtype=np.int64) - np.repeat(offsets[:-1], lengths)


def reduced(
    ufunc: Any, content: Array, offsets: Array, empty: Any, dtype: Any = None
) -> tuple[Array, Array]:
    """One value per row by ``ufunc`` over its elements, and which rows had none.

    A row with no elements takes ``empty``. ``reduceat`` walks each row's
    elements in order, one after another, as the C++ loop does.
    """
    lengths = np.diff(offsets)
    filled = lengths > 0
    kind = content.dtype if dtype is None else np.dtype(dtype)
    out = np.full(len(lengths), empty, dtype=kind)
    if filled.any():
        out[filled] = ufunc.reduceat(content.astype(kind, copy=False), offsets[:-1][filled])
    return out, ~filled


def summed(content: Array, offsets: Array, start: Any = None) -> Array:
    """``Sum``: each row's elements added up, in the elements' own type.

    Booleans are counted, as integers. ``start`` is the value a sum starts
    from, whose type it then takes, as ``Sum(v, 0.)`` sums in ``double``.
    """
    kind = np.int64 if content.dtype == np.bool_ else content.dtype
    if start is not None:
        kind = np.asarray(start).dtype
    out, _ = reduced(np.add, content, offsets, 0, kind)
    return out if start is None else out + np.asarray(start, kind)


def means(content: Array, offsets: Array) -> Array:
    """``Mean``: each row's average in ``double``, and zero for a row with nothing in it."""
    total, _ = reduced(np.add, content, offsets, 0.0, np.float64)
    lengths = np.diff(offsets)
    return total / np.maximum(lengths, 1)


def variances(content: Array, offsets: Array) -> Array:
    """``Var``: each row's spread about its mean, over ``n - 1``; zero for fewer than two."""
    values = content.astype(np.float64)
    lengths = np.diff(offsets)
    squares, _ = reduced(np.add, values * values, offsets, 0.0)
    total, _ = reduced(np.add, values, offsets, 0.0)
    size = lengths.astype(np.float64)
    spread = (squares - total * total / np.maximum(size, 1.0)) / np.maximum(size - 1.0, 1.0)
    return np.where(lengths < 2, 0.0, spread)


def arg_extreme(content: Array, offsets: Array, ufunc: Any) -> Array:
    """``ArgMax``/``ArgMin``: where in each row its first largest (smallest) element is.

    A row with no elements says ``0``, as ``std::distance`` from the start
    of an empty range to its end does.
    """
    extreme, _ = reduced(ufunc, content, offsets, 0)
    rows = rows_of(offsets)
    hits = np.flatnonzero(content == extreme[rows])
    out = np.zeros(len(offsets) - 1, np.int64)
    found, first = np.unique(rows[hits], return_index=True)
    out[found] = hits[first] - offsets[:-1][found]
    return out


def masked(content: Array, offsets: Array, keep: Array) -> tuple[Array, Array]:
    """``v[mask]``: the elements whose mask is true, rows kept apart."""
    keep = np.asarray(keep) != 0
    lengths = np.bincount(rows_of(offsets)[keep], minlength=len(offsets) - 1)
    return content[keep], offsets_of(lengths)


def taken(
    content: Array, offsets: Array, index: Array, index_offsets: Array
) -> tuple[Array, Array, Array]:
    """``Take(v, i)``: the elements at the positions ``i`` names, and which rows ran off.

    A row whose indices go past its end, or before its start, is marked; the
    value taken there is meaningless, and is the row's first, or zero.
    """
    rows = rows_of(index_offsets)
    local = np.asarray(index).astype(np.int64)
    lengths = np.diff(offsets)
    inside = (local >= 0) & (local < lengths[rows])
    bad = np.bincount(rows[~inside], minlength=len(offsets) - 1) > 0
    at = np.where(inside, offsets[:-1][rows] + local, 0)
    picked = content[at] if len(content) else np.zeros(len(at), content.dtype)
    return picked, index_offsets, bad


def first_n(content: Array, offsets: Array, count: Array, default: Any = None) -> Any:
    """``Take(v, n)``: the first ``n`` elements, or the last ``-n``; with a default to pad.

    Gives the values, their offsets, and which rows are too short for what
    was asked of them when there is no default to fill them with.
    """
    lengths = np.diff(offsets)
    want = np.abs(np.asarray(count, dtype=np.int64)) * np.ones(len(lengths), np.int64)
    out = offsets_of(want)
    local = positions(out)
    rows = rows_of(out)
    from_end = np.broadcast_to(np.asarray(count) < 0, lengths.shape)[rows]
    local = np.where(from_end, lengths[rows] - want[rows] + local, local)
    short = lengths < want
    if default is None:
        picked, _, bad = taken(content, offsets, local, out)
        return picked, out, bad | short
    inside = (local >= 0) & (local < lengths[rows])
    at = np.where(inside, offsets[:-1][rows] + local, 0)
    kind = np.result_type(content.dtype, np.asarray(default).dtype)
    source = content.astype(kind) if len(content) else np.zeros(1, kind)
    return np.where(inside, source[at], default), out, np.zeros(len(lengths), np.bool_)


def nonzero(content: Array, offsets: Array) -> tuple[Array, Array]:
    """``Nonzero``: the positions in each row of its elements that are not zero."""
    return masked(positions(offsets), offsets, content != 0)


def argsorted(content: Array, offsets: Array) -> Array:
    """``Argsort``: each row's positions in the order that sorts it, ties kept in order."""
    order = np.lexsort((content, rows_of(offsets)))
    return order - offsets[:-1][rows_of(offsets)]


def reversed_index(offsets: Array) -> Array:
    """Where each element of a reversed row comes from, in the flat values."""
    rows = rows_of(offsets)
    return offsets[1:][rows] - 1 - positions(offsets)


def concatenated(first: tuple[Array, Array], second: tuple[Array, Array]) -> tuple[Array, Array]:
    """``Concatenate``: each row of the first with the same row of the second after it."""
    (a, a_offsets), (b, b_offsets) = first, second
    kind = np.result_type(a.dtype, b.dtype)
    lengths = np.diff(a_offsets) + np.diff(b_offsets)
    out = offsets_of(lengths)
    values = np.empty(out[-1], kind)
    starts = out[:-1]
    values[starts[rows_of(a_offsets)] + positions(a_offsets)] = a
    shift = np.diff(a_offsets)
    values[(starts + shift)[rows_of(b_offsets)] + positions(b_offsets)] = b
    return values, out


def dropped(
    content: Array, offsets: Array, index: Array, index_offsets: Array
) -> tuple[Array, Array]:
    """``Drop``: each row without the elements at the positions named; bad ones ignored."""
    rows = rows_of(index_offsets)
    local = np.asarray(index).astype(np.int64)
    lengths = np.diff(offsets)
    inside = (local >= 0) & (local < lengths[rows])
    keep = np.ones(len(content), np.bool_)
    keep[(offsets[:-1][rows] + local)[inside]] = False
    return masked(content, offsets, keep)


def counting(count: Array, begin: Any = 0, step: Any = 1) -> tuple[Array, Array]:
    """``Range(n)`` and ``Range(begin, end)``: each row counting up, a step at a time."""
    start = np.asarray(begin, dtype=np.int64)
    stop = np.asarray(count, dtype=np.int64)
    stride = np.asarray(step, dtype=np.int64)
    size = np.maximum(0, -((start - stop) // stride))
    lengths = np.broadcast_to(size, np.broadcast(start, stop, stride).shape)
    out = offsets_of(lengths.ravel())
    rows = rows_of(out)
    first = np.broadcast_to(start, lengths.shape).ravel()[rows]
    every = np.broadcast_to(stride, lengths.shape).ravel()[rows]
    return first + positions(out) * every, out


#: The combinations of ``range(n)`` taken ``k`` at a time, made once for each pair.
_COMBINATIONS: dict[tuple[int, int], Array] = {}


def _table(length: int, k: int) -> Array:
    key = (length, k)
    if key not in _COMBINATIONS:
        made = list(itertools.combinations(range(length), k))
        _COMBINATIONS[key] = np.asarray(made, dtype=np.int64).reshape(len(made), k)
    return _COMBINATIONS[key]


def combinations(offsets: Array, k: int) -> tuple[list[Array], Array]:
    """``Combinations(v, k)``: every choice of ``k`` positions of each row, in order.

    Gives ``k`` runs of positions - the first of each combination, the
    second, and so on - and the offsets they share. Rows of one length share
    one table of combinations, so the work is one lookup per length there is.
    """
    lengths = np.diff(offsets)
    counts = np.asarray([len(_table(int(size), k)) for size in range(lengths.max(initial=0) + 1)])
    out = offsets_of(counts[lengths])
    picks = np.zeros((out[-1], k), np.int64)
    for size in np.unique(lengths):
        table = _table(int(size), k)
        where = np.flatnonzero(lengths == size)
        at = (out[:-1][where][:, None] + np.arange(len(table))).ravel()
        picks[at] = np.tile(table, (len(where), 1))
    return [picks[:, i] for i in range(k)], out


def cartesian(first: Array, second: Array) -> tuple[list[Array], Array]:
    """``Combinations(v1, v2)``: every pairing of a position of one row with one of the other."""
    across, down = np.diff(first), np.diff(second)
    out = offsets_of(across * down)
    local = positions(out)
    width = down[rows_of(out)]
    return [local // np.maximum(width, 1), local % np.maximum(width, 1)], out


def delta_phi(first: Array, second: Array, c: Any = np.pi) -> Array:
    """``DeltaPhi(a, b)``: ``b - a`` folded into ``[-c, c]``, as ``ROOT::VecOps`` folds it."""
    kind = np.result_type(first, second)
    turn = np.fmod(np.asarray(second, np.float64) - np.asarray(first, np.float64), 2.0 * c)
    turn = np.where(turn < -c, turn + 2.0 * c, np.where(turn > c, turn - 2.0 * c, turn))
    return turn.astype(kind if kind.kind == "f" else np.float64)


def _momentum(pt: Array, eta: Array, phi: Array, mass: Array) -> tuple[Array, ...]:
    """A particle's ``(x, y, z, e)`` from ``(pt, eta, phi, mass)``, in its own precision."""
    with np.errstate(all="ignore"):
        x = pt * np.cos(phi)
        y = pt * np.sin(phi)
        z = pt * np.sinh(eta)
        return x, y, z, np.sqrt(x * x + y * y + z * z + mass * mass)


def _mass(x: Array, y: Array, z: Array, e: Array) -> Array:
    """The invariant mass of a four-momentum, with the ``(+, -, -, -)`` metric."""
    with np.errstate(all="ignore"):
        return np.sqrt(e * e - x * x - y * y - z * z)


def invariant_mass(pt: Array, eta: Array, phi: Array, mass: Array, offsets: Array) -> Array:
    """``InvariantMass``: of the sum of every particle in each row, one number per row."""
    kind = np.result_type(pt, eta, phi, mass)
    parts = _momentum(*(np.asarray(each, kind) for each in (pt, eta, phi, mass)))
    sums = [reduced(np.add, part, offsets, 0, kind)[0] for part in parts]
    return _mass(*sums)


def pair_masses(first: tuple[Array, ...], second: tuple[Array, ...]) -> Array:
    """``InvariantMasses``: of each particle of one collection with its partner in the other."""
    kind = np.result_type(*first, *second)
    one = _momentum(*(np.asarray(each, kind) for each in first))
    two = _momentum(*(np.asarray(each, kind) for each in second))
    x, y, z, e = (a + b for a, b in zip(one, two))
    return _mass(x, y, z, e)


def elementwise(function: Callable[..., Array], *arrays: Array) -> Array:
    """A NumPy function over aligned arrays, with floating-point warnings kept quiet."""
    with np.errstate(all="ignore"):
        return function(*arrays)
