"""``ROOT::VecOps`` for Python: the functions of an ``RVec``, over a whole batch at once.

    >>> from xrdroot.rdf import vecops
    >>> pt = Jagged([40.0, 12.5, 33.0], [0, 2, 2, 3])       # doctest: +SKIP
    >>> vecops.Sum(pt)                                        # doctest: +SKIP
    array([52.5,  0. , 33. ])
    >>> vecops.Take(pt, vecops.Argsort(pt))                   # doctest: +SKIP

In ROOT each of these is called once per entry, on that entry's ``RVec``.
Here each is called once per batch, on a :class:`~xrdroot.Jagged` holding
every entry's collection, and gives back one value per entry - a NumPy array
- or a collection per entry - another ``Jagged``. That is what a callable
given to ``Define`` or ``Filter`` sees and gives back, so these are the
tools to write one with. The same functions, by the same names, are what a
string expression calls, and the numbers are the same either way.

A collection may be given as a ``Jagged``, or as a NumPy array of two
dimensions, whose rows are each one entry's collection of a fixed size.
Where a function takes a number per entry too - ``Take(v, n)``,
``DeltaPhi(v, phi0)`` - that is a NumPy array of one per entry, or a single
number for them all.

Where C++ would be undefined - the ``Max`` of an empty ``RVec``, ``Take`` of
more elements than there are - these do not make something up:
``Max``/``Min`` put ``default`` in such an entry (``NaN`` unless told), and
``Take`` refuses with the entry it ran out in unless given a default to pad
with.
"""

from __future__ import annotations

import typing
from collections.abc import Callable

import numpy as np

from ..tree import Jagged
from . import kernels

__all__ = [
    "Sum",
    "Product",
    "Mean",
    "Var",
    "StdDev",
    "Max",
    "Min",
    "ArgMax",
    "ArgMin",
    "Any",
    "All",
    "Dot",
    "Take",
    "Nonzero",
    "Where",
    "Argsort",
    "StableArgsort",
    "Sort",
    "Reverse",
    "Concatenate",
    "Drop",
    "Enumerate",
    "Range",
    "Combinations",
    "DeltaPhi",
    "DeltaR2",
    "DeltaR",
    "InvariantMass",
    "InvariantMasses",
    "Map",
    "Size",
]

Array = typing.Any
#: Anything a function here is given: a collection, a value per entry, a number.
Given = typing.Any


def _rows(values: Given, what: str) -> tuple[Array, Array]:
    """A collection per entry as flat values and compact offsets, or a refusal."""
    if isinstance(values, Jagged):
        return kernels.compact(values)
    given = np.asarray(values)
    if given.ndim >= 2:
        width = int(np.prod(given.shape[1:]))
        return given.reshape(-1), np.arange(len(given) + 1, dtype=np.int64) * width
    raise TypeError(
        f"{what} takes a collection per entry - a Jagged, or a 2-dimensional array of "
        f"one row per entry - and was given {type(values).__name__} of "
        f"{given.ndim} dimension{'' if given.ndim == 1 else 's'}"
    )


def _jagged(content: Array, offsets: Array) -> Jagged:
    return Jagged(content, offsets)


def _spread(value: Given, offsets: Array) -> Array:
    """A value per entry, or one for all, repeated to every element of its entry."""
    given = np.asarray(value)
    if given.ndim == 0:
        return given
    return np.repeat(given, np.diff(offsets))


def _same_sizes(offsets: list[Array], what: str) -> Array:
    first = offsets[0]
    for other in offsets[1:]:
        if len(other) != len(first) or not np.array_equal(other, first):
            raise ValueError(
                f"{what} is given collections that are not the same size in every entry, "
                f"and pairs their elements one for one"
            )
    return first


def _aligned(values: tuple[Given, ...], what: str) -> tuple[list[Array], Array | None]:
    """Collections and values per entry lined up element for element, as C++ does."""
    kinds = [_collection(value) for value in values]
    if not any(kinds):
        return [np.asarray(value) for value in values], None
    rows = {at: _rows(value, what) for at, value in enumerate(values) if kinds[at]}
    offsets = _same_sizes([found[1] for found in rows.values()], what)
    return [_lined_up(value, rows.get(at), offsets) for at, value in enumerate(values)], offsets


def _collection(value: Given) -> bool:
    return isinstance(value, Jagged) or np.ndim(value) >= 2


def _lined_up(value: Given, rows: tuple[Array, Array] | None, offsets: Array) -> Array:
    return _spread(value, offsets) if rows is None else rows[0]


def _wrapped(values: Array, offsets: Array | None) -> Given:
    return values if offsets is None else Jagged(values, offsets)


def Map(function: Callable[..., Array], *values: Given) -> Given:
    """Any NumPy function of elements, applied to collections element by element.

        >>> vecops.Map(np.sqrt, pt)                           # doctest: +SKIP

    Collections are paired element for element and must be the same size in
    every entry; a value per entry goes with each of its entry's elements.
    """
    arrays, offsets = _aligned(values, "Map")
    return _wrapped(kernels.elementwise(function, *arrays), offsets)


def Size(values: Given) -> Array:
    """How many elements each entry's collection holds: ``v.size()``."""
    return np.diff(_rows(values, "Size")[1])


def Sum(values: Given, zero: Given = None) -> Array:
    """``Sum``: each collection added up, in its elements' type or in ``zero``'s."""
    return kernels.summed(*_rows(values, "Sum"), zero)


def Product(values: Given) -> Array:
    """``Product``: each collection multiplied together; one for an empty one."""
    content, offsets = _rows(values, "Product")
    return kernels.reduced(np.multiply, content, offsets, 1)[0]


def Mean(values: Given) -> Array:
    """``Mean``: each collection's average, in ``double``; zero for an empty one."""
    return kernels.means(*_rows(values, "Mean"))


def Var(values: Given) -> Array:
    """``Var``: each collection's variance over ``n - 1``; zero for fewer than two."""
    return kernels.variances(*_rows(values, "Var"))


def StdDev(values: Given) -> Array:
    """``StdDev``: the square root of :func:`Var`."""
    return np.sqrt(Var(values))


def _extreme(values: Given, ufunc: Given, default: Given, what: str) -> Array:
    content, offsets = _rows(values, what)
    out, empty = kernels.reduced(ufunc, content, offsets, 0)
    if not empty.any():
        return out
    kind = np.result_type(out.dtype, np.asarray(default).dtype)
    return np.where(empty, np.asarray(default, kind), out.astype(kind))


def Max(values: Given, default: Given = np.nan) -> Array:
    """``Max``: each collection's largest element, ``default`` for an empty one."""
    return _extreme(values, np.maximum, default, "Max")


def Min(values: Given, default: Given = np.nan) -> Array:
    """``Min``: each collection's smallest element, ``default`` for an empty one."""
    return _extreme(values, np.minimum, default, "Min")


def ArgMax(values: Given) -> Array:
    """``ArgMax``: where each collection's first largest element is; ``0`` if empty."""
    return kernels.arg_extreme(*_rows(values, "ArgMax"), np.maximum)


def ArgMin(values: Given) -> Array:
    """``ArgMin``: where each collection's first smallest element is; ``0`` if empty."""
    return kernels.arg_extreme(*_rows(values, "ArgMin"), np.minimum)


def Any(values: Given) -> Array:
    """``Any``: whether any element of each collection is nonzero."""
    content, offsets = _rows(values, "Any")
    return kernels.reduced(np.logical_or, content != 0, offsets, False)[0]


def All(values: Given) -> Array:
    """``All``: whether every element of each collection is nonzero; true when empty."""
    content, offsets = _rows(values, "All")
    return kernels.reduced(np.logical_and, content != 0, offsets, True)[0]


def Dot(first: Given, second: Given) -> Array:
    """``Dot``: the sum of the products of two collections' elements, pair by pair."""
    (a, b), offsets = _aligned((first, second), "Dot")
    if offsets is None:
        raise TypeError("Dot takes two collections per entry, and was given numbers")
    return kernels.summed(a * b, offsets)


def _where_bad(bad: Array, what: str) -> None:
    if bad.any():
        raise IndexError(
            f"{what} asks for elements past the end of the collection in the entry at "
            f"row {int(np.flatnonzero(bad)[0])} of the batch; give default= to pad a "
            f"collection that is too short"
        )


def Take(values: Given, index: Given, default: Given = None) -> Jagged:
    """``Take``: the elements at the positions given, or the first ``n``, or the last ``-n``.

    ``index`` is a collection of positions per entry, such as
    :func:`Argsort` gives, or a number of elements - one per entry, or one
    for all. Positions past the end are refused; a count past the end is
    refused too unless ``default`` is given to pad with, as ROOT's
    three-argument ``Take`` pads.
    """
    content, offsets = _rows(values, "Take")
    if isinstance(index, Jagged) or np.ndim(index) >= 2:
        where, where_offsets = _rows(index, "Take")
        picked, out, bad = kernels.taken(content, offsets, where, where_offsets)
    else:
        picked, out, bad = kernels.first_n(content, offsets, index, default)
    _where_bad(bad, "Take")
    return Jagged(picked, out)


def Nonzero(values: Given) -> Jagged:
    """``Nonzero``: the positions of each collection's elements that are not zero."""
    return _jagged(*kernels.nonzero(*_rows(values, "Nonzero")))


def Where(condition: Given, then: Given, otherwise: Given) -> Given:
    """``Where``: element by element, ``then`` where the condition holds, else ``otherwise``."""
    (test, a, b), offsets = _aligned((condition, then, otherwise), "Where")
    return _wrapped(np.where(test != 0, a, b), offsets)


def Argsort(values: Given) -> Jagged:
    """``Argsort``: the positions that put each collection in increasing order."""
    content, offsets = _rows(values, "Argsort")
    return Jagged(kernels.argsorted(content, offsets), offsets)


#: ``StableArgsort`` is ``Argsort``: this one keeps equal elements in order already.
StableArgsort = Argsort


def Sort(values: Given) -> Jagged:
    """``Sort``: each collection in increasing order."""
    content, offsets = _rows(values, "Sort")
    order = kernels.argsorted(content, offsets) + offsets[:-1][kernels.rows_of(offsets)]
    return Jagged(content[order], offsets)


def Reverse(values: Given) -> Jagged:
    """``Reverse``: each collection back to front."""
    content, offsets = _rows(values, "Reverse")
    return Jagged(content[kernels.reversed_index(offsets)], offsets)


def Concatenate(first: Given, second: Given) -> Jagged:
    """``Concatenate``: each entry's first collection, then its second."""
    return _jagged(*kernels.concatenated(_rows(first, "Concatenate"), _rows(second, "Concatenate")))


def Drop(values: Given, index: Given) -> Jagged:
    """``Drop``: each collection without the elements at the positions given."""
    content, offsets = _rows(values, "Drop")
    return _jagged(*kernels.dropped(content, offsets, *_rows(index, "Drop")))


def Enumerate(values: Given) -> Jagged:
    """``Enumerate``: the positions ``0, 1, ...`` of each collection's elements."""
    offsets = _rows(values, "Enumerate")[1]
    return Jagged(kernels.positions(offsets), offsets)


def Range(first: Given, second: Given = None, step: Given = 1) -> Jagged:
    """``Range(n)``, ``Range(begin, end)``, ``Range(begin, end, stride)``, per entry."""
    begin, end = (0, first) if second is None else (first, second)
    return _jagged(*kernels.counting(end, begin, step))


def Combinations(first: Given, second: Given) -> tuple[Jagged, ...]:
    """``Combinations``: positions of each collection taken ``k`` at a time, or paired.

        >>> i, j = vecops.Combinations(pt, 2)                # doctest: +SKIP
        >>> i, j = vecops.Combinations(muons, electrons)     # doctest: +SKIP

    With a number, every choice of that many different positions, in
    increasing order; with two collections, every position of the first
    paired with every position of the second. Either way one ``Jagged`` of
    positions per member of the combination, as ROOT gives an ``RVec`` of
    ``RVec``\\ s.
    """
    offsets = _rows(first, "Combinations")[1]
    if isinstance(second, (int, np.integer)):
        picks, out = kernels.combinations(offsets, int(second))
    else:
        picks, out = kernels.cartesian(offsets, _rows(second, "Combinations")[1])
    return tuple(Jagged(pick, out) for pick in picks)


def DeltaPhi(first: Given, second: Given, c: float = np.pi) -> Given:
    """``DeltaPhi``: ``second - first``, folded into ``[-c, c]``, element by element."""
    (a, b), offsets = _aligned((first, second), "DeltaPhi")
    return _wrapped(kernels.delta_phi(a, b, c), offsets)


def DeltaR2(eta1: Given, eta2: Given, phi1: Given, phi2: Given, c: float = np.pi) -> Given:
    """``DeltaR2``: ``(eta1 - eta2)**2 + DeltaPhi(phi1, phi2)**2``, element by element."""
    (e1, e2, p1, p2), offsets = _aligned((eta1, eta2, phi1, phi2), "DeltaR2")
    turn = kernels.delta_phi(p1, p2, c)
    return _wrapped((e1 - e2) * (e1 - e2) + turn * turn, offsets)


def DeltaR(eta1: Given, eta2: Given, phi1: Given, phi2: Given, c: float = np.pi) -> Given:
    """``DeltaR``: the square root of :func:`DeltaR2`."""
    squared = DeltaR2(eta1, eta2, phi1, phi2, c)
    if isinstance(squared, Jagged):
        return Jagged(np.sqrt(squared.content), squared.offsets)
    return np.sqrt(squared)


def InvariantMass(pt: Given, eta: Given, phi: Given, mass: Given) -> Array:
    """``InvariantMass``: the mass of everything in each entry's collections, added up."""
    arrays, offsets = _aligned((pt, eta, phi, mass), "InvariantMass")
    if offsets is None:
        raise TypeError("InvariantMass takes a collection of particles per entry")
    momenta, etas, phis, masses = arrays
    return kernels.invariant_mass(momenta, etas, phis, masses, offsets)


def InvariantMasses(
    pt1: Given,
    eta1: Given,
    phi1: Given,
    mass1: Given,
    pt2: Given,
    eta2: Given,
    phi2: Given,
    mass2: Given,
) -> Given:
    """``InvariantMasses``: the mass of each particle of one collection with its partner."""
    arrays, offsets = _aligned((pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2), "InvariantMasses")
    return _wrapped(kernels.pair_masses(tuple(arrays[:4]), tuple(arrays[4:])), offsets)
