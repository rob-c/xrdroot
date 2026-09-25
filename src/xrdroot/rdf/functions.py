"""The functions a string expression may call: ``<cmath>``, ``TMath`` and ``ROOT::VecOps``.

The mathematical functions are the ones ``TTree::Draw`` expressions call -
:data:`xrdroot.formula.functions.FUNCTIONS` - applied as ``ROOT::VecOps``
applies them to an ``RVec``: element by element, with the arguments lined up
the way the operators line them up. The ``ROOT::VecOps`` functions -
``Sum``, ``Take``, ``DeltaR``, ``InvariantMass`` and the rest - are the
kernels :mod:`.vecops` is built on, called on a whole batch of collections at
once. Every one answers to its bare name, as ROOT's ``using namespace
ROOT::VecOps`` has it, and to ``VecOps::`` and ``ROOT::VecOps::`` in front.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..formula.errors import FormulaError
from ..formula.functions import FUNCTIONS
from . import kernels
from .values import Value, align, per_entry, undefined_where

__all__ = ["Function", "TABLE", "call"]

Array = Any


class Function:
    """What a name calls: how many arguments it takes, and what it does with them."""

    __slots__ = ("least", "most", "apply")

    def __init__(self, least: int, most: int, apply: Callable[..., Value]) -> None:
        self.least = least
        self.most = most
        self.apply = apply


def _mathematical(name: str, function: Any) -> Function:
    def apply(*args: Value) -> Value:
        arrays, offsets = align(args, name)
        with np.errstate(all="ignore"):
            return Value(np.asarray(function.apply(*arrays)), offsets)

    return Function(function.least, function.most, apply)


def _rows(value: Value, what: str) -> tuple[Array, Array]:
    if value.offsets is None or value.items is not None:
        raise UnsupportedFeatureError(
            f"{what} takes a collection per entry - an RVec - and was given a number per "
            f"entry; a single number has nothing to {what} over"
        )
    return np.asarray(value.data), value.offsets


def _integral(value: Value, what: str) -> Array:
    data = np.asarray(value.data)
    if data.dtype.kind not in "biu":
        raise UnsupportedFeatureError(f"{what} takes an integer, and was given {data.dtype}")
    return data


# -- reductions: a collection per entry to a number per entry ------------------------


def _sum(values: Value, zero: Value | None = None) -> Value:
    start = None if zero is None else np.asarray(zero.data)
    if start is not None and start.ndim:
        raise UnsupportedFeatureError("Sum takes a single number to start from, not one per entry")
    return Value(kernels.summed(*_rows(values, "Sum"), start))


def _product(values: Value) -> Value:
    content, offsets = _rows(values, "Product")
    return Value(kernels.reduced(np.multiply, content, offsets, 1)[0])


def _mean(values: Value) -> Value:
    return Value(kernels.means(*_rows(values, "Mean")))


def _var(values: Value) -> Value:
    return Value(kernels.variances(*_rows(values, "Var")))


def _std(values: Value) -> Value:
    return Value(np.sqrt(kernels.variances(*_rows(values, "StdDev"))))


def _extreme(ufunc: Any, what: str) -> Callable[[Value], Value]:
    def apply(values: Value) -> Value:
        out, empty = kernels.reduced(ufunc, *_rows(values, what), 0)
        undefined_where(empty, f"the {what} of an empty collection")
        return Value(out)

    return apply


def _arg(ufunc: Any, what: str) -> Callable[[Value], Value]:
    def apply(values: Value) -> Value:
        return Value(kernels.arg_extreme(*_rows(values, what), ufunc))

    return apply


def _any(values: Value) -> Value:
    content, offsets = _rows(values, "Any")
    return Value(kernels.reduced(np.logical_or, content != 0, offsets, False)[0])


def _all(values: Value) -> Value:
    content, offsets = _rows(values, "All")
    return Value(kernels.reduced(np.logical_and, content != 0, offsets, True)[0])


def _dot(first: Value, second: Value) -> Value:
    (a, b), offsets = align((first, second), "Dot")
    _rows(Value(a, offsets), "Dot")
    return Value(kernels.summed(a * b, offsets))


def _mass(pt: Value, eta: Value, phi: Value, mass: Value) -> Value:
    arrays, offsets = align((pt, eta, phi, mass), "InvariantMass")
    _rows(Value(arrays[0], offsets), "InvariantMass")
    pt_, eta_, phi_, mass_ = arrays
    return Value(kernels.invariant_mass(pt_, eta_, phi_, mass_, offsets))


# -- collections to collections ------------------------------------------------------


def _take(values: Value, index: Value, default: Value | None = None) -> Value:
    content, offsets = _rows(values, "Take")
    if index.offsets is not None:
        picked, out, bad = kernels.taken(content, offsets, _integral(index, "Take"), index.offsets)
    else:
        fill = None if default is None else np.asarray(default.data)
        picked, out, bad = kernels.first_n(content, offsets, _integral(index, "Take"), fill)
    undefined_where(bad, "Take of elements past the end of a collection")
    return Value(picked, out)


def _nonzero(values: Value) -> Value:
    return Value(*kernels.nonzero(*_rows(values, "Nonzero")))


def _where(condition: Value, then: Value, otherwise: Value) -> Value:
    (test, a, b), offsets = align((condition, then, otherwise), "Where")
    return Value(np.where(test != 0, a, b), offsets)


def _argsort(values: Value) -> Value:
    content, offsets = _rows(values, "Argsort")
    return Value(kernels.argsorted(content, offsets), offsets)


def _sort(values: Value) -> Value:
    content, offsets = _rows(values, "Sort")
    order = kernels.argsorted(content, offsets) + offsets[:-1][kernels.rows_of(offsets)]
    return Value(content[order], offsets)


def _reverse(values: Value) -> Value:
    content, offsets = _rows(values, "Reverse")
    return Value(content[kernels.reversed_index(offsets)], offsets)


def _concatenate(first: Value, second: Value) -> Value:
    return Value(*kernels.concatenated(_rows(first, "Concatenate"), _rows(second, "Concatenate")))


def _drop(values: Value, index: Value) -> Value:
    content, offsets = _rows(values, "Drop")
    where, where_offsets = _rows(index, "Drop")
    return Value(*kernels.dropped(content, offsets, _integral(Value(where), "Drop"), where_offsets))


def _enumerate(values: Value) -> Value:
    offsets = _rows(values, "Enumerate")[1]
    return Value(kernels.positions(offsets), offsets)


def _range(first: Value, second: Value | None = None, step: Value | None = None) -> Value:
    numbers = [_integral(value, "Range") for value in (first, second, step) if value is not None]
    begin, end = (0, numbers[0]) if len(numbers) == 1 else (numbers[0], numbers[1])
    stride = numbers[2] if len(numbers) == 3 else 1
    if np.any(np.asarray(stride) == 0):
        raise FormulaError("Range with a stride of zero would never end")
    return Value(*kernels.counting(end, begin, stride))


def _combinations(first: Value, second: Value) -> Value:
    offsets = _rows(first, "Combinations")[1]
    if second.offsets is None:
        size = np.asarray(second.data)
        if size.ndim:
            raise UnsupportedFeatureError("Combinations takes one size for every entry")
        picks, out = kernels.combinations(offsets, int(size))
    else:
        picks, out = kernels.cartesian(offsets, _rows(second, "Combinations")[1])
    members = tuple(Value(pick.astype(np.uint64), out) for pick in picks)
    return Value(np.zeros(0), items=members)


def _delta_phi(first: Value, second: Value, c: Value | None = None) -> Value:
    (a, b), offsets = align((first, second), "DeltaPhi")
    turn = np.pi if c is None else np.asarray(c.data)
    return Value(kernels.delta_phi(a, b, turn), offsets)


def _delta_r2(*args: Value) -> Value:
    (e1, e2, p1, p2), offsets = align(args[:4], "DeltaR2")
    turn = kernels.delta_phi(p1, p2, np.pi if len(args) < 5 else np.asarray(args[4].data))
    return Value((e1 - e2) * (e1 - e2) + turn * turn, offsets)


def _delta_r(*args: Value) -> Value:
    squared = _delta_r2(*args)
    return Value(np.sqrt(squared.data), squared.offsets)


def _masses(*args: Value) -> Value:
    arrays, offsets = align(args, "InvariantMasses")
    return Value(kernels.pair_masses(tuple(arrays[:4]), tuple(arrays[4:])), offsets)


#: ``ROOT::VecOps``, by the names ROOT gives each function.
VECOPS: dict[str, Function] = {
    "Sum": Function(1, 2, _sum),
    "Product": Function(1, 1, _product),
    "Mean": Function(1, 1, _mean),
    "Var": Function(1, 1, _var),
    "StdDev": Function(1, 1, _std),
    "Max": Function(1, 1, _extreme(np.maximum, "Max")),
    "Min": Function(1, 1, _extreme(np.minimum, "Min")),
    "ArgMax": Function(1, 1, _arg(np.maximum, "ArgMax")),
    "ArgMin": Function(1, 1, _arg(np.minimum, "ArgMin")),
    "Any": Function(1, 1, _any),
    "All": Function(1, 1, _all),
    "Dot": Function(2, 2, _dot),
    "Take": Function(2, 3, _take),
    "Nonzero": Function(1, 1, _nonzero),
    "Where": Function(3, 3, _where),
    "Argsort": Function(1, 1, _argsort),
    "StableArgsort": Function(1, 1, _argsort),
    "Sort": Function(1, 1, _sort),
    "Reverse": Function(1, 1, _reverse),
    "Concatenate": Function(2, 2, _concatenate),
    "Drop": Function(2, 2, _drop),
    "Enumerate": Function(1, 1, _enumerate),
    "Range": Function(1, 3, _range),
    "Combinations": Function(2, 2, _combinations),
    "DeltaPhi": Function(2, 3, _delta_phi),
    "DeltaR2": Function(4, 5, _delta_r2),
    "DeltaR": Function(4, 5, _delta_r),
    "InvariantMass": Function(4, 4, _mass),
    "InvariantMasses": Function(8, 8, _masses),
}

#: Every name a function may be called by, against what it is.
TABLE: dict[str, Function] = {
    **{name: _mathematical(name, function) for name, function in FUNCTIONS.items()},
    **VECOPS,
    **{f"VecOps::{name}": function for name, function in VECOPS.items()},
    **{f"ROOT::VecOps::{name}": function for name, function in VECOPS.items()},
}


def call(name: str, args: Sequence[Value], text: str) -> Value:
    """Call the function ``name`` with the values of its arguments."""
    function = TABLE[name]
    if not function.least <= len(args) <= function.most:
        wanted = (
            str(function.least)
            if function.least == function.most
            else f"{function.least} to {function.most}"
        )
        raise FormulaError(
            f"{name} takes {wanted} argument{'' if wanted == '1' else 's'}, and {text!r} "
            f"gives it {len(args)}"
        )
    return function.apply(*args)


def as_size(values: Any) -> Any:
    """A count, as ``size()`` gives it: a ``Long64_t`` here, so ``v.size() - 1`` can be ``-1``."""
    return np.asarray(values, dtype=np.int64)


def cast(value: Value, dtype: Any) -> Value:
    """A C++ conversion: ``bool`` is nonzero; an integer truncates, a NaN refused."""
    (data,), offsets = align((value,), "a cast")
    if np.dtype(dtype) == np.bool_:
        return Value(data != 0, offsets)
    if np.dtype(dtype).kind in "iu" and data.dtype.kind == "f":
        undefined_where(per_entry(~np.isfinite(data), offsets), "a NaN cast to an integer")
        with np.errstate(all="ignore"):
            return Value(np.trunc(data).astype(np.int64).astype(dtype), offsets)
    return Value(data.astype(dtype), offsets)
