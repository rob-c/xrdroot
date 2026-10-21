"""``ROOT.RVec`` and ``ROOT::VecOps``: one entry's collection, and what is done to it.

    >>> v = RVec['double']([1.0, 2.0, 3.0])
    >>> (v * 2)[v > 1]
    <ROOT::VecOps::RVec<double> [4.0, 6.0]>
    >>> VecOps.Sum(v), VecOps.Max(v)
    (6.0, 3.0)

An ``RVec`` is a ``std::vector`` - ``push_back``, ``size``, ``[]`` - that does
arithmetic element by element and is indexed by a mask, as ROOT's is. Here it
is kept in NumPy, so anything NumPy does to an array it does to one, and
what comes back is an ``RVec`` again. ``RVecF``, ``RVecD``, ``RVecI``,
``RVecL`` and ``RVecB`` are ROOT's short names for the usual types.

``VecOps`` has ROOT's functions of one collection. They are
:mod:`xrdroot.rdf.vecops`'s - the functions a ``Define`` string calls, over
whole batches of entries - given a batch of one, so the numbers are the
same whichever way they were reached.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...rdf import kernels
from ...rdf import vecops as _batched
from ...tree import Jagged
from ..stl import _NUMERIC_ASSIGN_READERS, _Template, _Vector, _vector_class, cpp_name
from ..vectorassign import register as _register_vector_assignment
from ..vectorvalue import register as _register_vector_value

__all__ = ["RVec", "RVecF", "RVecD", "RVecI", "RVecL", "RVecB", "VecOps"]


class _RVec(np.lib.mixins.NDArrayOperatorsMixin, _Vector):  # type: ignore[misc]
    """An ``RVec`` of one element type: a vector that does NumPy's arithmetic."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {self.data().tolist()!r}>"

    def __str__(self) -> str:
        """As ROOT's ``operator<<`` writes an ``RVec``: ``{ 1, 2, 0.666667 }``, each element as
        a C++ stream writes it - six digits - and a ``bool`` as ``1`` or ``0``."""
        kind = self.dtype.kind
        items = [f"{value:g}" if kind == "f" else str(int(value)) for value in self.data()]
        return "{ " + ", ".join(items) + " }"

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        given = [each.data() if isinstance(each, _Vector) else each for each in inputs]
        found = getattr(ufunc, method)(*given, **kwargs)
        return _made(found)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, (list, np.ndarray, _Vector)):
            if isinstance(index, _RVec):
                if len(index) != len(self):
                    raise TypeError(
                        f"Cannot index RVecN of size {len(self)} with condition vector of "
                        f"different size ({len(index)}).")
                index = np.asarray(index, dtype=bool)
            else:
                index = np.asarray(index)
            return _made(self.data()[index])
        return _Vector.__getitem__(self, index)

    def __eq__(self, other: object) -> Any:
        return _made(np.equal(self.data(), np.asarray(other)))

    def __ne__(self, other: object) -> Any:
        return _made(np.not_equal(self.data(), np.asarray(other)))

    __hash__ = None


def _rvec_class(kind: Any) -> type:
    """``RVec<kind>``: of numbers, a vector that does NumPy's arithmetic; of collections or
    strings - ``RVec<RVec<size_t>>`` - a vector of them, as ``std::vector`` holds them."""
    if not isinstance(kind, np.dtype):
        return _vector_class(kind)
    name = f"ROOT::VecOps::RVec<{cpp_name(kind)}>"
    members = {"dtype": kind, "value_type": cpp_name(kind), "__cpp_name__": name}
    made = _register_vector_value(type(name, (_RVec,), {"__slots__": (), **members}),
                                  kind, True)
    return _register_vector_assignment(made, "rvec", _NUMERIC_ASSIGN_READERS)


#: ``ROOT.RVec``, and ``ROOT.VecOps.RVec``: ``RVec['float']`` is ``RVec<float>``.
RVec = _Template("RVec", 1, _rvec_class)
RVecF = RVec["float"]
RVecD = RVec["double"]
RVecI = RVec["int"]
RVecL = RVec["long"]
RVecB = RVec["bool"]


def _made(values: Any) -> Any:
    """What a NumPy result is given back as: an ``RVec`` for an array, else as it is."""
    if isinstance(values, np.ndarray) and values.ndim == 1:
        return RVec[values.dtype](values)
    if isinstance(values, np.generic):
        return values.item()
    return values


def _batch(value: Any) -> Any:
    """One entry's argument as the batched functions take it: a batch of one collection."""
    if isinstance(value, (_Vector, list, tuple, np.ndarray)):
        content = np.asarray(value.data() if isinstance(value, _Vector) else value)
        return Jagged(content, [0, len(content)])
    return value


def _unbatched(value: Any) -> Any:
    if isinstance(value, Jagged):
        return _made(np.asarray(value[0]))
    if isinstance(value, tuple):
        return tuple(_unbatched(each) for each in value)
    if isinstance(value, np.ndarray):
        return _made(value.reshape(-1)[0]) if value.size else _made(value)
    return _made(value)


class _OneEntry:
    """A batched function of :mod:`xrdroot.rdf.vecops` as a function of one entry's values -
    and, by ``[...]``, the same function, as C++'s ``InvariantMass<float>`` names it."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self._function = function
        #: The function of whole batches, which a frame is given in place of this one.
        self.batched = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__

    def __call__(self, *values: Any) -> Any:
        return _unbatched(self._function(*(_batch(value) for value in values)))

    def __getitem__(self, kinds: Any) -> _OneEntry:
        return self


#: These scalars convert without a user-defined integer coercion.
_CHOICE_INTS = (int, bool, np.int8, np.int16, np.int32, np.int64, np.longlong,
                np.uint8, np.uint16, np.uint32, np.uint64, np.ulonglong)

#: Replacing the batched callback must keep using the general evaluator.
_COMBINATIONS_FUNCTION = _batched.Combinations


def _collection(value: Any) -> bool:
    kind = type(value)
    if type(kind) is not type:
        return False
    return kind in (list, tuple, np.ndarray) or _Vector in kind.__mro__


class _CombinationsEntry(_OneEntry):
    """A single collection can use the same integer table without a batch of one."""

    def __call__(self, *values: Any) -> Any:
        if len(values) == 2 and self._function is _COMBINATIONS_FUNCTION:
            first, second = values
            if type(type(second)) is type and type(second) in _CHOICE_INTS and _collection(first):
                content = np.asarray(first.data() if isinstance(first, _Vector) else first)
                count, width = len(content), int(second)
                table = kernels._table(count, width)
                return tuple(_made(table[:, i]) for i in range(width))
        return super().__call__(*values)


def _one(function: Callable[..., Any]) -> _OneEntry:
    """A batched function as a function of one entry's values."""
    if function is _COMBINATIONS_FUNCTION:
        return _CombinationsEntry(function)
    return _OneEntry(function)


def _map(*arguments: Any) -> Any:
    """``Map(v, f)`` - and ``Map(v1, v2, f)`` - ``f`` of each element, in an ``RVec``."""
    *values, function = arguments
    return _made(np.asarray([function(*each) for each in zip(*values, strict=False)]))


def _filter(values: Any, function: Callable[[Any], Any]) -> Any:
    """``Filter(v, f)``: the elements ``f`` is true for."""
    return _made(np.asarray([each for each in values if function(each)], dtype=values.dtype))


class _VecOps:
    """``ROOT.VecOps``: ROOT's functions of one collection, and ``RVec`` itself."""

    RVec = RVec
    Map = staticmethod(_map)
    Filter = staticmethod(_filter)

    def __getattr__(self, name: str) -> Any:
        found = getattr(_batched, name, None)
        if found is None or name.startswith("_") or name not in _batched.__all__:
            raise AttributeError(f"ROOT has VecOps.{name}; xrdroot.pyroot does not yet")
        return _one(found)

    def __repr__(self) -> str:
        return "<namespace ROOT::VecOps>"


#: ``ROOT.VecOps``.
VecOps = _VecOps()
