"""Objects, arrays and overloads: the C++ a translation leans on that is not arithmetic.

Arrays of numbers are NumPy arrays of the declared width (``double`` is
``float64``, ``float`` is ``float32``, ``int`` is ``int32``), so ``&a[3]`` is
the view ``a[3:]`` and a ROOT function writing into it writes into ``a``.
Arrays of anything else are lists. :class:`Overloaded` is C++ overloading:
one Python name that picks, by the number and then the types of its
arguments, which of several translated functions to call.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

__all__ = [
    "array",
    "Overloaded",
    "delete",
    "construct_at",
    "value_copy",
    "dynamic_cast",
    "set_item",
    "set_attr",
    "Pair",
    "iterate",
    "sort_range",
    "reverse_range",
    "CppException",
    "throw",
    "deref",
    "pointee",
    "preinc",
    "postinc",
    "INTEGRAL",
    "REAL",
]

#: The NumPy type an array of each C++ arithmetic type is made of.
DTYPES = {
    "bool": np.bool_,
    "char": np.int8,
    "signed char": np.int8,
    "unsigned char": np.uint8,
    "short": np.int16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "unsigned int": np.uint32,
    "long": np.int64,
    "unsigned long": np.uint64,
    "long long": np.int64,
    "unsigned long long": np.uint64,
    "float": np.float32,
    "double": np.float64,
    "long double": np.float64,
}


def _filled(shape: tuple[int, ...], make: Callable[[], Any]) -> list[Any]:
    if len(shape) == 1:
        return [make() for _ in range(shape[0])]
    return [_filled(shape[1:], make) for _ in range(shape[0])]


def _flat(items: Any) -> list[Any]:
    if isinstance(items, (list, tuple)):
        return [value for item in items for value in _flat(item)]
    return [items]


def array(ctype: str, shape: Any, init: Any = None, make: Callable[[], Any] | None = None) -> Any:
    """A C array of ``ctype`` - zeros, or ``init`` padded with zeros as C pads it.

    ``shape`` is a length or a tuple of them; ``make`` builds each element of
    an array of objects, which is a (nested) list.
    """
    dims = tuple(int(n) for n in shape) if isinstance(shape, (tuple, list)) else (int(shape),)
    dtype = DTYPES.get(ctype)
    if dtype is None:
        return _object_array(dims, init, make)
    out = np.zeros(dims, dtype=dtype)
    if init is not None:
        _fill(out, init)
    return out


def _fill(out: Any, init: Any) -> None:
    if out.ndim == 1:
        values = _flat(init)[: out.shape[0]]
        out[: len(values)] = values
        return
    for row, values in zip(out, init, strict=False):
        _fill(row, values)


def _object_array(dims: tuple[int, ...], init: Any, make: Callable[[], Any] | None) -> list[Any]:
    out = _filled(dims, make or (lambda: None))
    if init is not None:
        for index, value in enumerate(list(init)[: dims[0]]):
            out[index] = value
    return out


#: What an argument for an integer parameter may be: a Python or a NumPy integer.
INTEGRAL = (int, np.integer)

#: What an argument for a floating parameter may be: any real number.
REAL = (int, float, np.integer, np.floating)


#: One function of an overload set: it, its fewest and most arguments, and their kinds.
Candidate = tuple[Callable[..., Any], int, int, tuple[Any, ...]]


class Overloaded:
    """A C++ overload set: calls go to the candidate the arguments fit, arity first.

    Each candidate is ``(function, fewest, most, kinds)``: how many arguments
    it takes with and without its defaults, and what Python types each
    argument must be (``None`` for anything). As a class attribute it binds
    like a method, so overloaded members and constructors work too.
    """

    def __init__(self, name: str, *candidates: Candidate) -> None:
        self.name = name
        self.candidates = candidates

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if obj is None:
            return self
        return lambda *args: self(obj, *args)

    def __call__(self, *args: Any) -> Any:
        fitting = [c for c in self.candidates if c[1] <= len(args) <= c[2]]
        for function, _, _, kinds in fitting:
            if all(_fits(value, kind) for value, kind in zip(args, kinds, strict=False)):
                return function(*args)
        if fitting:
            return fitting[0][0](*args)
        raise TypeError(f"no overload of {self.name} takes {len(args)} arguments")


def _fits(value: Any, kind: Any) -> bool:
    return kind is None or isinstance(value, kind)


def delete(obj: Any) -> None:
    """``delete p``: a user class's destructor runs; a ROOT file is closed, as its would be."""
    destructor = getattr(obj, "_destruct", None)
    if callable(destructor):
        destructor()
        return
    close = getattr(obj, "Close", None)
    if callable(close) and hasattr(obj, "IsOpen"):
        close()


def construct_at(container: Any, index: Any, obj: Any) -> Any:
    """``new (clones[i]) T(args)``: ``obj`` put in slot ``index`` - ROOT's ``AddAt``, or ``[i] =``.

    A ``TClonesArray`` hands placement new the memory of its slot ``i``;
    here the object is built first and the array given it, which is the
    same array of the same objects afterwards.
    """
    add = getattr(container, "AddAt", None)
    if callable(add):
        add(obj, int(index))
    else:
        container[index] = obj
    return obj


def value_copy(value: Any) -> Any:
    """``T b = a;`` for an object: a copy of it, not the same object under a second name."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:
        return copy.copy(value)


def dynamic_cast(kind: Any, obj: Any) -> Any:
    """``dynamic_cast<T*>(p)``: ``p`` if it is a ``T``, ``None`` if it is not."""
    if isinstance(kind, type) and not isinstance(obj, kind):
        return None
    return obj


def set_item(container: Any, key: Any, value: Any) -> Any:
    """``a[i] = v`` where C++ uses its value: stores it and gives it back."""
    container[key] = value
    return value


def set_attr(obj: Any, name: str, value: Any) -> Any:
    """``o.x = v`` where C++ uses its value."""
    setattr(obj, name, value)
    return value


class Pair:
    """``std::pair``: ``first`` and ``second``, and unpacking as two."""

    __slots__ = ("first", "second")

    def __init__(self, first: Any = None, second: Any = None) -> None:
        self.first = first
        self.second = second

    def __iter__(self) -> Iterator[Any]:
        return iter((self.first, self.second))

    def __repr__(self) -> str:
        return f"Pair({self.first!r}, {self.second!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Pair) and tuple(self) == tuple(other)

    def __lt__(self, other: Pair) -> bool:
        return tuple(self) < tuple(other)

    __hash__ = None  # type: ignore[assignment]


def iterate(container: Any) -> Iterable[Any]:
    """What a range-for over ``container`` visits: a map's entries as pairs, anything's items."""
    items = getattr(container, "items", None)
    if callable(items) and not isinstance(container, np.ndarray):
        return [Pair(key, value) for key, value in items()]
    python = hasattr(type(container), "__iter__")  # a library container: Python iterates it
    if python or hasattr(container, "_deref") or not hasattr(type(container), "begin"):
        return container  # type: ignore[no-any-return]
    return _walked(container)


def _walked(container: Any) -> Iterator[Any]:
    """A range-for over a class of the macro's own with ``begin()`` and ``end()``.

    Its iterators are the macro's too: ``*it`` is their ``_deref``, ``++it``
    their ``_preinc``, and ``it != end`` their ``__ne__``, as C++ calls them.
    """
    it, end = container.begin(), container.end()
    while it != end:
        yield it._deref()
        it._preinc()


def sort_range(
    container: Any, start: int, stop: Any, less: Callable[..., Any] | None = None
) -> None:
    """``std::sort(begin, end[, less])`` over ``container[start:stop]``, in place."""
    end = len(container) if stop is None else int(stop)
    values = list(container[start:end])
    if less is None:
        values.sort()
    else:
        import functools

        values.sort(key=functools.cmp_to_key(_comparison(less)))
    container[start:end] = values


def _comparison(less: Callable[..., Any]) -> Callable[[Any, Any], int]:
    """A C++ "less than" predicate as the three-way comparison Python's sort takes."""

    def compare(a: Any, b: Any) -> int:
        if less(a, b):
            return -1
        return 1 if less(b, a) else 0

    return compare


def reverse_range(container: Any, start: int, stop: Any) -> None:
    """``std::reverse(begin, end)``, in place."""
    end = len(container) if stop is None else int(stop)
    container[start:end] = list(container[start:end])[::-1]


class CppException(Exception):
    """A C++ ``throw`` of anything: the value thrown, and ``what()`` as ``std::exception`` has."""

    def __init__(self, value: Any = None) -> None:
        super().__init__(value)
        self.value = value

    def what(self) -> str:
        what = getattr(self.value, "what", None)
        return str(what() if callable(what) else self.value)


def throw(value: Any = None) -> Any:
    """``throw value`` where C++ allows it as an expression, in a ``?:``."""
    if isinstance(value, BaseException):
        raise value
    raise CppException(value)


def pointee(pointer: Any, name: str) -> Any:
    """``p.release()`` or ``p.get()`` when the type of ``p`` was not known: the object's own
    method if it has one, else the object - which is what a smart pointer holds here."""
    method = getattr(pointer, name, None)
    return method() if callable(method) else pointer


def deref(pointer: Any) -> Any:
    """``*p`` when the type of ``p`` was not known: a cell's value, an array's first element.

    A library object that holds a value as a cell does - an RNTuple field's
    ``shared_ptr<int>`` - says so with a true ``_cint_cell``.
    """
    if isinstance(pointer, (np.ndarray, list)):
        return pointer[0]
    named = type(pointer).__name__ in ("Cell", "ItemRef", "AttrRef")
    if hasattr(pointer, "value") and (named or getattr(pointer, "_cint_cell", False)):
        return pointer.value
    return pointer


def preinc(ref: Any, delta: Any) -> Any:
    """``++x`` of something held by reference: add, store, give back the new value."""
    ref.value = ref.value + delta
    return ref.value


def postinc(ref: Any, delta: Any) -> Any:
    """``x++`` of something held by reference: add and store, give back the old value."""
    old = ref.value
    ref.value = old + delta
    return old
