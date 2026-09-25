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
    "preinc",
    "postinc",
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
    for row, values in zip(out, init):
        _fill(row, values)


def _object_array(dims: tuple[int, ...], init: Any, make: Callable[[], Any] | None) -> list[Any]:
    out = _filled(dims, make or (lambda: None))
    if init is not None:
        for index, value in enumerate(list(init)[: dims[0]]):
            out[index] = value
    return out


class Overloaded:
    """A C++ overload set: calls go to the candidate the arguments fit, arity first.

    Each candidate is ``(function, fewest, most, kinds)``: how many arguments
    it takes with and without its defaults, and what Python types each
    argument must be (``None`` for anything). As a class attribute it binds
    like a method, so overloaded members and constructors work too.
    """

    def __init__(self, name: str, *candidates: tuple[Callable[..., Any], int, int, tuple[Any, ...]]):
        self.name = name
        self.candidates = candidates

    def __get__(self, obj: Any, owner: Any = None) -> Any:
        if obj is None:
            return self
        return lambda *args: self(obj, *args)

    def __call__(self, *args: Any) -> Any:
        fitting = [c for c in self.candidates if c[1] <= len(args) <= c[2]]
        for function, _, _, kinds in fitting:
            if all(_fits(value, kind) for value, kind in zip(args, kinds)):
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


def value_copy(value: Any) -> Any:
    """``T b = a;`` for an object: a copy of it, not the same object under a second name."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.copy()
    try:
        return copy.deepcopy(value)
    except Exception:  # noqa: BLE001 - an object that will not deep-copy is copied shallowly
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
    return container  # type: ignore[no-any-return]


def sort_range(container: Any, start: int, stop: Any, less: Callable[..., Any] | None = None) -> None:
    """``std::sort(begin, end[, less])`` over ``container[start:stop]``, in place."""
    end = len(container) if stop is None else int(stop)
    values = list(container[start:end])
    if less is None:
        values.sort()
    else:
        import functools

        values.sort(key=functools.cmp_to_key(lambda a, b: -1 if less(a, b) else int(bool(less(b, a)))))
    container[start:end] = values


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


def deref(pointer: Any) -> Any:
    """``*p`` when the type of ``p`` was not known: a cell's value, an array's first element."""
    if isinstance(pointer, (np.ndarray, list)):
        return pointer[0]
    if hasattr(pointer, "value") and type(pointer).__name__ in ("Cell", "ItemRef", "AttrRef"):
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
