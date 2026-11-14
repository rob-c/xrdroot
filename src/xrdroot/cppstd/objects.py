"""Objects, arrays and overloads: the C++ a translation leans on that is not arithmetic.

Arrays of numbers are NumPy arrays of the declared width (``double`` is
``float64``, ``float`` is ``float32``, ``int`` is ``int32``), so ``&a[3]`` is
the view ``a[3:]`` and a ROOT function writing into it writes into ``a``.
Arrays of anything else are lists. :class:`Overloaded` is C++ overloading:
one Python name that picks, by the number and then the types of its
arguments, which of several translated functions to call.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np

from .buffers import CharBuffer
from .valuecopy import value_copy as value_copy

__all__ = [
    "array",
    "convertible",
    "function_template",
    "UNGIVEN",
    "Overloaded",
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
    "pointee",
    "preinc",
    "postinc",
    "INTEGRAL",
    "REAL",
]

class _Ungiven:
    """The default of a parameter whose C++ default is worked out at each call."""

    def __repr__(self) -> str:
        return "UNGIVEN"


#: What a parameter holds when the call gave it nothing and its default is yet to be made.
UNGIVEN = _Ungiven()

#: The character types, whose arrays are :class:`~.buffers.CharBuffer` s.
CHARACTERS = frozenset({"char", "signed char", "unsigned char"})

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


def array(ctype: str, shape: Any, init: Any = None, make: Callable[..., Any] | None = None) -> Any:
    """A C array of ``ctype`` - zeros, or ``init`` padded with zeros as C pads it.

    ``shape`` is a length or a tuple of them; ``make`` builds each element of
    an array of objects, which is a (nested) list.
    """
    dims = tuple(int(n) for n in shape) if isinstance(shape, (tuple, list)) else (int(shape),)
    dtype = DTYPES.get(ctype)
    if dtype is None:
        return _object_array(dims, init, make)
    out = np.zeros(dims, dtype=dtype)
    if ctype in CHARACTERS:  # memory a char* writes C strings into: :mod:`.buffers`
        out = out.view(CharBuffer)
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


def _object_array(dims: tuple[int, ...], init: Any, make: Callable[..., Any] | None) -> list[Any]:
    out = _filled(dims, make or (lambda: None))
    for index, value in enumerate(list(init or ())[: dims[0]]):
        # ``TString names[2] = {"a", "b"}``: each literal made into the element's class.
        literal = isinstance(value, (str, int, float))
        out[index] = make(value) if make is not None and literal else value
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


def preinc(ref: Any, delta: Any) -> Any:
    """``++x`` of something held by reference: add, store, give back the new value."""
    ref.value = ref.value + delta
    return ref.value


def postinc(ref: Any, delta: Any) -> Any:
    """``x++`` of something held by reference: add and store, give back the old value."""
    old = ref.value
    ref.value = old + delta
    return old


def _operator(name: str) -> Callable[..., Any]:
    """:mod:`operator`'s function for ``name``: ``add``, or ``and_`` for ``and``."""
    found: Callable[..., Any] = getattr(operator, name, None) or getattr(operator, name + "_")
    return found


#: The operators a class converting to a number takes as that number does, by method: each
#: with how many operands it has, and whether its object is the right one (``1 + s``).
CONVERTED: dict[str, tuple[Callable[..., Any], int, bool]] = {
    **{f"__{name}__": (_operator(name), 2, False) for name in
       "add sub mul truediv floordiv mod lt le gt ge and or xor lshift rshift".split()},
    **{f"__r{name}__": (_operator(name), 2, True) for name in
       "add sub mul truediv floordiv mod and or xor".split()},
    "__neg__": (operator.neg, 1, False),
    "__pos__": (operator.pos, 1, False),
}


def _number(obj: Any) -> Any:
    """The number a convertible object is: by the conversion its class itself has."""
    return getattr(obj, type(obj)._cxx_number_)()


def convertible(cls: type) -> type:
    """A class with ``operator int()`` or ``operator double()``: in arithmetic and as a
    number, it is that number, as C++ converts it - unless it has the operator itself.
    ``==`` stays the object's: a pointer to one compares as a pointer."""
    own = _own(cls)
    integer = "__int__" in own
    cls._cxx_number_ = "__int__" if integer else "__float__"  # type: ignore[attr-defined]
    for name, method in _made(cls, integer).items():
        if name not in own or name == "__index__":
            setattr(cls, name, method)
    return cls


def _own(cls: type) -> set[str]:
    """What a class and its bases define themselves - not what every object has."""
    return {name for klass in cls.__mro__ if klass is not object for name in vars(klass)}


def _made(cls: type, integer: bool) -> dict[str, Callable[..., Any]]:
    """The methods a convertible class is given: its other conversion, and the operators."""
    made = {"__index__": cls.__int__} if integer else {}  # type: ignore[attr-defined]
    made["__float__" if integer else "__int__"] = _as(float if integer else int)
    for name, (apply, operands, reflected) in CONVERTED.items():
        made[name] = _converted(apply, operands, reflected)
    return made


def _as(kind: type) -> Callable[[Any], Any]:
    return lambda self: kind(_number(self))


def _converted(apply: Callable[..., Any], operands: int, reflected: bool) -> Callable[..., Any]:
    """An operator of a convertible object, done on the number it is."""
    if operands == 1:
        return lambda self: apply(_number(self))
    if reflected:
        return lambda self, other: apply(_operand(other), _number(self))
    return lambda self, other: apply(_number(self), _operand(other))


def _operand(value: Any) -> Any:
    """The other side of an operation: a number, or an object converting to one."""
    return _number(value) if hasattr(type(value), "_cxx_number_") else value


class function_template:
    """A function template written once for every use, which a library - PyROOT's
    ``ROOT.f["ULong64_t"]`` - may also ask for an instance of by its arguments: that is
    the same function, as Python's values carry their own types."""

    def __init__(self, function: Callable[..., Any]) -> None:
        self.function = function
        functools.update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)

    def __getitem__(self, arguments: Any) -> Callable[..., Any]:
        return self.function
