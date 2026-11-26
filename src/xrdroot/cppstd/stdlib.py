"""``std::numeric_limits``, the smart pointers and ``std::remove``, for ``ROOT.std``.

``std::numeric_limits<int>::min()`` is a number a macro uses as a sentinel
(df036); ``std::unique_ptr<RooArgSet>{p}`` holds an object the macro goes on
to use, which here is the object itself, Python keeping it while it is
used; ``std::remove(name)`` deletes a file, as C's ``remove`` does, zero
when it did. ``rand`` and ``srand`` are the C library's of macOS, which
ROOT's suite is set up for: the Park-Miller "minimal standard" generator,
seeded with 1, so a program drawing from it draws the numbers it does there.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["numeric_limits", "remove", "smart_pointer", "rand", "srand", "RAND_MAX",
           "complex_type", "atomic_type", "integral_constant", "true_type", "false_type"]

#: ``RAND_MAX``: the largest number ``rand`` gives.
RAND_MAX = 0x7FFFFFFF
#: ``rand``'s state: the last number it gave, or the seed.
_RAND_STATE = [1]


def srand(seed: Any) -> None:
    """``srand``: ``rand`` started again from ``seed``."""
    _RAND_STATE[0] = int(seed) & 0xFFFFFFFF


def rand() -> int:
    """``rand``: the next of the macOS C library's numbers, from 0 to ``RAND_MAX``.

    ``x = 16807 x mod (2^31 - 1)`` by Schrage's method, as its ``do_rand``
    computes it, with a zero state taken as 123459876.
    """
    state = _RAND_STATE[0] or 123459876
    high, low = divmod(state, 127773)
    made = 16807 * low - 2836 * high
    if made < 0:
        made += 0x7FFFFFFF
    _RAND_STATE[0] = made
    return made % (RAND_MAX + 1)

#: The NumPy type of each C++ arithmetic type ``numeric_limits`` is asked of, as LP64 macOS
#: on ARM64 has them: ``char`` signed, ``wchar_t`` a signed 32-bit integer, ``long double``
#: the same as ``double``.
_KINDS = {
    "bool": np.bool_,
    "char": np.int8,
    "signed char": np.int8,
    "unsigned char": np.uint8,
    "wchar_t": np.int32,
    "char8_t": np.uint8,
    "char16_t": np.uint16,
    "char32_t": np.uint32,
    "short": np.int16,
    "unsigned short": np.uint16,
    "int": np.int32,
    "unsigned int": np.uint32,
    "unsigned": np.uint32,
    "long": np.int64,
    "unsigned long": np.uint64,
    "long long": np.int64,
    "unsigned long long": np.uint64,
    "size_t": np.uint64,
    "float": np.float32,
    "double": np.float64,
    "long double": np.float64,
}

#: The members of ``numeric_limits<T>`` that are functions, ``epsilon()``; the rest are
#: constants, ``digits``.
FUNCTIONS = frozenset(
    "min max lowest epsilon round_error infinity quiet_NaN signaling_NaN denorm_min".split()
)


def _floating(info: Any) -> dict[str, Any]:
    """The members of ``numeric_limits`` of an IEEE floating type, from NumPy's description."""
    digits = int(info.nmant) + 1
    return {
        "min": float(info.tiny),
        "max": float(info.max),
        "lowest": -float(info.max),
        "epsilon": float(info.eps),
        "round_error": 0.5,
        "infinity": math.inf,
        "quiet_NaN": math.nan,
        "signaling_NaN": math.nan,
        "denorm_min": float(info.smallest_subnormal),
        "digits": digits,
        "digits10": int(info.precision),
        "max_digits10": math.ceil(digits * math.log10(2) + 1),
        "is_signed": True,
        "is_integer": False,
        "is_exact": False,
        "has_infinity": True,
        "has_quiet_NaN": True,
        "has_signaling_NaN": True,
        "is_iec559": True,
        "is_modulo": False,
        "min_exponent": int(info.minexp) + 1,
        "max_exponent": int(info.maxexp),
        "min_exponent10": math.ceil(math.log10(float(info.tiny))),
        "max_exponent10": math.floor(math.log10(float(info.max))),
    }


def _integral(kind: Any) -> dict[str, Any]:
    """The members of ``numeric_limits`` of an integer type, ``bool`` among them."""
    low: Any
    high: Any
    nothing: Any
    if kind == np.bool_:
        low, high, digits, signed, nothing = False, True, 1, False, False
    else:
        info = np.iinfo(kind)
        low, high, nothing = int(info.min), int(info.max), 0
        signed = low < 0
        digits = info.bits - signed
    found: dict[str, Any] = dict.fromkeys(
        "epsilon round_error infinity quiet_NaN signaling_NaN denorm_min".split(), nothing
    )
    found.update(dict.fromkeys(
        "max_digits10 min_exponent max_exponent min_exponent10 max_exponent10".split(), 0
    ))  # fmt: skip
    found.update(min=low, max=high, lowest=low, digits=digits, is_signed=signed)
    found.update(digits10=int(digits * math.log10(2)), is_integer=True, is_exact=True)
    found.update(dict.fromkeys("has_infinity has_quiet_NaN has_signaling_NaN is_iec559".split(),
                               False))  # fmt: skip
    found["is_modulo"] = not signed and kind != np.bool_
    return found


def limits_of(kind: Any) -> dict[str, Any]:
    """Every member of ``numeric_limits`` of the arithmetic type whose NumPy type is ``kind``."""
    dtype = np.dtype(kind)
    members = _floating(np.finfo(dtype)) if dtype.kind == "f" else _integral(dtype.type)
    members.update(radix=2, is_specialized=True, is_bounded=True)
    return members


def limit(name: str, member: str) -> Any:
    """``std::numeric_limits<name>::member``, for a C++ arithmetic type spelled canonically -
    or ``None`` when ``name`` is not one or ``member`` not one of its members."""
    kind = _KINDS.get(name)
    return None if kind is None else limits_of(kind).get(member)


class _Limits:
    """``std::numeric_limits<T>``: the range and precision of one arithmetic type."""

    def __init__(self, kind: Any) -> None:
        for member, value in limits_of(kind).items():
            setattr(self, member, (lambda value=value: value) if member in FUNCTIONS else value)


class _NumericLimits:
    """``std.numeric_limits``: ``numeric_limits['int']`` is ``numeric_limits<int>``."""

    def __getitem__(self, kind: Any) -> _Limits:
        name = " ".join(str(getattr(kind, "__name__", kind)).replace("std::", "").split())
        if isinstance(kind, type) and kind in (int, float, bool):
            name = {int: "int", float: "double", bool: "bool"}[kind]
        if name not in _KINDS:
            raise TypeError(f"std::numeric_limits is of an arithmetic type, not {name!r}")
        return _Limits(_KINDS[name])


#: ``std.numeric_limits``.
numeric_limits = _NumericLimits()


class _SmartPointer:
    """``std::unique_ptr<T>`` and ``std::shared_ptr<T>``: what they hold is the object."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __getitem__(self, kind: Any) -> _SmartPointer:
        return self

    def __call__(self, held: Any = None) -> Any:
        return held

    def __repr__(self) -> str:
        return f"<std.{self._name} template>"


def smart_pointer(name: str) -> _SmartPointer:
    return _SmartPointer(name)


class _Complex:
    """``std::complex<T>``: a real part and an imaginary one, and C++'s arithmetic on them.

    The parts are kept at the width the template was asked for - a
    ``complex<float>`` keeps single-precision parts, which is what a file of one
    holds - and every operator answers what ``std::complex`` answers: the four
    rules, the two mixed with a number, the comparisons, and ``real``/``imag``
    with and without an argument.
    """

    __slots__ = ("_real", "_imag")
    #: How C++ spells this one, which each class made by :func:`complex_type` sets.
    __cpp_name__ = "complex<double>"
    #: The type of each part, as NumPy's scalar type: ``numpy.float32`` for ``complex<float>``.
    part: Any = np.float64

    def __init__(self, real: Any = 0.0, imag: Any = 0.0) -> None:
        held = getattr(real, "_real", None)
        if held is not None:  # complex(other), or complex(other, 0)
            real, imag = real._real, real._imag
        self._real, self._imag = self.part(real), self.part(imag)

    def _part(self, value: Any) -> tuple[Any, Any]:
        """Another complex's two parts, or a real number's."""
        held = getattr(value, "_real", None)
        return (held, value._imag) if held is not None else (value, 0.0)

    def real(self, value: Any = None) -> Any:
        """``real()``: the real part - or, given one, the complex with that part set."""
        if value is None:
            return _number(self._real)
        self._real = self.part(value)
        return None

    def imag(self, value: Any = None) -> Any:
        if value is None:
            return _number(self._imag)
        self._imag = self.part(value)
        return None

    def __complex__(self) -> complex:
        return complex(float(self._real), float(self._imag))

    def __abs__(self) -> float:
        """``std::abs(z)``: the magnitude."""
        return float(math.hypot(float(self._real), float(self._imag)))

    def _made(self, real: Any, imag: Any) -> _Complex:
        made = object.__new__(type(self))
        made._real, made._imag = self.part(real), self.part(imag)
        return made

    def __add__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        return self._made(self._real + real, self._imag + imag)

    __radd__ = __add__

    def __sub__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        return self._made(self._real - real, self._imag - imag)

    def __rsub__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        return self._made(real - self._real, imag - self._imag)

    def __mul__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        made = self._made
        return made(self._real * real - self._imag * imag, self._real * imag + self._imag * real)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        size = real * real + imag * imag
        return self._made((self._real * real + self._imag * imag) / size,
                          (self._imag * real - self._real * imag) / size)

    def __neg__(self) -> _Complex:
        return self._made(-self._real, -self._imag)

    def __pos__(self) -> _Complex:
        return self

    def __iadd__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        self._real, self._imag = self.part(self._real + real), self.part(self._imag + imag)
        return self

    def __isub__(self, other: Any) -> _Complex:
        real, imag = self._part(other)
        self._real, self._imag = self.part(self._real - real), self.part(self._imag - imag)
        return self

    def __eq__(self, other: object) -> bool:
        real, imag = self._part(other)
        return bool(self._real == real and self._imag == imag)

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        """As ``operator<<`` prints one: ``(1,2)``."""
        return f"({float(self._real):g},{float(self._imag):g})"

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {self}>"


def _number(value: Any) -> Any:
    """A stored part as Python's number, not NumPy's."""
    return value.item() if isinstance(value, np.generic) else value


def _parts(value: Any) -> tuple[float, float]:
    """A complex's two parts as Python's floats, which the functions of ``<complex>`` use."""
    return float(value._real), float(value._imag)


def complex_math(name: str, *args: Any) -> Any:
    """``std::NAME`` of a complex argument, as ``<complex>`` defines it.

    C++ finds these overloads by the type of their argument - ``exp(z)`` of a
    ``complex<double>`` is ``std::exp(std::complex)``, not the C library's - so
    the arithmetic functions dispatch here when one of their arguments is a
    complex, and answer in the same width the argument was.
    """
    found = COMPLEX_MATH.get(name)
    if found is None:
        raise AttributeError(f"xrdroot has no std::{name} of a std::complex")
    return found(args[0], args[1:])


def _magnitude(first: Any, rest: tuple[Any, ...]) -> Any:
    return math.hypot(*_parts(first))


def _norm(first: Any, rest: tuple[Any, ...]) -> Any:
    real, imag = _parts(first)
    return real * real + imag * imag


def _argument(first: Any, rest: tuple[Any, ...]) -> Any:
    return math.atan2(*reversed(_parts(first)))


def _conjugate(first: Any, rest: tuple[Any, ...]) -> Any:
    real, imag = _parts(first)
    return first._made(real, -imag)


def _exponential(first: Any, rest: tuple[Any, ...]) -> Any:
    real, imag = _parts(first)
    size = math.exp(real)
    return first._made(size * math.cos(imag), size * math.sin(imag))


def _logarithm(first: Any, rest: tuple[Any, ...]) -> Any:
    real, imag = _parts(first)
    size = math.hypot(real, imag)
    return first._made(math.log(size) if size else -math.inf, math.atan2(imag, real))


def _logarithm_ten(first: Any, rest: tuple[Any, ...]) -> Any:
    return _logarithm(first, rest) / math.log(10.0)


def _square_root(first: Any, rest: tuple[Any, ...]) -> Any:
    return _exponential(_logarithm(first, rest) / 2.0, ())


def _power(first: Any, rest: tuple[Any, ...]) -> Any:
    """``pow(z, w)``: ``exp(w log z)``, of a real or a complex exponent."""
    exponent, base = rest[0], _logarithm(first, ())
    if hasattr(exponent, "_real"):
        return _exponential(base * exponent, ())
    return _exponential(base._made(base._real * float(exponent), base._imag * float(exponent)), ())


def _trigonometry(first: Any, rest: tuple[Any, ...]) -> Any:
    """``sin(z)``: ``(sin x cosh y, cos x sinh y)`` - and ``cos``, which is the pair swapped."""
    sine = COMPLEX_NAMES_SINE[rest[0]] if rest else "sin"
    real, imag = _parts(first)
    one = math.sin(real) if sine == "sin" else math.cos(real)
    two = math.cos(real) if sine == "sin" else -math.sin(real)
    return first._made(one * math.cosh(imag), two * math.sinh(imag))


def _hyperbolic(first: Any, rest: tuple[Any, ...]) -> Any:
    """``sinh(z)``: ``(sinh x cos y, cosh x sin y)`` - and ``cosh``, the pair swapped."""
    sine = COMPLEX_NAMES_HYPERBOLIC[rest[0]] if rest else "sinh"
    real, imag = _parts(first)
    one = math.sinh(real) if sine == "sinh" else math.cosh(real)
    two = math.cosh(real) if sine == "sinh" else math.sinh(real)
    return first._made(one * math.cos(imag), two * math.sin(imag))


def _tangent(first: Any, rest: tuple[Any, ...]) -> Any:
    """``tan(z)``: ``sin z / cos z`` and, for ``tanh``, ``sinh z / cosh z``."""
    hyperbolic = rest[0] == "tanh" if rest else False
    pair = ("sinh", "cosh") if hyperbolic else ("sin", "cos")
    top = COMPLEX_MATH[pair[0]](first, ())
    return top / COMPLEX_MATH[pair[1]](first, ())


#: What each function of ``<complex>`` is here: the name ``<cmath>`` has for it, and the
#: function that answers it. A complex argument is what sends a call here.
COMPLEX_MATH: dict[str, Callable[[Any, tuple[Any, ...]], Any]] = {
    "abs": _magnitude, "norm": _norm, "arg": _argument, "conj": _conjugate,
    "exp": _exponential, "log": _logarithm, "log10": _logarithm_ten,
    "sqrt": _square_root, "pow": _power,
    "sin": lambda first, rest: _trigonometry(first, ("sin",)),
    "cos": lambda first, rest: _trigonometry(first, ("cos",)),
    "tan": lambda first, rest: _tangent(first, ("tan",)),
    "sinh": lambda first, rest: _hyperbolic(first, ("sinh",)),
    "cosh": lambda first, rest: _hyperbolic(first, ("cosh",)),
    "tanh": lambda first, rest: _tangent(first, ("tanh",)),
}  # fmt: skip

#: The two of each pair of the trigonometric and hyperbolic functions.
COMPLEX_NAMES_SINE = {"sin": "sin", "cos": "cos"}
COMPLEX_NAMES_HYPERBOLIC = {"sinh": "sinh", "cosh": "cosh"}


#: The functions of ``<complex>`` a complex argument dispatches to
#: :func:`complex_math` for, by the name ``<cmath>``'s function has.
COMPLEX_NAMES = frozenset({"exp", "log", "log10", "sqrt", "pow", "sin", "cos", "tan", "sinh",
                           "cosh", "tanh", "abs", "norm", "arg", "conj"})  # fmt: skip


#: What each C++ name a complex is asked for keeps its parts in, and how C++ spells it.
COMPLEX_PARTS: dict[str, tuple[Any, str]] = {
    "float": (np.float32, "float"),
    "double": (np.float64, "double"),
    "long double": (np.float64, "long double"),
}


#: What NumPy calls each width a template may be asked for, as C++ spells it.
NUMPY_NAMES: dict[str, str] = {"float32": "float", "float64": "double", "int8": "char",
                               "uint8": "unsigned char", "int16": "short",
                               "uint16": "unsigned short", "int32": "int",
                               "uint32": "unsigned int", "int64": "long", "uint64": "unsigned long",
                               "bool": "bool"}  # fmt: skip


def _cxx_name(kind: Any) -> str:
    """A template argument's C++ name, whether it came as a name or as a NumPy type."""
    if isinstance(kind, str):
        return kind
    if isinstance(kind, np.dtype):
        name: str = kind.name
        return NUMPY_NAMES.get(name, name)
    if isinstance(kind, type) and issubclass(kind, np.generic):
        scalar: str = np.dtype(kind).name
        return NUMPY_NAMES.get(scalar, scalar)
    return str(getattr(kind, "__name__", kind))


#: The one class made for each template argument set, so that either spelling is the class.
_COMPLEX_CLASSES: dict[str, type] = {}
_ATOMIC_CLASSES: dict[str, type] = {}


def complex_type(kind: Any) -> type:
    """``std::complex<float>`` and its kin: the class of that name keeping that width.

    C++ makes one of a floating type and of nothing else, so a complex of a
    class is refused by name rather than kept at a width it never had.
    """
    spelled = _cxx_name(kind)
    known = COMPLEX_PARTS.get(spelled)
    if known is None:
        raise TypeError(
            f"std::complex<{spelled}> is not a complex this kit has: C++ makes one of float, "
            f"double or long double, and {spelled} is none of them")
    part, name = known
    if name not in _COMPLEX_CLASSES:
        members = {"part": part, "__cpp_name__": f"complex<{name}>"}
        _COMPLEX_CLASSES[name] = type(f"complex<{name}>", (_Complex,), {"__slots__": (), **members})
    return _COMPLEX_CLASSES[name]


class _Atomic:
    """``std::atomic<T>``: a value read and written whole, as C++'s atomics are.

    Nothing here runs two threads through one atomic - xrdroot's event loops are
    one thread, or processes - so what is kept is the value and the operations
    C++ names for it, so that a program counting in a lambda counts.
    """

    __slots__ = ("_value",)
    #: How C++ spells this one, which each class made by :func:`atomic_type` sets.
    __cpp_name__ = "atomic<int>"
    #: The type of the value: Python's ``int``, or ``float``, or ``bool``.
    value_type: Any = int
    #: A cell's word for "``*p`` reads ``.value``", which is what an atomic is read through.
    _cint_cell = True

    def __init__(self, value: Any = 0) -> None:
        self._value = self.value_type(value)

    @property
    def value(self) -> Any:
        """``operator T()`` and ``*p``: the value as it is now."""
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        self._value = self.value_type(value)

    def load(self) -> Any:
        return self._value

    def store(self, value: Any) -> None:
        self._value = self.value_type(value)

    def exchange(self, value: Any) -> Any:
        """``exchange(v)``: the value it had, with ``v`` in its place."""
        old, self._value = self._value, self.value_type(value)
        return old

    def fetch_add(self, value: Any = 1) -> Any:
        old, self._value = self._value, self.value_type(self._value + value)
        return old

    def fetch_sub(self, value: Any = 1) -> Any:
        old, self._value = self._value, self.value_type(self._value - value)
        return old

    def __iadd__(self, value: Any) -> _Atomic:
        self._value = self.value_type(self._value + value)
        return self

    def __isub__(self, value: Any) -> _Atomic:
        self._value = self.value_type(self._value - value)
        return self

    def __add__(self, value: Any) -> Any:
        return self._value + value

    __radd__ = __add__

    def __sub__(self, value: Any) -> Any:
        return self._value - value

    def __rsub__(self, value: Any) -> Any:
        return value - self._value

    def __mul__(self, value: Any) -> Any:
        return self._value * value

    __rmul__ = __mul__

    def __truediv__(self, value: Any) -> Any:
        return self._value / value

    def __lt__(self, value: Any) -> bool:
        return bool(self._value < value)

    def __le__(self, value: Any) -> bool:
        return bool(self._value <= value)

    def __gt__(self, value: Any) -> bool:
        return bool(self._value > value)

    def __ge__(self, value: Any) -> bool:
        return bool(self._value >= value)

    def __int__(self) -> int:
        return int(self._value)

    def __index__(self) -> int:
        return int(self._value)

    def __float__(self) -> float:
        return float(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        return bool(self._value == getattr(other, "_value", other))

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {self._value!r}>"


#: The value type ``std::atomic<T>`` keeps for each C++ arithmetic name.
ATOMIC_VALUES: dict[str, Any] = {
    "bool": bool, "char": int, "signed char": int, "unsigned char": int,
    "short": int, "unsigned short": int, "int": int, "unsigned int": int,
    "long": int, "unsigned long": int, "long long": int, "unsigned long long": int,
    "float": float, "double": float,
}  # fmt: skip


def atomic_type(kind: Any) -> type:
    """``std::atomic<int>`` and its kin: the class of that name keeping that value.

    ``std::atomic_int`` and the other ``std::atomic_*`` spellings are the same
    classes, so one made through either name is the one class.
    """
    spelled = _cxx_name(kind)
    if spelled.startswith("atomic_"):  # ``std::atomic_int`` is ``std::atomic<int>``
        spelled = spelled[len("atomic_"):]
    elif spelled.startswith("atomic<") and spelled.endswith(">"):
        spelled = spelled[len("atomic<"):-1]
    name = f"atomic<{spelled}>"
    if name not in _ATOMIC_CLASSES:
        members = {"value_type": ATOMIC_VALUES.get(spelled, int), "__cpp_name__": name}
        _ATOMIC_CLASSES[name] = type(name, (_Atomic,), {"__slots__": (), **members})
    return _ATOMIC_CLASSES[name]


class _Forwarded:
    """What ``std::forward_as_tuple`` gives: the arguments, kept for the constructor that
    reads them - which is how a ``std::map`` of ``std::unique_ptr`` is filled, the key and
    the value each built from their own tuple."""

    __slots__ = ("_values",)

    def __init__(self, *values: Any) -> None:
        self._values = tuple(values)

    def __iter__(self) -> Any:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> Any:
        return self._values[index]

    def __repr__(self) -> str:
        return f"<tuple {self._values!r}>"


def forward_as_tuple(*values: Any) -> _Forwarded:
    """``std::forward_as_tuple(values...)``: the values as the tuple a constructor reads."""
    return _Forwarded(*values)


class _Tag:
    """A tag C++ only ever passes: the type is the argument, the value says nothing."""

    __slots__ = ("__cpp_name__",)

    def __init__(self, name: str) -> None:
        self.__cpp_name__ = name

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} tag>"


#: What an empty ``std::any`` holds: no Python value can be it, so ``has_value`` can tell.
_EMPTY = object()


def kind_of(value: Any) -> str:
    """The C++ spelling of a Python value's type, as ``std::any::type`` names it."""
    if isinstance(value, AnyValue):
        return value.kind or "void"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "double"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


class AnyValue:
    """``std::any``: one value of any type, kept with the type it was made as.

    ``std::any_cast<T>(a)`` is ``std.any_cast['T'](a)`` here, converting what it holds to ``T``
    as the cast does, and ``std::make_any<T>(args)`` is ``std.make_any['T'](args)``.
    """

    def __init__(self, value: Any = _EMPTY, kind: str = "") -> None:
        self._value = value
        self._kind = kind

    @property
    def kind(self) -> str:
        """The C++ spelling of the held type, or ``""`` for an empty any."""
        return self._kind

    @property
    def held(self) -> Any:
        """The value itself, empty or not: what ``TPython::Exec`` reads to fill another any."""
        return self._value

    def has_value(self) -> bool:
        return self._value is not _EMPTY

    def reset(self) -> None:
        self._value, self._kind = _EMPTY, ""

    def type(self) -> Any:
        """``std::any::type``: the held type's name, or ``void`` when it holds nothing."""
        return _Tag(self._kind or "void")

    def value(self) -> Any:
        if not self.has_value():
            raise RuntimeError("std::any::value: this any holds nothing; give it a value first")
        return self._value

    def set(self, value: Any, kind: str = "") -> None:
        """Hold ``value``: what ``TPython::Exec`` writes into the any the C++ side handed it."""
        self._value, self._kind = value, kind or kind_of(value)

    def __bool__(self) -> bool:
        return self.has_value()

    def __repr__(self) -> str:
        held = "empty" if not self.has_value() else f"{self._kind} {self._value!r}"
        return f"<std::any {held}>"


_TYPE_CONVERTERS = {"int": int, "long": int, "double": float, "float": float, "bool": bool,
                    "string": str, "std::string": str}


class _AnyCast:
    """``std::any_cast<T>(a)``: ``std.any_cast['T'](a)``, giving T's value back."""

    def __getitem__(self, kind: str) -> Callable[..., Any]:
        def cast(held: Any) -> Any:
            value = held.value() if isinstance(held, AnyValue) else held
            converter = _TYPE_CONVERTERS.get(kind)
            return converter(value) if converter is not None else value

        return cast


class _MakeAny:
    """``std::make_any<T>(args)``: ``std.make_any['T'](args)``, an any holding a T."""

    def __getitem__(self, kind: str) -> Callable[..., Any]:
        def make(*args: Any) -> AnyValue:
            value = args[0] if len(args) == 1 else args
            return AnyValue(value, kind)

        return make


#: ``std::any_cast<T>`` and ``std::make_any<T>``: the subscript names ``T``.
any_cast = _AnyCast()
make_any = _MakeAny()


def remove(path: Any) -> int:
    """C's ``remove``: the file deleted, and 0 - or -1, if there was none to delete."""
    try:
        os.remove(str(path))
    except OSError:
        return -1
    return 0


class integral_constant:
    """``std::integral_constant``: a type that stands for a value a template chooses on.

    A ROOT test writes ``struct IsCollectionProxy<X> : std::true_type {}`` - the base says the
    specialisation is the true one - so the base has to be a class here, with the value it stands
    for reachable as ``value``, by ``bool``, and by calling it, as ``std::integral_constant`` is.
    """

    #: The value this type is, which a subclass fixes.
    value: Any = 0

    def __bool__(self) -> bool:
        return bool(bool(self.value))

    def __call__(self) -> Any:
        return self.value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type) and issubclass(other, integral_constant):
            return bool(self.value == other.value)
        return bool(self.value == other)

    def __hash__(self) -> int:
        return hash(self.value)


class true_type(integral_constant):
    """``std::true_type``: ``integral_constant<bool, true>``."""

    value = True


class false_type(integral_constant):
    """``std::false_type``: ``integral_constant<bool, false>``."""

    value = False
