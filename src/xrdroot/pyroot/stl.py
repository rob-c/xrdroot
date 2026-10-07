"""``ROOT.std``: the C++ standard library's containers, as PyROOT scripts use them.

    >>> from xrdroot.pyroot.stl import std
    >>> v = std.vector['float']()
    >>> v.push_back(1.5); v.push_back(2.5)
    >>> v.size(), list(v)
    (2, [1.5, 2.5])
    >>> std.vector('double')(3, 0.5).data()
    array([0.5, 0.5, 0.5])

A PyROOT script makes a ``std::vector<float>`` as ``ROOT.std.vector['float']()``
or ``ROOT.std.vector('float')()`` - both spellings give the same class, one per
element type - and hands it to ``TTree::Branch`` or ``SetBranchAddress``,
which read it and fill it in place. That is what these are for, so the
numbers are kept in a NumPy array that grows as C++'s does, and ``data()`` is
a view of them rather than a copy.

The spelling a C++ macro is translated to is exactly this one: ``std::vector<float>
v;`` becomes ``v = std.vector['float']()``, ``std::vector<std::vector<int>>`` is
``std.vector['std::vector<int>']``, ``std::map<std::string, int>`` is
``std.map['std::string', 'int']`` and ``std::pair<int, double>(1, 2.)`` is
``std.pair['int', 'double'](1, 2.)``. An element type may be spelled as C++
does (``'float'``, ``'unsigned int'``, ``'std::string'``), as ROOT does
(``'Float_t'``, ``'Long64_t'``), as a Python type (``float`` is ``double``,
``int`` is ``int``) or as one of these classes (``std.string``,
``std.vector['int']``). A vector of numbers is the fast kind; a vector of
anything else - strings, vectors - keeps its elements in a list.

What is not here - ``std.deque``, ``std.cout`` and the rest - is refused by
name, as the rest of ``xrdroot.pyroot`` refuses what it does not have.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from . import stlrandom as _random

__all__ = ["std"]

#: What ``find`` gives back when there is nothing to find: ``std::string::npos``.
NPOS = 2**64 - 1

#: The number types C++ and ROOT spell, and the NumPy type each is.
NUMBERS: dict[str, str] = {
    "float": "float32",
    "Float_t": "float32",
    "Float16_t": "float32",
    "double": "float64",
    "Double_t": "float64",
    "Double32_t": "float64",
    "long double": "float64",
    "char": "int8",
    "Char_t": "int8",
    "signed char": "int8",
    "int8_t": "int8",
    "unsigned char": "uint8",
    "UChar_t": "uint8",
    "uint8_t": "uint8",
    "short": "int16",
    "Short_t": "int16",
    "int16_t": "int16",
    "unsigned short": "uint16",
    "UShort_t": "uint16",
    "uint16_t": "uint16",
    "int": "int32",
    "Int_t": "int32",
    "int32_t": "int32",
    "unsigned int": "uint32",
    "unsigned": "uint32",
    "UInt_t": "uint32",
    "uint32_t": "uint32",
    "long": "int64",
    "Long_t": "int64",
    "long long": "int64",
    "Long64_t": "int64",
    "int64_t": "int64",
    "unsigned long": "uint64",
    "ULong_t": "uint64",
    "unsigned long long": "uint64",
    "ULong64_t": "uint64",
    "uint64_t": "uint64",
    "size_t": "uint64",
    "bool": "bool",
    "Bool_t": "bool",
}

#: The C++ name each NumPy type is printed with, as ROOT prints a vector's class.
CPP_NAMES: dict[str, str] = {
    "float32": "float",
    "float64": "double",
    "int8": "char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long",
    "uint64": "unsigned long",
    "bool": "bool",
}

#: What a Python type given as an element type stands for, as cppyy takes it.
PYTHON_TYPES: dict[type, str] = {float: "float64", int: "int32", bool: "bool"}

#: The names a string is spelled with.
STRINGS = ("string", "std::string", "TString")


def _spelled(name: str) -> str:
    """A C++ type name with its ``std::`` and spare spaces taken out."""
    text = " ".join(name.replace("std::", "").split())
    return re.sub(r"\s*([<>,])\s*", r"\1", text)


def _split_arguments(text: str) -> list[str]:
    """The template arguments inside ``<...>``, split at the commas outside any brackets."""
    parts, depth, start = [], 0, 0
    for at, char in enumerate(text):
        depth += (char == "<") - (char == ">")
        if char == "," and depth == 0:
            parts.append(text[start:at])
            start = at + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


#: The classes standing for element types kept as they are given: ``TH1F*``, ``Interval*``.
_OPAQUE: dict[str, type] = {}

#: The smart pointers, whose elements are the objects they point to - ``None`` until set.
SMART_POINTERS = frozenset({"unique_ptr", "shared_ptr"})


def _opaque(text: str) -> type:
    """An element type that is neither a number, a string nor a container: kept as given.

    A ``std::vector<TMVA::Interval*>`` holds whatever objects are pushed into
    it - the pointers C++ would hold are the objects themselves here.
    """
    made = _OPAQUE.get(text)
    if made is None:
        members = {"__cpp_name__": text, "opaque": True, "emplaced": _emplaced(text)}
        made = type(text, (), members)
        _OPAQUE[text] = made
    return made


def _emplaced(text: str) -> Any:
    """The class ``emplace_back`` makes an element of, for the one kind a vector must make:
    ``std::thread``, started from the function it is given."""
    if text != "thread":
        return None
    from ..cint.runtime.threads import thread

    return thread


def _templated(text: str) -> Any:
    """A type with template arguments: ``vector<float>``, ``map<string,int>``, ``pair<...>``."""
    outer, _, inner = text.partition("<")
    if (not inner and re.fullmatch(r"[A-Za-z_][\w:]*\s*\**", text)) or outer in SMART_POINTERS:
        return _opaque(text)
    maker = TEMPLATES.get(outer)
    if maker is None or not inner.endswith(">"):
        raise TypeError(
            f"{text!r} is not a type the std stand-ins know; they have vector, map and pair "
            f"of numbers, strings and one another"
        )
    return maker[tuple(_split_arguments(inner[:-1]))]


def element_type(given: Any) -> Any:
    """What an element type is: a NumPy dtype for a number, or a class for anything else.

    A string's element type is :class:`string`; a vector's, map's or pair's is
    its class, so a ``vector<vector<int>>`` holds ``vector<int>`` objects.
    """
    if isinstance(given, type):
        return _class_type(given)
    if isinstance(given, np.dtype):
        return given
    if not isinstance(given, str):
        raise TypeError(f"{given!r} is not a type an STL container can hold")
    text = _spelled(given)
    if text in NUMBERS:
        return np.dtype(NUMBERS[text])
    if text in STRINGS:
        return string
    return _templated(text)


def _class_type(given: type) -> Any:
    """An element type given as a class: a Python or NumPy number, a string, a container."""
    if given in PYTHON_TYPES:
        return np.dtype(PYTHON_TYPES[given])
    if issubclass(given, np.generic):
        return np.dtype(given)
    if issubclass(given, (string, _Container)):
        return given
    if issubclass(given, str):
        return string
    raise TypeError(f"{given.__name__} is not a type an STL container can hold")


def cpp_name(kind: Any) -> str:
    """How C++ spells an element type: ``float``, ``string``, ``vector<int>``."""
    if isinstance(kind, np.dtype):
        return CPP_NAMES[kind.name]
    return str(kind.__cpp_name__)


def _converted(kind: Any, value: Any) -> Any:
    """A value as an element of this type is kept: a string, or a container made from it."""
    if kind is string:
        return string(value)
    if getattr(kind, "opaque", False):
        return value
    if isinstance(value, kind):
        return value
    return kind(value)


class string:
    """``std::string``: text that can be changed where it is, as C++'s can.

        >>> s = std.string("hello")
        >>> s += " world"
        >>> s.size(), s.find("world"), s.find("nothing") == std.string.npos
        (11, 6, True)

    A branch of ``/C`` or of ``std::string`` reads what it holds and a tree's
    ``GetEntry`` puts the entry's text into it, which a Python ``str`` could
    not have done in place.
    """

    __slots__ = ("_text",)
    __cpp_name__ = "string"
    #: ``std::string::npos``: what ``find`` gives when it finds nothing.
    npos = NPOS

    def __init__(self, text: Any = "") -> None:
        self._text = text.decode() if isinstance(text, bytes) else str(text)

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return repr(self._text)

    def __eq__(self, other: object) -> bool:
        return self._text == str(other) if isinstance(other, (str, string)) else False

    def __hash__(self) -> int:
        return hash(self._text)

    def __len__(self) -> int:
        return len(self._text)

    def __add__(self, other: Any) -> string:
        return string(self._text + str(other))

    def __radd__(self, other: Any) -> string:
        return string(str(other) + self._text)

    def __iadd__(self, other: Any) -> string:
        self._text += str(other)
        return self

    def __getitem__(self, index: Any) -> str:
        return self._text[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self._text)

    def __lt__(self, other: Any) -> bool:
        return self._text < str(other)

    def size(self) -> int:
        return len(self._text)

    length = size

    def empty(self) -> bool:
        return not self._text

    def c_str(self) -> str:
        return self._text

    data = c_str

    def clear(self) -> None:
        self._text = ""

    def assign(self, text: Any) -> None:
        self._text = str(text)

    def append(self, text: Any) -> string:
        self._text += str(text)
        return self

    def find(self, what: Any, start: int = 0) -> int:
        """Where ``what`` first is from ``start`` on, or :attr:`npos`."""
        found = self._text.find(str(what), start)
        return NPOS if found < 0 else found

    def rfind(self, what: Any) -> int:
        found = self._text.rfind(str(what))
        return NPOS if found < 0 else found

    def substr(self, start: int = 0, count: int = NPOS) -> string:
        return string(self._text[start : start + count])

    def compare(self, other: Any) -> int:
        text = str(other)
        return (self._text > text) - (self._text < text)


class _Container:
    """What every container class here shares: the name C++ gives it."""

    __slots__ = ()
    __cpp_name__ = ""


class _Template:
    """``std.vector``: a class for each element type, by ``[...]`` or by ``(...)``.

    Both spellings give the one class for the one type, so ``isinstance``
    and a tree's check of what it was handed work whichever was used.
    """

    def __init__(self, name: str, arity: int, build: Any) -> None:
        self._name = name
        self._arity = arity
        self._build = build
        self._made: dict[tuple[Any, ...], type] = {}

    def __repr__(self) -> str:
        return f"<std.{self._name} template>"

    def __getitem__(self, arguments: Any) -> Any:
        given = arguments if isinstance(arguments, tuple) else (arguments,)
        if len(given) != self._arity:
            raise TypeError(
                f"std::{self._name} takes {self._arity} template argument"
                f"{'' if self._arity == 1 else 's'}, and was given {len(given)}"
            )
        kinds = tuple(element_type(each) for each in given)
        key = tuple(cpp_name(kind) for kind in kinds)
        if key not in self._made:
            self._made[key] = self._build(*kinds)
        return self._made[key]

    def __call__(self, *arguments: Any) -> Any:
        return self[arguments]


class _Vector(_Container):
    """``std::vector`` of numbers: a NumPy array that grows by doubling, as C++'s does."""

    __slots__ = ("_data", "_size")
    #: The element type, as NumPy calls it.
    dtype: np.dtype[Any] = np.dtype("float64")
    value_type = "double"

    def __init__(self, *arguments: Any) -> None:
        self._data = np.zeros(0, dtype=self.dtype)
        self._size = 0
        if arguments:
            self._construct(*arguments)

    def _construct(self, first: Any, *rest: Any) -> None:
        """``vector(n[, value])`` - a real ``n`` taken as C++ converts it, whole - or ``vector
        (values)``."""
        if isinstance(first, (int, float, np.integer, np.floating)) and not isinstance(first, bool):
            self.resize(int(first), *rest)
        else:
            self.assign(first)

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {self.data().tolist()!r}>"

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self.data().tolist())

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return self.data()[index].copy()
        return self.data()[self._checked(index)].item()

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice):  # std::sort's range, written back where it was
            self.data()[index] = value
            return
        self._data[self._checked(index)] = value

    def __eq__(self, other: object) -> bool:
        try:
            given = np.asarray(list(other))  # type: ignore[call-overload]
        except TypeError:
            return False
        return bool(len(given) == self._size and np.array_equal(given, self.data()))

    __hash__ = None  # type: ignore[assignment]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray[Any, Any]:
        """The values, as NumPy asks for them: a copy when it asks for one."""
        made = np.asarray(self.data(), dtype=dtype)
        return made.copy() if copy else made

    def _checked(self, index: Any) -> int:
        at = int(index)
        at = at + self._size if at < 0 else at
        if not 0 <= at < self._size:
            raise IndexError(f"{index} is outside this {self.__cpp_name__} of {self._size}")
        return at

    def _room(self, size: int) -> None:
        if size > len(self._data):
            grown = np.zeros(max(size, 2 * len(self._data)), dtype=self.dtype)
            grown[: self._size] = self._data[: self._size]
            self._data = grown

    def push_back(self, value: Any) -> None:
        self._room(self._size + 1)
        self._data[self._size] = value
        self._size += 1

    emplace_back = push_back

    def pop_back(self) -> None:
        self._checked(-1)
        self._size -= 1

    def size(self) -> int:
        return self._size

    def empty(self) -> bool:
        return self._size == 0

    def capacity(self) -> int:
        return len(self._data)

    def reserve(self, size: int) -> None:
        self._room(int(size))

    def clear(self) -> None:
        self._size = 0

    def resize(self, size: int, value: Any = 0) -> None:
        size = int(size)
        self._room(size)
        if size > self._size:
            self._data[self._size : size] = value
        self._size = size

    def assign(self, values: Iterable[Any]) -> None:
        """Replace everything held with ``values``: any sequence, array or vector."""
        given = np.asarray(values if isinstance(values, np.ndarray) else list(values))
        self._data = given.astype(self.dtype, copy=True).reshape(-1)
        self._size = len(self._data)

    def data(self) -> np.ndarray[Any, Any]:
        """The elements, as a NumPy view of the vector's own storage."""
        return self._data[: self._size]

    def at(self, index: int) -> Any:
        return self[index]

    def front(self) -> Any:
        return self[0]

    def back(self) -> Any:
        return self[-1]

    def begin(self) -> _VectorIterator:
        """``begin()``: an iterator at the first element, which ``*it`` and ``it += 1`` use."""
        return _VectorIterator(self, 0)

    def end(self) -> _VectorIterator:
        return _VectorIterator(self, self._size)


class _ObjectVector(_Container):
    """``std::vector`` of strings or of other containers, kept in a list."""

    __slots__ = ("_items",)
    #: The class every element is made into.
    element: Any = string
    value_type = "string"
    dtype = np.dtype(object)

    def __init__(self, *arguments: Any) -> None:
        self._items: list[Any] = []
        if arguments:
            self._construct(*arguments)

    def _construct(self, first: Any, *rest: Any) -> None:
        if isinstance(first, (int, np.integer)):
            self.resize(int(first), *rest)
        else:
            self.assign(first)

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {self._items!r}>"

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __getitem__(self, index: Any) -> Any:
        return self._items[index]

    def __setitem__(self, index: int, value: Any) -> None:
        self._items[index] = _converted(self.element, value)

    def __eq__(self, other: object) -> bool:
        try:
            return bool(self._items == list(other))  # type: ignore[call-overload]
        except TypeError:
            return False

    __hash__ = None  # type: ignore[assignment]

    def push_back(self, value: Any) -> None:
        self._items.append(_converted(self.element, value))

    def emplace_back(self, *args: Any) -> None:
        """``emplace_back(args...)``: an element made of ``args`` - a thread started from a
        function - or, for any other kind, the one value given."""
        made = getattr(self.element, "emplaced", None)
        if made is not None and not (len(args) == 1 and isinstance(args[0], made)):
            self._items.append(made(*args))
            return
        self.push_back(*args)

    def pop_back(self) -> None:
        self._items.pop()

    def size(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    def reserve(self, size: int) -> None:
        """``reserve``: room made ahead, which a Python list makes as it grows."""

    def capacity(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def resize(self, size: int, value: Any = None) -> None:
        del self._items[size:]
        while len(self._items) < size:
            self._items.append(_default(self.element) if value is None else
                               _converted(self.element, value))  # fmt: skip

    def assign(self, values: Iterable[Any]) -> None:
        self._items = [_converted(self.element, value) for value in values]

    def data(self) -> list[Any]:
        return self._items

    def at(self, index: int) -> Any:
        return self._items[index]

    def front(self) -> Any:
        return self._items[0]

    def back(self) -> Any:
        return self._items[-1]

    def begin(self) -> _VectorIterator:
        """``begin()``: an iterator at the first element, which ``*it`` and ``it += 1`` use."""
        return _VectorIterator(self, 0)

    def end(self) -> _VectorIterator:
        return _VectorIterator(self, len(self._items))


class _VectorIterator:
    """A ``std::vector`` iterator: ``*it`` - here the iterator itself - is the element it is at.

    A translated macro writes ``(*it)->GetMin()`` as ``it.GetMin()`` and
    ``cout << *it`` as ``cout << it``, so the element's attributes are the
    iterator's, and :meth:`__deref__` is what a stream writes.
    """

    def __init__(self, owner: Any, position: int) -> None:
        self._owner, self._position = owner, position

    def __deref__(self) -> Any:
        return self._owner[self._position]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.__deref__(), name)

    def __float__(self) -> float:
        return float(self.__deref__())

    def __iadd__(self, step: int) -> _VectorIterator:
        self._position += int(step)
        return self

    def __add__(self, step: int) -> _VectorIterator:
        return _VectorIterator(self._owner, self._position + int(step))

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, _VectorIterator):
            return self._position - other._position
        return _VectorIterator(self._owner, self._position - int(other))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _VectorIterator):
            return NotImplemented
        return self._owner is other._owner and self._position == other._position

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def __lt__(self, other: _VectorIterator) -> bool:
        return self._position < other._position

    __hash__ = None  # type: ignore[assignment]


def _vector_class(kind: Any) -> type:
    """The class ``std::vector<kind>`` is, made once per element type."""
    name = f"vector<{cpp_name(kind)}>"
    if isinstance(kind, np.dtype):
        members = {"dtype": kind, "value_type": cpp_name(kind), "__cpp_name__": name}
        return type(name, (_Vector,), {"__slots__": (), **members})
    members = {"element": kind, "value_type": cpp_name(kind), "__cpp_name__": name}
    return type(name, (_ObjectVector,), {"__slots__": (), **members})


def _element(kind: Any, value: Any) -> Any:
    """A value as an element of type ``kind`` is kept in a map or pair."""
    if isinstance(kind, np.dtype):
        return np.asarray(value, dtype=kind).item()
    return _converted(kind, value)


def _default(kind: Any) -> Any:
    """What a new element of type ``kind`` is before anything is put in it: zero, or empty."""
    if isinstance(kind, np.dtype):
        return np.zeros((), dtype=kind).item()
    return None if getattr(kind, "opaque", False) else kind()


class _Pair(_Container):
    """``std::pair``: ``first`` and ``second``, and unpacked as two things."""

    __slots__ = ("first", "second")
    kinds: tuple[Any, Any] = (np.dtype("float64"), np.dtype("float64"))

    def __init__(self, first: Any = None, second: Any = None) -> None:
        one, two = self.kinds
        self.first = _default(one) if first is None else _element(one, first)
        self.second = _default(two) if second is None else _element(two, second)

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} ({self.first!r}, {self.second!r})>"

    def __iter__(self) -> Iterator[Any]:
        return iter((self.first, self.second))

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        return (self.first, self.second)[index]

    def __eq__(self, other: object) -> bool:
        try:
            return tuple(self) == tuple(other)  # type: ignore[arg-type]
        except TypeError:
            return False

    __hash__ = None  # type: ignore[assignment]


def _pair_class(first: Any, second: Any) -> type:
    name = f"pair<{cpp_name(first)},{cpp_name(second)}>"
    members = {"kinds": (first, second), "__cpp_name__": name}
    return type(name, (_Pair,), {"__slots__": (), **members})


class _Map(_Container):
    """``std::map``: kept in key order, with ``operator[]`` making what it does not find.

    Iterating gives pairs, as cppyy's does - ``for key, value in m`` unpacks
    them - and ``m[key]`` of a key not there puts a zero or an empty one in,
    as C++'s does.
    """

    __slots__ = ("_items",)
    kinds: tuple[Any, Any] = (string, np.dtype("float64"))

    def __init__(self, items: Any = None) -> None:
        self._items: dict[Any, Any] = {}
        if items is not None:
            pairs = items.items() if isinstance(items, dict) else items
            for key, value in pairs:
                self[key] = value

    def _key(self, key: Any) -> Any:
        return _element(self.kinds[0], key)

    def __repr__(self) -> str:
        return f"<{self.__cpp_name__} {dict(self.items())!r}>"

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Any]:
        made = _pair_class(*self.kinds)
        for key, value in self.items():
            yield made(key, value)

    def __contains__(self, key: Any) -> bool:
        return self._key(key) in self._items

    def __getitem__(self, key: Any) -> Any:
        found = self._key(key)
        if found not in self._items:
            self._items[found] = _default(self.kinds[1])
        return self._items[found]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._items[self._key(key)] = _element(self.kinds[1], value)

    def items(self) -> list[tuple[Any, Any]]:
        """Every key and its value, in the key order a ``std::map`` keeps."""
        return sorted(self._items.items(), key=lambda item: item[0])

    def keys(self) -> list[Any]:
        return [key for key, _ in self.items()]

    def values(self) -> list[Any]:
        return [value for _, value in self.items()]

    def size(self) -> int:
        return len(self._items)

    def empty(self) -> bool:
        return not self._items

    def clear(self) -> None:
        self._items.clear()

    def count(self, key: Any) -> int:
        return int(key in self)

    def at(self, key: Any) -> Any:
        if key not in self:
            raise IndexError(f"{key!r} is not a key of this {self.__cpp_name__}")
        return self._items[self._key(key)]

    def erase(self, key: Any) -> int:
        return int(self._items.pop(self._key(key), None) is not None)

    def insert(self, pair: Any) -> None:
        key, value = pair
        if key not in self:
            self[key] = value

    def begin(self) -> _MapIterator:
        """``begin()``: an iterator at the first key, which ``it += 1`` moves along."""
        return _MapIterator(self, 0)

    def end(self) -> _MapIterator:
        """``end()``: the iterator one past the last key."""
        return _MapIterator(self, len(self._items))

    def find(self, key: Any) -> _MapIterator:
        """``find(key)``: an iterator at ``key``, or ``end()`` for a key not there."""
        keys = self.keys()
        found = self._key(key)
        return _MapIterator(self, keys.index(found) if found in self._items else len(keys))


class _MapIterator:
    """A ``std::map`` iterator: ``it->first``, ``it->second`` (assignable), ``++it``."""

    def __init__(self, owner: _Map, position: int) -> None:
        self._owner, self._position = owner, position

    def _key(self) -> Any:
        return self._owner.keys()[self._position]

    @property
    def first(self) -> Any:
        return self._key()

    @property
    def second(self) -> Any:
        return self._owner[self._key()]

    @second.setter
    def second(self, value: Any) -> None:
        self._owner[self._key()] = value

    def __iadd__(self, step: int) -> _MapIterator:
        self._position += int(step)
        return self

    def __add__(self, step: int) -> _MapIterator:
        return _MapIterator(self._owner, self._position + int(step))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MapIterator):
            return NotImplemented
        return self._owner is other._owner and self._position == other._position

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    __hash__ = None  # type: ignore[assignment]


def _map_class(key: Any, value: Any) -> type:
    name = f"map<{cpp_name(key)},{cpp_name(value)}>"
    members = {"kinds": (key, value), "__cpp_name__": name}
    return type(name, (_Map,), {"__slots__": (), **members})


#: The templates, by the name C++ gives each.
def _rvec_of(kind: Any) -> type:
    """``RVec<kind>``, as an element type spells it: ``RVec<RVec<size_t>>``'s inner one."""
    from .rdf.rvec import RVec

    return RVec[kind]  # type: ignore[no-any-return]


TEMPLATES: dict[str, _Template] = {
    "vector": _Template("vector", 1, _vector_class),
    "map": _Template("map", 2, _map_class),
    "pair": _Template("pair", 2, _pair_class),
    **{
        name: _Template("RVec", 1, _rvec_of)
        for name in ("RVec", "ROOT::RVec", "ROOT::VecOps::RVec", "VecOps::RVec")
    },
}


class _ArrayTemplate:
    """``std.array['double', 3]``: a vector of that many, filled from the list it is given.

    It is the vector class of its element type, so it indexes, iterates and
    converts as one; that it never grows is left to the macro.
    """

    def __repr__(self) -> str:
        return "<std.array template>"

    def __getitem__(self, arguments: tuple[Any, Any]) -> Any:
        kind, size = arguments
        vector = TEMPLATES["vector"][kind]

        def made(values: Iterable[Any] = ()) -> Any:
            filled = vector(int(size))
            for index, value in enumerate(values):
                filled[index] = value
            return filled

        return made


class _Stream:
    """``std::cout`` and ``std::cerr``: what ``<<`` is given, written as text."""

    def __init__(self, name: str) -> None:
        self._name = name

    def __lshift__(self, value: Any) -> _Stream:
        import sys

        target = sys.stdout if self._name == "cout" else sys.stderr
        target.write(str(value))
        return self

    def flush(self) -> None:
        """``flush``: every write goes out as it is made."""

    def __repr__(self) -> str:
        return f"<std::{self._name}>"


class _Namespace:
    """``ROOT.std``: the containers, the streams, and a refusal by name for the rest of it."""

    cout = _Stream("cout")
    cerr = _Stream("cerr")
    endl = "\n"
    vector = TEMPLATES["vector"]
    map = TEMPLATES["map"]
    pair = TEMPLATES["pair"]
    array = _ArrayTemplate()
    mt19937 = _random.mt19937
    normal_distribution = _random.normal_distribution
    uniform_real_distribution = _random.uniform_real_distribution
    string = string

    def __repr__(self) -> str:
        return "<namespace std>"

    @staticmethod
    def move(value: Any) -> Any:
        """``std::move``: the object itself, which handing it on gives away in Python too."""
        return value

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"ROOT has std.{name}; xrdroot.pyroot does not yet")


#: ``ROOT.std``.
std = _Namespace()


def is_vector(value: Any) -> bool:
    """Is this one of the ``std::vector`` classes here?"""
    return isinstance(value, (_Vector, _ObjectVector))
