"""Addresses: what ``Branch("x", &x)`` and ``SetBranchAddress("x", &x)`` are given.

C++ hands a tree a pointer, and the tree reads through it when ``Fill`` is
called and writes through it when ``GetEntry`` is. Python has no pointers,
so a tree here is handed something it can read and change in place: a NumPy
array, an :mod:`array` array, a :mod:`ctypes` number or array, a
``std.vector`` or ``std.string`` from :mod:`xrdroot.pyroot.stl`, or any
object with a ``.value`` it can read and set - which is what a C++ macro's
``Float_t x;`` is translated to. What the tree keeps is an :class:`Address`,
and every kind of thing it was given is read and written the same way
through it.

A NumPy array, an :mod:`array` array and a :mod:`ctypes` object are all
turned into a NumPy view of the memory they own, so a value put there by
``GetEntry`` is in the caller's object with no copying back, as C++'s would
be. Something that is none of these is refused by name, because a tree that
quietly read a copy would fill every entry with the first one's values.
"""

from __future__ import annotations

import array
import ctypes
from typing import Any

import numpy as np

from ..stl import string
from ..stl import is_vector as _is_vector

__all__ = ["Address", "address_of", "members_of"]


def _refused(given: Any, what: str) -> TypeError:
    return TypeError(
        f"{what} was given a {type(given).__name__}, and a tree reads and fills only what "
        f"it can change in place: a NumPy array, an array.array, a ctypes number or array, "
        f"a std.vector or std.string, or an object with a .value to read and set"
    )


class Address:
    """Where one leaf's values are: read by ``Fill``, written by ``GetEntry``.

    ``get`` gives a number when ``count`` is ``None`` and the first
    ``count`` values as an array when it is not; ``put`` puts either back.
    ``dtype`` is the type the memory holds, when it says, which is how a
    branch declared without a leaf list learns its type; ``room`` is how many
    values it can hold, which ``None`` says is any number.
    """

    dtype: np.dtype[Any] | None = None
    room: int | None = 1
    #: Does this hold a collection whose length it keeps itself - a vector?
    sized = False
    #: Does this hold text rather than numbers?
    text = False

    def get(self, count: int | None = None) -> Any:  # pragma: no cover - every kind has one
        raise NotImplementedError

    def put(self, value: Any) -> None:  # pragma: no cover - every kind has one
        raise NotImplementedError


class View(Address):
    """Memory NumPy can look at: an array, an ``array.array``, a ``ctypes`` object."""

    def __init__(self, view: np.ndarray[Any, Any], start: int = 0) -> None:
        self.view = view
        self.start = start
        self.dtype = view.dtype
        self.room = len(view) - start

    def get(self, count: int | None = None) -> Any:
        if count is None:
            return self.view[self.start].item()
        return self.view[self.start : self.start + count].copy()

    def put(self, value: Any) -> None:
        if np.ndim(value) == 0:
            self.view[self.start] = value
            return
        given = np.asarray(value)
        if len(given) > (self.room or 0):
            raise ValueError(
                f"this entry holds {len(given)} values and the array it is read into has "
                f"room for {self.room}; give the address room for the longest entry"
            )
        self.view[self.start : self.start + len(given)] = given


class Value(Address):
    """Anything with a ``.value``: a translated macro's variable, or a ``ctypes`` stand-in."""

    room = None

    def __init__(self, holder: Any) -> None:
        self.holder = holder
        current = holder.value
        self.text = isinstance(current, (str, bytes, string))
        declared = getattr(holder, "dtype", None)
        self.dtype = _dtype_of(current) if declared is None else np.dtype(declared)

    def get(self, count: int | None = None) -> Any:
        current = self.holder.value
        if count is None:
            return str(current) if self.text else current
        return np.asarray(current)[:count].copy()

    def put(self, value: Any) -> None:
        current = self.holder.value
        if np.ndim(value) and hasattr(current, "__setitem__") and not isinstance(current, str):
            current[: len(value)] = value  # an array held in the variable, filled in place
            return
        self.holder.value = value


class Vector(Address):
    """A ``std.vector``: read whole, and made to hold the entry's values."""

    room = None
    sized = True

    def __init__(self, vector: Any) -> None:
        self.vector = vector
        self.dtype = vector.dtype if vector.dtype != np.dtype(object) else None
        self.text = vector.value_type == "string"

    def get(self, count: int | None = None) -> Any:
        data = self.vector.data()
        return list(map(str, data)) if self.text else np.array(data)

    def put(self, value: Any) -> None:
        self.vector.assign(value)


class Text(Address):
    """A ``std.string``, or a ``bytearray`` standing for a ``char[]``."""

    room = None
    text = True

    def __init__(self, holder: Any) -> None:
        self.holder = holder

    def get(self, count: int | None = None) -> str:
        if isinstance(self.holder, bytearray):
            return bytes(self.holder).split(b"\0", 1)[0].decode()
        return str(self.holder)

    def put(self, value: Any) -> None:
        if isinstance(self.holder, bytearray):
            raw = str(value).encode() + b"\0"
            self.holder[: len(raw)] = raw
            return
        self.holder.assign(value)


class Member(Address):
    """One member of an object a multi-leaf branch was given: ``event.a``."""

    def __init__(self, holder: Any, name: str) -> None:
        self.holder = holder
        self.name = name
        self.dtype = _dtype_of(getattr(holder, name))

    def get(self, count: int | None = None) -> Any:
        current = getattr(self.holder, self.name)
        return current if count is None else np.asarray(current)[:count].copy()

    def put(self, value: Any) -> None:
        setattr(self.holder, self.name, value)


def _dtype_of(value: Any) -> np.dtype[Any] | None:
    """The type a plain Python value stands for: ``bool``, ``Int_t`` or ``Double_t``."""
    if isinstance(value, bool):
        return np.dtype("bool")
    if isinstance(value, int):
        return np.dtype("int32")
    if isinstance(value, float):
        return np.dtype("float64")
    if isinstance(value, np.ndarray):
        return value.dtype
    return None


def _numpy_view(given: np.ndarray[Any, Any], what: str) -> np.ndarray[Any, Any]:
    if not given.flags.c_contiguous or not given.flags.writeable:
        raise ValueError(
            f"{what} was given an array that is not one contiguous, writable block, so "
            f"what the tree put in it would not be where the caller looks; give it "
            f"np.ascontiguousarray of it, kept in a name"
        )
    return given.reshape(-1)


def _ctypes_view(given: Any) -> np.ndarray[Any, Any]:
    if isinstance(given, ctypes.Array):
        return np.ctypeslib.as_array(given).reshape(-1)
    return np.frombuffer(given, dtype=np.dtype(type(given))).reshape(-1)


def address_of(given: Any, what: str = "the branch") -> Address:
    """What a tree keeps for the thing it was handed, or a refusal naming it."""
    if isinstance(given, Address):
        return given
    if isinstance(given, np.ndarray):
        return View(_numpy_view(given, what))
    if isinstance(given, array.array):
        return View(np.frombuffer(given, dtype=np.dtype(given.typecode)))
    if isinstance(given, (ctypes._SimpleCData, ctypes.Array)):
        return View(_ctypes_view(given))
    if _is_vector(given):
        return Vector(given)
    if isinstance(given, (string, bytearray)):
        return Text(given)
    if hasattr(given, "value"):
        return Value(given)
    raise _refused(given, what)


def members_of(given: Any, names: list[str], sizes: list[int], what: str) -> list[Address]:
    """An address for each leaf of a leaf list, all given one object between them.

    ``Branch("p", &p, "px/F:py/F:pz/F")`` hands the tree one struct and its
    leaves are its members, one after another. Here that is a NumPy
    structured array or a :mod:`ctypes` structure, whose fields are taken by
    the leaves' names; an array of numbers, laid out one leaf after another
    as the struct's members were; or any object whose attributes are named as
    the leaves are.
    """
    if isinstance(given, np.ndarray) and given.dtype.names:
        return [View(_numpy_view(given[name], what)) for name in names]
    if isinstance(given, (np.ndarray, array.array, ctypes.Array)):
        whole = address_of(given, what)
        assert isinstance(whole, View)
        starts = np.cumsum([0, *sizes[:-1]])
        return [View(whole.view, int(start)) for start in starts]
    missing = [name for name in names if not hasattr(given, name)]
    if missing:
        raise _refused(given, what) if len(missing) == len(names) else AttributeError(
            f"{what} has leaves {', '.join(names)}, and the {type(given).__name__} it was "
            f"given has no {', '.join(missing)}"
        )
    return [_member(given, name, what) for name in names]


def _member(given: Any, name: str, what: str) -> Address:
    """A struct's member: a view if it is memory, or the attribute itself if it is a number."""
    value = getattr(given, name)
    if isinstance(value, (np.ndarray, array.array, ctypes.Array)) or _is_vector(value):
        return address_of(value, what)
    if isinstance(given, ctypes.Structure):
        return Member(given, name)
    return address_of(value, what) if hasattr(value, "value") else Member(given, name)
