"""Addresses across the Python boundary: what ``&x`` becomes when ``x`` is a number.

Python cannot hand out the address of a local ``double``, so a variable
whose address a macro takes lives in a :class:`Cell` instead, and every use
of it reads or writes ``.value``. ROOT's functions that write through a
pointer or reference - ``TTree::SetBranchAddress``, ``TRandom::Rannor``,
``TGraph::GetPoint`` - take anything with a read/write ``.value``, which is
the contract :mod:`xrdroot.pyroot` keeps. A cell also answers ``p[0]``, so a
function translated from ``void f(double *out) { *out = 1; }`` works whether
it is handed a cell or an array.

:class:`ItemRef` and :class:`AttrRef` are the same promise made about an
element of a container, ``&a[i]``, or a member of an object, ``&obj.x``.

An object's address as a number - printed with ``%p``, or cast to an integer
- is its ``id``, and the object is remembered under it, so that a pointer cast
back from that number, ``(TFile*)0x...`` in a line of C++ a program wrote out
for an interpreter to run, is the very object (:func:`pointer_at`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

__all__ = ["Cell", "Static", "ItemRef", "AttrRef", "address_of", "pointer_at", "integer_cast",
           "byte_offset", "moved_pointer", "pointer_now"]


class Cell:
    """A number (or string) that can be handed out by address: read and write ``.value``.

    ``ctype`` is the C++ type the macro declared it with - ``"float"``,
    ``"int"``, ``"double"`` - so that whoever writes into it (a branch being
    read, say) knows the width the macro meant.
    """

    __slots__ = ("value", "ctype")

    def __init__(self, value: Any = 0, ctype: str | None = None) -> None:
        self.value = value
        self.ctype = ctype

    def __repr__(self) -> str:
        return f"Cell({self.value!r})"

    def __getitem__(self, index: int) -> Any:
        _only_zero(index)
        return self.value

    def __setitem__(self, index: int, value: Any) -> None:
        _only_zero(index)
        self.value = value


class Static(Cell):
    """A static local: one cell for the whole run, and whether it has been initialised yet.

    C++ initialises a static local the first time its declaration is
    reached; the translation tests ``ready`` there, and sets it once the
    value is in - so an initialiser that throws is tried again next time,
    as C++ tries it.
    """

    __slots__ = ("ready",)

    def __init__(self, value: Any = 0, ctype: str | None = None, ready: bool = True) -> None:
        super().__init__(value, ctype)
        self.ready = ready


def _only_zero(index: Any) -> None:
    if index != 0:
        raise IndexError(f"a pointer to one variable has only element 0, not element {index}")


class ItemRef:
    """``&a[i]``: a pointer at element ``key`` of ``container``, read and written as ``.value``.

    C++ takes the address of an element because the element is where a pointer
    points, so this is a pointer rather than that one element: ``p[n]``, ``*p``
    and a range ``p[0:n]`` an algorithm sorts or copies into all reach into the
    container itself, and a call that wants an array - ``TGraph(n, &x[0], ...)``
    - converts this to the ``len(container) - key`` elements from ``key`` on.
    """

    __slots__ = ("container", "key")

    def __init__(self, container: Any, key: Any) -> None:
        self.container = container
        self.key = key

    def __repr__(self) -> str:
        return f"ItemRef({self.container!r}, {self.key!r})"

    @property
    def value(self) -> Any:
        return self.container[self.key]

    @value.setter
    def value(self, value: Any) -> None:
        self.container[self.key] = value

    def __len__(self) -> int:
        return len(self.container) - int(self.key)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.container[self.key:])

    def _shifted(self, index: Any) -> Any:
        """``index`` as the container sees it: a pointer's offset moved into the container.

        The slice that starts nowhere starts at the pointer, since a pointer has no
        elements before itself: ``p[0:n]`` is the ``n`` elements from ``p`` onward.
        """
        if not isinstance(index, slice):
            return self.key + index
        start = self.key if index.start is None else self.key + index.start
        stop = None if index.stop is None else self.key + index.stop
        return slice(start, stop, index.step)

    def __getitem__(self, index: Any) -> Any:
        return self.container[self._shifted(index)]

    def __setitem__(self, index: Any, value: Any) -> None:
        self.container[self._shifted(index)] = value


class AttrRef:
    """``&obj.x``: the attribute ``name`` of ``obj``, read and written as ``.value``."""

    __slots__ = ("obj", "name")

    def __init__(self, obj: Any, name: str) -> None:
        self.obj = obj
        self.name = name

    @property
    def value(self) -> Any:
        return getattr(self.obj, self.name)

    @value.setter
    def value(self, value: Any) -> None:
        setattr(self.obj, self.name, value)

    def __getitem__(self, index: int) -> Any:
        _only_zero(index)
        return self.value

    def __setitem__(self, index: int, value: Any) -> None:
        _only_zero(index)
        self.value = value


#: The objects whose addresses a program has seen as numbers, by address: kept alive, so
#: that no other object is given the same address while the number may be cast back.
SEEN: dict[int, Any] = {}


def address_of(obj: Any) -> int:
    """A pointer as the number C++ casts it to: ``0`` for null, else the object's address."""
    if obj is None:
        return 0
    SEEN[id(obj)] = obj
    return id(obj)


def pointer_at(address: Any) -> Any:
    """``(T*)n``: the object at the address ``n`` - one :func:`address_of` gave - or null."""
    number = int(address)
    if number == 0:
        return None
    if number not in SEEN:
        raise ValueError(
            f"0x{number:x} is no address of an object this program turned into a number, "
            "and only such an address can be made a pointer again here"
        )
    return SEEN[number]


#: What makes a value a number to C's cast: a library's numbers, a class converting to one.
NUMBERS = ("__index__", "__float__")


def integer_cast(value: Any) -> int:
    """``(Long_t)x`` of a value whose type is not known: a number truncated as C casts it, a
    character its code, null ``0`` - and an object, a pointer to it, its address."""
    if value is None:
        return 0
    if isinstance(value, str) and len(value) == 1:
        return ord(value)
    if isinstance(value, (int, float)) or any(hasattr(value, name) for name in NUMBERS):
        return int(value)
    return address_of(value)


def byte_offset(obj: Any, offset: Any) -> Any:
    """``(char*)obj + offset``: the object itself when the offset is none - a base class at
    the start of the object, as single inheritance puts it; any other offset is the middle
    of an object, which here has no address of its own."""
    if int(offset) == 0:
        return obj
    raise ValueError(
        f"an object's address moved by {int(offset)} bytes, into the middle of it, which "
        "names nothing here: only an offset of 0 - a base at the object's start - is followed"
    )


def moved_pointer(value: Any, offset: Any) -> Any:
    """``p + n`` of a ``char *``: the text from there on, or an object moved by its offset.

    A ``char *`` a macro holds may be a C string - ``slash + 1`` is the path after it - or an
    object's address (``(char *)obj``), which C++ moves through the object's members. The two
    are one type, so the move is decided by what is moved.
    """
    if isinstance(value, str):
        return value[int(offset):]
    return byte_offset(value, offset)


def pointer_now(value: Any) -> Any:
    """A library's global pointer stored into a variable: what it points at now. A global
    that follows what is current - ROOT's ``gDirectory`` - says what that is by
    ``__real__``; anything else is itself."""
    real = getattr(type(value), "__real__", None)
    return real(value) if callable(real) else value
