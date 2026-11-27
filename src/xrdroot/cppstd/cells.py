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
           "byte_offset", "ObjectAddress", "pointer_now", "load", "store"]


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

    def __eq__(self, other: object) -> bool:
        """A pointer at an element is the element when it is asked what it holds.

        C++ compares a pointer with a pointer; a program here compares what a runtime call answers
        with the value a reader expects - ``proxy.At(0) == 4`` - so an item reference answers for
        the value it points at, and a pair compares as the two it holds.
        """
        value = self.value
        if value == other:
            return True
        from .objects import Pair

        return isinstance(value, Pair) and tuple(value) == other

    __hash__ = None  # type: ignore[assignment]

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
    """``(char*)obj + offset``: the object's bytes at that offset, or the object itself.

    An address already stepped - ``(char *)&obj.member + off`` - is stepped again from the member:
    into the member's own class when it has one (``&obj.m_d.m_b``), or to the member at the sum
    (``(char *)obj + 32 + 8`` when a member of the object's class starts at 40).

    An offset a member of the object starts at is that member's bytes, which
    :class:`ObjectBytes` reads and writes - a macro fills a class that way. Anything else that is
    an offset of 0 is the object: a base class at the start of it, as single inheritance puts it,
    which ``TClass::GetBaseClassOffset`` answers 0 for. Any other object moved into its middle has
    no address here.
    """
    if isinstance(obj, ObjectAddress):
        # ``(char *)obj`` of an object is that object's address: stepping it steps the object
        return byte_offset(obj._object, offset)
    if isinstance(obj, ObjectBytes):
        return _further(obj, int(offset))
    started = _member_starts(type(obj))
    if _member_at(type(obj), int(offset)) is not None:
        return ObjectBytes(obj, int(offset))
    if int(offset) == 0:
        return obj
    raise ValueError(
        f"an object of {type(obj).__name__} moved by {int(offset)} bytes, into the middle of it, "
        f"which names nothing here: its members start at {sorted((started or {}).values())}, and "
        "only an offset of 0 - a base at the object's start - is followed without one"
    )


def _further(stepped: ObjectBytes, offset: int) -> Any:
    """``(char *)&obj.member + offset``: the member's own bytes at that offset when it is an object
    of a class laid out here, and otherwise the object's bytes at the two offsets added."""
    inner = getattr(stepped._object, stepped._member, None)
    started = getattr(type(inner), "_cxx_offsets_", None) if inner is not None else None
    if started is not None and offset in started.values():
        return ObjectBytes(inner, offset)
    return byte_offset(stepped._object, stepped._offset + offset)


def _member_starts(cls: type) -> dict[str, int] | None:
    """Where each member of a class a macro declared starts, by name, or ``None`` for any other
    class - and for one whose layout the translation did not work out.

    The numbers are the translation's own (``_cxx_offsets_``, from
    :func:`xrdroot.pycxx.sizes.member_offsets`) - the very ones ``TDataMember::GetOffset`` answers,
    a base's members among them - so a macro is never told one offset and handed another. A class
    without them is not laid out here rather than laid out a second time, differently.
    """
    found = getattr(cls, "_cxx_offsets_", None)
    return dict(found) if found is not None else None


def _member_at(cls: type, offset: int) -> str | None:
    """The member of ``cls`` - or of a base it starts with - that starts at ``offset``."""
    started = _member_starts(cls)
    if started is None:
        return None
    for name, at in started.items():
        if at == offset:
            return name
    return None


class ObjectBytes:
    """``(char *)obj + off``: the object's bytes there, named by the member that starts at ``off``.

    A macro writes into an object through its own bytes - ``*(int *)(((char *)obj) +
    member->GetOffset()) = value`` is how roottest's ``datamodelevolution`` fills a class - and
    Python has no address to hand out. So this answers what such a program does with the address:
    ``p[0]`` is the member starting at ``off``, read or written, and every other name is the
    object's own, because ``(T *)((char *)obj + 0)`` cast back is the object itself.
    """

    __slots__ = ("_object", "_offset", "_member")

    def __init__(self, obj: Any, offset: int) -> None:
        member = _member_at(type(obj), offset)
        if member is None:
            raise ValueError(
                f"{offset} is not the start of a member of {type(obj).__name__}, whose members "
                f"start at {sorted((_member_starts(type(obj)) or {}).values())}"
            )
        self._object = obj
        self._offset = offset
        self._member = member

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice) or index != 0:
            raise TypeError("an object's bytes are read at the member that starts there, as p[0]")
        return getattr(self._object, self._member)

    def __setitem__(self, index: Any, value: Any) -> None:
        if isinstance(index, slice) or index != 0:
            raise TypeError("an object's bytes are written at the member that starts there, as "
                            "p[0] = value")
        setattr(self._object, self._member, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._object, name)

    def __repr__(self) -> str:
        return f"ObjectBytes({self._object!r}, {self._offset})"


class ObjectAddress:
    """``(char *)obj``: an object's address, which indexing moves through its own bytes.

    C++ casts an object to ``char *`` to move it by a base class's offset and cast it back -
    ``(T *)((char *)obj + off)``, which is how ``TDirectory::CloneObject`` makes a copy. Python
    has no address to hand out, so this holds the object and answers the one operation a program
    performs on the address, ``p[off:]``: the object moved by that offset, which is what
    :func:`byte_offset` follows (an offset of 0 is the object itself).
    """

    __slots__ = ("_object",)

    def __init__(self, obj: Any) -> None:
        self._object = obj

    def __getitem__(self, index: Any) -> Any:
        offset = index.start if isinstance(index, slice) else index
        return byte_offset(self._object, 0 if offset is None else offset)

    def __setitem__(self, index: Any, value: Any) -> None:
        raise TypeError("an object's address moved by an offset names the object itself, and a "
                        "whole object is not written through its address; assign its members")

    def __repr__(self) -> str:
        return f"ObjectAddress({self._object!r})"


def pointer_now(value: Any) -> Any:
    """A library's global pointer stored into a variable: what it points at now. A global
    that follows what is current - ROOT's ``gDirectory`` - says what that is by
    ``__real__``; anything else is itself."""
    real = getattr(type(value), "__real__", None)
    return real(value) if callable(real) else value


def store(target: Any, value: Any) -> None:
    """Put ``value`` where ``target`` keeps it, as ``*target = value`` writes through a pointer
    or a reference: an array's first element, or anything's ``.value`` - which is what a
    translated local, a NumPy array of one and a ``Cell`` have in common."""
    if target is None:
        return
    if hasattr(target, "value"):
        target.value = value
        return
    if not hasattr(target, "__setitem__"):
        raise TypeError(
            f"a {type(target).__name__} is not somewhere to put a number: hand over a NumPy "
            f"array, an array.array, a ctypes number or anything with a .value"
        )
    target[0] = value


def load(target: Any) -> Any:
    """What ``target`` holds now: ``*target``."""
    return target.value if hasattr(target, "value") else target[0]
