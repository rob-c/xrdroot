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

from typing import Any

__all__ = ["Cell", "Static", "ItemRef", "AttrRef", "address_of", "pointer_at"]


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
    """``&a[i]``: element ``key`` of ``container``, read and written as ``.value``."""

    __slots__ = ("container", "key")

    def __init__(self, container: Any, key: Any) -> None:
        self.container = container
        self.key = key

    @property
    def value(self) -> Any:
        return self.container[self.key]

    @value.setter
    def value(self, value: Any) -> None:
        self.container[self.key] = value

    def __getitem__(self, index: int) -> Any:
        return self.container[self.key + index]

    def __setitem__(self, index: int, value: Any) -> None:
        self.container[self.key + index] = value


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
