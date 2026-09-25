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
"""

from __future__ import annotations

from typing import Any

__all__ = ["Cell", "ItemRef", "AttrRef"]


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
