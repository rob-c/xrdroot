"""``SetBranchAddress`` of a branch of objects: what each entry is read into.

``Vector3 *v = 0; T->SetBranchAddress("v3", &v)`` hands the tree a pointer
it is to make an object for and fill at every ``GetEntry``. A split object
is read a member at a time - each branch under it into its member of the one
object, so ``T->GetBranch("fY")->GetEntry(i)`` fills ``v->fY`` alone, as
ROOT's does - and a whole object, a histogram or a collection is made again
from what the entry holds.
"""

from __future__ import annotations

from typing import Any

from .addresses import Address

__all__ = ["bind_object", "is_object"]

#: GenVector's Cartesian four-vector, by the name its branches are written under.
LORENTZ = "ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >"
#: Where each of its coordinates is among ``(x, y, z, t)``.
COORDINATES = {"fX": 0, "fY": 1, "fZ": 2, "fT": 3}


def is_object(branch: Any) -> bool:
    """Is this a branch of objects - not of numbers, nor of a ``std::vector`` of them?"""
    if not branch.classname or not branch.leaves:
        return bool(branch.children)
    return not branch.leaves[0].vector or bool(branch.children)


def _class(classname: str) -> Any:
    """What makes an object of ``classname``: a macro's class, or ROOT's."""
    import importlib

    from ...cint.runtime.root import DECLARED

    if classname in DECLARED:
        return DECLARED[classname]
    if classname == LORENTZ:
        return importlib.import_module("xrdroot.pyroot").XYZTVector
    return getattr(importlib.import_module("xrdroot.pyroot"), classname)


def _object(address: Any, classname: str) -> Any:
    """The object a pointer points at, made of ``classname`` if it points at nothing yet."""
    if not hasattr(address, "value"):
        return address
    held = address.value
    if held is None or (isinstance(held, int) and held == 0):
        held = address.value = _class(classname)()
    return held


class _Member(Address):
    """One member of the object a split branch is read into."""

    def __init__(self, obj: Any, name: str) -> None:
        from ...cint.symbols import member_name

        self.obj, self.name = obj, member_name(name)

    def get(self, count: int | None = None) -> Any:
        return getattr(self.obj, self.name)

    def put(self, value: Any) -> None:
        setattr(self.obj, self.name, value)


class _Coordinate(Address):
    """One Cartesian coordinate of a GenVector four-vector read into."""

    def __init__(self, obj: Any, at: int) -> None:
        self.obj, self.at = obj, at

    def get(self, count: int | None = None) -> Any:
        return self.obj._cartesian()[self.at]

    def put(self, value: Any) -> None:
        now = list(self.obj._cartesian())
        now[self.at] = float(value)
        self.obj._set_cartesian(now)


def _member(obj: Any, column: str) -> Address:
    name = column.rpartition(".")[2]
    if getattr(obj, "SYSTEM", None) is not None and name in COORDINATES:
        return _Coordinate(obj, COORDINATES[name])
    return _Member(obj, name)


class _Whole(Address):
    """A whole object, made again from each entry's members and put where the pointer is."""

    def __init__(self, address: Any, classname: str) -> None:
        self.address, self.classname = address, classname

    def get(self, count: int | None = None) -> Any:
        return getattr(self.address, "value", self.address)

    def put(self, value: Any) -> None:
        from .rebuild import rebuilt

        made = rebuilt(self.classname, value, self.get())
        if hasattr(self.address, "value"):
            self.address.value = made


def bind_object(branch: Any, address: Any) -> dict[str, Address]:
    """What each column of a branch of objects is read into, given the pointer to read into."""
    if branch.fid == -2 and branch.children:  # split: a member at a time, into one object
        obj = _object(address, branch.classname)
        leaves = [leaf for child in branch.children for b in child.walk() for leaf in b.leaves]
        return {leaf.column: _member(obj, leaf.column) for leaf in leaves}
    return {branch.leaves[0].column: _Whole(address, branch.classname)}
