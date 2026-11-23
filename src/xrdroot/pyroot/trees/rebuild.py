"""An object made again from what one entry of a branch of objects holds.

The reader hands back a whole object as its members - a dictionary, by
the names its class's streamer gives them - and a split collection as a
column of each member. :func:`rebuilt` makes the object a macro's pointer
points at from that: the histogram, the ``TLorentzVector``, the
``std::vector`` of four-vectors or the ``TClonesArray``, reusing the one
already there where ROOT would fill it in place.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["rebuilt"]

#: GenVector's Cartesian coordinates, in the order ``SetCoordinates`` takes them.
COORDINATES = ("fX", "fY", "fZ", "fT")


def rebuilt(classname: str, value: Any, current: Any, clones: str = "") -> Any:
    """The object ``value`` - one entry's members - stands for, ``current`` filled if it can be;
    ``clones`` is the class a ``TClonesArray`` branch says its objects are."""
    if classname.startswith("vector<") and "PxPyPzE4D" in classname:
        return _four_vectors(value, current)
    if classname == "TLorentzVector":
        return _lorentz(value, current)
    if classname == "TClonesArray":
        return _clones(value, current, clones)
    declared = _declared(classname, value, current)
    if declared is not None:
        return declared
    from ...kinds import dress
    from ..core.wrapping import wrap

    return wrap(dress(classname, value))


def _declared(classname: str, value: Any, current: Any) -> Any:
    """An object of a class C++ declared, made again from one entry's members - into the one
    the pointer points at already, as ROOT reads into it - or ``None`` for any other class."""
    from ...wdeclared import parse_type
    from ..core.declared import converted, find_class

    cls = find_class(classname)
    if cls is None:
        return None
    return converted(value, parse_type(classname), current if isinstance(current, cls) else None)


def _four_vectors(value: dict[str, Any], current: Any) -> Any:
    """A ``std::vector<XYZTVector>``, one vector per row of each coordinate's column."""
    from ..core.genvector import XYZTVector
    from ..stl import std

    made = current if hasattr(current, "push_back") else std.vector["ROOT::Math::XYZTVector"]()
    made.clear()
    columns = [np.asarray(value[f"fCoordinates.{name}"]) for name in COORDINATES]
    for row in zip(*columns, strict=True):
        made.push_back(XYZTVector(*(float(x) for x in row)))
    return made


def _lorentz(value: dict[str, Any], current: Any) -> Any:
    from ..core.vectors import TLorentzVector

    made = current if isinstance(current, TLorentzVector) else TLorentzVector()
    p = value["fP"]
    made.SetPxPyPzE(p["fX"], p["fY"], p["fZ"], value["fE"])
    return made


def _clones(value: Any, current: Any, clones: str = "") -> Any:
    """A ``TClonesArray``: each object the entry held, made again from its members - which
    a split one's branches hand back as a column of each member, a row per object.

    An element of a class the session declared is made by that declaration, which is what
    fills the ``TNamed`` part of ``roottest``'s ``tref_test_pid``: no branch under the
    collection names a base of the class, so a member set one at a time leaves every element
    of the array empty. A class ROOT or the kit describes instead is made by name, as before.
    """
    from ..core.collections import TClonesArray
    from ..core.references import filed_object
    from ..core.wrapping import from_members

    classname = str(getattr(value, "classname", None) or getattr(current, "_class", clones))
    made = current if isinstance(current, TClonesArray) else TClonesArray(classname)
    made.Clear()
    for at, members in enumerate(_objects(value)):
        obj = _member(classname, members)
        if obj is None:
            flat = _flat(members)
            obj = from_members(classname, flat)
            if obj is None:
                obj = _drawn(classname, flat)
        filed_object(obj)
        made.AddAt(obj, at)
    return made


def _member(classname: str, members: Any) -> Any:
    """One object of a class C++ declared in this session, from the members read for it - or
    ``None`` when no class of that name was declared, and a part of the kit makes it instead.

    The branches under a split object name a member by its own name alone, never by the base
    that declares it, so the row a class's reader takes - a base's members under the base's
    name - is put together here from the declaration, which is all ROOT's dictionary knows
    of a class too.
    """
    from ..core.declared import made_from

    cls = _described(classname)
    if cls is None or not isinstance(members, dict):
        return None
    return made_from(classname, _by_base(cls, members))


def _described(classname: str) -> Any:
    """The class ``classname`` if C++ declared it - if its translation says its members."""
    from ..core.declared import find_class

    cls = find_class(classname)
    return cls if getattr(cls, "__dict__", {}).get("_cxx_layout_") is not None else None


def _by_base(cls: Any, flat: dict[str, Any]) -> dict[str, Any]:
    """One object's members grouped under the base each belongs to, as a class's reader takes
    them: the members ``cls`` declares itself, and under each of its bases - declared, or one
    of the namespace's own - the members of that base.
    """
    _name, bases, fields = cls.__dict__["_cxx_layout_"]
    row = {str(key): flat[str(key)] for key, *_ in fields if str(key) in flat}
    rest = {key: value for key, value in flat.items() if not _declares(cls, key)}
    for base in bases:
        owner = _described(str(base))
        row[str(base)] = _by_base(owner, flat) if owner is not None else dict(rest)
    return row


def _declares(cls: Any, key: str) -> bool:
    """Does ``cls`` or a base of it name ``key`` among the members it declares?"""
    _name, bases, fields = cls.__dict__["_cxx_layout_"]
    if any(str(name) == key for name, *_ in fields):
        return True
    return any(
        owner is not None and _declares(owner, key)
        for owner in map(_described, (str(base) for base in bases))
    )


def _objects(value: Any) -> list[Any]:
    """The objects of an entry: as they are, or, from a column per member, a row each."""
    if not isinstance(value, dict):
        return list(value)
    columns = {name: list(np.asarray(column).tolist()) for name, column in value.items()}
    count = len(next(iter(columns.values()), []))
    return [{name: column[at] for name, column in columns.items()} for at in range(count)]


def _drawn(classname: str, members: dict[str, Any]) -> Any:
    """A drawing object - a ``TLine``, a ``TMarker`` - made and given the members it keeps."""
    import importlib

    obj = getattr(importlib.import_module("xrdroot.pyroot"), classname)()
    kept = getattr(obj, "members", {})
    kept.update({name: value for name, value in members.items() if name in kept})
    return obj


def _flat(members: dict[str, Any]) -> dict[str, Any]:
    """An object's members with its bases' members beside its own, as ROOT names them."""
    flat: dict[str, Any] = {}
    for name, value in members.items():
        if isinstance(value, dict):
            flat.update(_flat(value))
        else:
            flat[name] = value
    return flat
