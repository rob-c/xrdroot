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


def rebuilt(classname: str, value: Any, current: Any) -> Any:
    """The object ``value`` - one entry's members - stands for, ``current`` filled if it can be."""
    if classname.startswith("vector<") and "PxPyPzE4D" in classname:
        return _four_vectors(value, current)
    if classname == "TLorentzVector":
        return _lorentz(value, current)
    if classname == "TClonesArray":
        return _clones(value, current)
    from ...kinds import dress
    from ..core.wrapping import wrap

    return wrap(dress(classname, value))


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


def _clones(value: Any, current: Any) -> Any:
    """A ``TClonesArray``: each object the entry held, made again from its members."""
    from ..core.collections import TClonesArray
    from ..core.wrapping import from_members

    made = current if isinstance(current, TClonesArray) else TClonesArray(value.classname)
    made.Clear()
    for at, members in enumerate(value):
        flat = _flat(members)
        obj = from_members(value.classname, flat)
        made.AddAt(_drawn(value.classname, flat) if obj is None else obj, at)
    return made


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
