"""A ``TGeometry`` read from a file, made one of :mod:`.legacy`'s again.

The reader hands back a geometry as its members: lists of rotation
matrices and shapes, and the tree of nodes, each node's shape the very
record in the list of shapes. Each shape's class is known by the members
it has - ``fDx, fDy, fDz`` a ``TBRIK``, ``fH1`` and the rest a ``TTRAP``
- and is made again with those numbers; each node is made again in its
mother, with its place, its matrix, its look, its visibility and whether
its sons are hidden. The materials ROOT writes in a form this reader does
not decode are left out: nothing drawn needs them.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.wrapping import register_members
from . import legacyshapes as old
from .legacy import TGeometry, TNode, TRotMatrix

__all__ = ["geometry_from"]

#: Each shape class and the members its constructor's numbers are kept in, most specific
#: first: a shape is the first whose members it has all of. A ``TTRAP`` keeps its theta and
#: phi in the ``fDx`` and ``fDy`` of the ``TBRIK`` it derives from, as ROOT does.
_TRAP = ("fH1", "fBl1", "fTl1", "fAlpha1", "fH2", "fBl2", "fTl2", "fAlpha2")
SHAPES: tuple[tuple[type, tuple[str, ...]], ...] = (
    (old.TGTRA, ("fDz", "fDx", "fDy", "fTwist", *_TRAP)),
    (old.TTRAP, ("fDz", "fDx", "fDy", *_TRAP)),
    (old.TTRD2, ("fDx", "fDx2", "fDy", "fDy2", "fDz")),
    (old.TTRD1, ("fDx", "fDx2", "fDy", "fDz")),
    (old.TSPHE, ("fRmin", "fRmax", "fThemin", "fThemax", "fPhimin", "fPhimax")),
    (old.TCONS, ("fDz", "fRmin", "fRmax", "fRmin2", "fRmax2", "fPhi1", "fPhi2")),
    (old.TCONE, ("fDz", "fRmin", "fRmax", "fRmin2", "fRmax2")),
    (old.TTUBS, ("fRmin", "fRmax", "fDz", "fPhi1", "fPhi2")),
    (old.TTUBE, ("fRmin", "fRmax", "fDz", "fAspectRatio")),
    (old.TBRIK, ("fDx", "fDy", "fDz")),
)


def _flat(members: dict[str, Any]) -> dict[str, Any]:
    """Every member by name, a class's own before its bases'."""
    found: dict[str, Any] = {}
    for name, value in members.items():
        if isinstance(value, dict) and name[:1] == "T":
            for inner, held in _flat(value).items():
                found.setdefault(inner, held)
        else:
            found.setdefault(name, value)
    return found


def _shape_class(flat: dict[str, Any]) -> tuple[type, tuple[str, ...]]:
    found = next(((kind, names) for kind, names in SHAPES if all(n in flat for n in names)), None)
    if found is None:
        raise UnsupportedFeatureError(
            f"a shape of the old geometry package with members {sorted(flat)} is not one this "
            f"reader makes again; it knows TBRIK, TTRD1, TTRD2, TTRAP, TGTRA, TSPHE and the tubes"
        )
    return found


def _shape(record: dict[str, Any], made: dict[int, Any]) -> Any:
    if id(record) not in made:
        flat = _flat(record)
        kind, names = _shape_class(flat)
        shape = kind(flat["fName"], flat["fTitle"], "", *(flat[name] for name in names))
        shape.SetLineColor(flat["fLineColor"])
        shape.SetLineStyle(flat["fLineStyle"])
        shape.SetLineWidth(flat["fLineWidth"])
        shape.SetVisibility(flat["fVisibility"])
        made[id(record)] = shape
    return made[id(record)]


def _matrix(record: Any, made: dict[int, Any]) -> Any:
    if not isinstance(record, dict):
        return None
    if id(record) not in made:
        flat = _flat(record)
        made[id(record)] = TRotMatrix(flat["fName"], flat["fTitle"], list(flat["fMatrix"]))
    return made[id(record)]


def _node(record: dict[str, Any], geometry: TGeometry, parent: Any, made: dict[int, Any]) -> None:
    flat = _flat(record)
    geometry.SetCurrentNode(parent)
    node = TNode(flat["fName"], flat["fTitle"], _shape(flat["fShape"], made), flat["fX"],
                 flat["fY"], flat["fZ"], _matrix(flat["fMatrix"], made) or "")  # fmt: skip
    for member in ("LineColor", "LineStyle", "LineWidth"):
        getattr(node, f"Set{member}")(flat[f"f{member}"])
    node._visibility, node._bits = int(flat["fVisibility"]), int(flat["fBits"])
    for daughter in flat["fNodes"] or ():
        _node(daughter, geometry, node, made)


def _each(flat: dict[str, Any], key: str) -> list[Any]:
    """A list the geometry holds, empty if it wrote none."""
    return list(flat.get(key) or ())


def geometry_from(members: dict[str, Any]) -> TGeometry:
    """The ``TGeometry`` whose members these are, its shapes and nodes made again."""
    flat = _flat(members)
    geometry = TGeometry(flat["fName"], flat["fTitle"])
    geometry.SetBomb(flat.get("fBomb", 1.0))
    made: dict[int, Any] = {}
    for record in _each(flat, "fMatrices"):
        _matrix(record, made)
    for record in _each(flat, "fShapes"):
        _shape(record, made)
    for record in _each(flat, "fNodes"):
        _node(record, geometry, None, made)
    geometry.SetCurrentNode(None)
    return geometry


register_members("TGeometry", geometry_from)
