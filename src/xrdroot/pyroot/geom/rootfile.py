"""The geometry a ROOT file holds, made into a ``TGeoManager`` as ``Import`` makes one.

``TGeoManager::Export`` writes a geometry as a ROOT file - a ``TGeoManager``
key among the file's records - and ``Import`` opens that file and reads the
manager back; ROOT's own ``roottest/root/io/geo`` keeps eighteen geometries
so. The file is read by the reader every ROOT file is read with, which hands
back the manager's members as plain tables, and the manager is made again
from them: its materials, its media and its volumes by name, one volume per
volume UID, the nodes that place them, and every solid as ROOT's class for
the kind the file names.

The numbers ``CloseGeometry`` reports are counted from the tree the file
describes, not by walking the instances it makes: the file keeps the node
count ROOT itself counted, and the deepest level and the volume with most
daughters come from the volumes' own daughter lists, which are a few
thousand where the placements those daughters make run to millions. A
node's matrix is not read back, so the tree's shape and its counts are what
an imported geometry answers; it is not a picture that can be drawn.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from ..core.files import TFile
from .manager import TGeoManager
from .materials import TGeoMaterial, TGeoMedium, TGeoMixture
from .polyshapes import TGeoPcon, TGeoPgon
from .shapes import (
    TGeoArb8,
    TGeoBBox,
    TGeoCone,
    TGeoConeSeg,
    TGeoShape,
    TGeoSphere,
    TGeoTrap,
    TGeoTrd1,
    TGeoTrd2,
    TGeoTube,
    TGeoTubeSeg,
)
from .volumes import TGeoVolume

__all__ = ["from_file"]

#: The solid kinds xrdroot has ROOT's class for, and the members of ROOT's file that class is
#: made from, in its constructor's order.
_KINDS: dict[str, tuple[type, tuple[str, ...]]] = {
    "TGeoBBox": (TGeoBBox, ("fDX", "fDY", "fDZ")),
    "TGeoTube": (TGeoTube, ("fRmin", "fRmax", "fDz")),
    "TGeoTubeSeg": (TGeoTubeSeg, ("fRmin", "fRmax", "fDz", "fPhi1", "fPhi2")),
    "TGeoCone": (TGeoCone, ("fDz", "fRmin1", "fRmax1", "fRmin2", "fRmax2")),
    "TGeoConeSeg": (TGeoConeSeg, ("fDz", "fRmin1", "fRmax1", "fRmin2", "fRmax2", "fPhi1",
                                  "fPhi2")),
    "TGeoTrap": (TGeoTrap, ("fDz", "fTheta", "fPhi", "fH1", "fBl1", "fTl1", "fAlpha1", "fH2",
                            "fBl2", "fTl2", "fAlpha2")),
    "TGeoTrd1": (TGeoTrd1, ("fDx1", "fDx2", "fDy", "fDz")),
    "TGeoTrd2": (TGeoTrd2, ("fDx1", "fDx2", "fDy1", "fDy2", "fDz")),
    "TGeoArb8": (TGeoArb8, ("fDz",)),
    "TGeoSphere": (TGeoSphere, ("fRmin", "fRmax", "fTheta1", "fTheta2", "fPhi1", "fPhi2")),
    "TGeoPcon": (TGeoPcon, ("fPhi1", "fDphi", "fNz")),
    "TGeoPgon": (TGeoPgon, ("fPhi1", "fDphi", "fNedges", "fNz")),
}

#: The solids whose sections - a ``(z, rmin, rmax)`` each - the file keeps in three arrays rather
#: than in the constructor's arguments.
_SECTIONED = (TGeoPcon,)

#: What tells one solid kind from another: the numbers only its constructor takes, the kinds
#: that take the numbers of the kinds above them first. A file writes a solid's bounding box
#: under it as well - every solid derives from ``TGeoBBox`` - so the class a record is filed
#: under names the kind only in some of ROOT's releases; the numbers always do.
_SIGNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TGeoPgon", ("fNz", "fNedges")),
    ("TGeoPcon", ("fNz", "fZ")),
    ("TGeoSphere", ("fTheta1",)),
    ("TGeoConeSeg", ("fRmin1", "fPhi1")),
    ("TGeoCone", ("fRmin1",)),
    ("TGeoTubeSeg", ("fRmin", "fPhi1")),
    ("TGeoTube", ("fRmin",)),
    ("TGeoTrd2", ("fDx1", "fDy1")),
    ("TGeoTrd1", ("fDx1",)),
    ("TGeoTrap", ("fH1",)),
    ("TGeoArb8", ("fXY",)),
    ("TGeoBBox", ("fDX",)),
)


class TGeoImportedShape(TGeoShape):
    """A solid of a kind xrdroot has no class for - a ``TGeoPcon``, say - as the file holds it.

    The file names the solid's class and keeps every number in it; those are kept, and a
    measurement or a picture of the solid is refused by name rather than guessed at.
    """

    def __init__(self, name: Any, kind: str, members: Any) -> None:
        super().__init__(str(name))
        self._kind, self._members = kind, dict(members)

    def GetKind(self) -> str:
        """The class ROOT's file gave the solid."""
        return self._kind

    def _mesh(self, steps: int) -> Any:
        raise UnsupportedFeatureError(
            f"the solid {self.GetName() or self._kind} is a {self._kind}, which "
            f"xrdroot.pyroot has no class for, so it cannot be measured or drawn; make the "
            f"geometry with TGeoManager's own Make methods instead of reading this file"
        )


def from_file(filename: str) -> TGeoManager:
    """The ``TGeoManager`` the ROOT file at ``filename`` holds, closed and current."""
    reading = TFile.Open(filename)
    if reading is None:
        raise UnsupportedFeatureError(
            f"the file {filename!r} is neither a GDML document nor a ROOT file, so there is no "
            f"geometry in it for TGeoManager::Import to read; name a .gdml file, or a ROOT file "
            f"holding a TGeoManager"
        )
    key = _manager_key(reading, filename)
    read = reading.Get(key.GetName())
    if isinstance(read, TGeoManager):
        return read
    members = getattr(read, "_xrd", None)
    if not isinstance(members, dict):
        raise UnsupportedFeatureError(
            f"the TGeoManager in {filename!r} was read as no more than its name, so the "
            f"geometry it holds is not there to be built; write the geometry as GDML and read "
            f"that instead"
        )
    return _built(members)


def _manager_key(reading: Any, filename: str) -> Any:
    """The first key of the file holding a ``TGeoManager``, as ROOT's ``Import`` finds it."""
    for key in reading.GetListOfKeys():
        if key.GetClassName() == "TGeoManager":
            return key
    raise UnsupportedFeatureError(
        f"the ROOT file {filename!r} holds no TGeoManager, so there is no geometry in it to "
        f"read; read the file that holds the geometry, or build it with TGeoManager's own "
        f"methods instead"
    )


def _built(members: Any) -> TGeoManager:
    """The manager, and everything in it, made from the members ROOT's file records."""
    named = members["TNamed"]
    manager = TGeoManager(named["fName"], named["fTitle"], _hello=False)
    materials = {str(_named(entry)["fName"]): _material(entry)
                 for entry in members.get("fMaterials") or []}
    media = {str(entry["TNamed"]["fName"]): _medium(entry, materials)
             for entry in members.get("fMedia") or []}
    volumes, entries = _volumes(members, media)
    for uid, entry in entries.items():
        for number, node in enumerate(entry.get("fNodes") or []):
            daughter = _placed(node)
            if daughter in volumes:
                volumes[uid].AddNode(volumes[daughter], number)
    top = _top(members, entries)
    manager._loaded_from_file(volumes[top], (int(members.get("fNNodes", 0)),
                                             _uids(members, volumes), _level(entries, top),
                                             _widest(entries)))
    manager.CloseGeometry()
    return manager


def _volumes(members: Any, media: dict[str, Any]) -> tuple[dict[int, Any], dict[int, Any]]:
    """One volume per volume UID, and the entry of the file that describes it.

    ROOT's file lists a volume more than once - a geometry's own list of what it holds counts
    a volume reached by several mothers each time - and the copies describe one and the same
    volume, so the first is taken.
    """
    volumes: dict[int, Any] = {}
    entries: dict[int, Any] = {}
    for entry in members.get("fVolumes") or []:
        uid = int(entry["fNumber"])
        if uid in volumes:
            continue
        named = entry["TNamed"]
        entries[uid] = entry
        volumes[uid] = TGeoVolume(named["fName"], _shape(entry.get("fShape")),
                                  _pointed(entry.get("fMedium"), media))
        volumes[uid].SetTitle(named["fTitle"])
    return volumes, entries


def _uids(members: Any, volumes: dict[int, Any]) -> int:
    """How many volume UIDs the geometry has: ROOT keeps them in a list whose first slot is
    free, and a file that does not keep it is counted by the volume numbers it wrote."""
    unique = members.get("fUniqueVolumes")
    return len(unique) - 1 if unique else len(volumes)


def _top(members: Any, entries: dict[int, Any]) -> int:
    """The volume the tree hangs from: the file's master volume, by the name it gives it."""
    named = (members.get("fMasterVolume") or {}).get("TNamed") or {}
    wanted = str(named.get("fName", ""))
    found = next((uid for uid, entry in entries.items()
                  if str(entry["TNamed"]["fName"]) == wanted), None)
    return found if found is not None else next(iter(entries))


def _daughters(entry: Any) -> list[int]:
    """The UIDs of the volumes one volume's nodes place."""
    return [daughter for daughter in (_placed(node) for node in entry.get("fNodes") or [])
            if daughter is not None]  # fmt: skip


def _placed(node: Any) -> Any:
    """The UID of the volume a node places, or nothing where the node places none."""
    part = node.get("TGeoNode", node) if isinstance(node, dict) else {}
    volume = part.get("fVolume")
    return int(volume["fNumber"]) if isinstance(volume, dict) else None


def _level(entries: dict[int, Any], top: int) -> int:
    """How many levels of daughters hang below the top volume, as ``CountLevels`` counts."""
    depth: dict[int, int] = {}

    def deepest(uid: int) -> int:
        if uid not in depth:
            depth[uid] = 0  # a volume being counted answers with no daughters below it
            depth[uid] = 1 + max((deepest(child) for child in _daughters(entries[uid])
                                  if child in entries), default=0)
        return depth[uid]

    return deepest(top) - 1


def _widest(entries: dict[int, Any]) -> int:
    """The most daughters any one volume has, as ``CountLevels`` counts them."""
    return max((len(entry.get("fNodes") or []) for entry in entries.values()), default=0)


def _named(entry: Any) -> dict[str, Any]:
    """A record's name and title, which a class deriving from another keeps in its base's part."""
    named = entry.get("TNamed") or (entry.get("TGeoMaterial") or {}).get("TNamed")
    return named or {}


def _pointed(entry: Any, table: dict[str, Any]) -> Any:
    """The object a file's pointer member names, out of the table that holds it by name."""
    return table.get(str(_named(entry or {}).get("fName", "")))


def _medium(entry: Any, materials: dict[str, Any]) -> Any:
    """A ``TGeoMedium``: its number, its material and the parameters a transport code reads."""
    return TGeoMedium(entry["TNamed"]["fName"], int(entry.get("fId", 0)),
                      _pointed(entry.get("fMaterial"), materials),
                      entry.get("fParams"))


def _material(entry: Any) -> Any:
    """A ``TGeoMaterial`` - one element - or the ``TGeoMixture`` a material of parts is."""
    named = _named(entry)
    density = _density(entry)
    if int(entry.get("fNelements", 1)) > 1:
        made = TGeoMixture(named["fName"], int(entry["fNelements"]), density)
        for a, z, weight in zip(np.ravel(entry["fAmixture"]), np.ravel(entry["fZmixture"]),
                                np.ravel(entry["fWeights"]), strict=True):  # fmt: skip
            made.AddElement(float(a), float(z), float(weight))
        return made
    return TGeoMaterial(named["fName"], float(entry["fA"]), float(entry["fZ"]), density)


def _density(entry: Any) -> float:
    """A material's density, which a mixture keeps in the material part it derives from."""
    if "fDensity" in entry:
        return float(entry["fDensity"])
    return float(entry["TGeoMaterial"]["fDensity"])


def _shape(entry: Any) -> Any:
    """The solid ROOT's file describes, as ROOT's class for the kind its numbers make it."""
    if not entry:
        return None
    for kind, signs in _SIGNS:
        if all(_field(entry, sign) is not None for sign in signs):
            return _solid(kind, entry)
    name = (_field(entry, "fName") or "")
    return TGeoImportedShape(name, next(iter(entry)), entry)


def _solid(kind: str, entry: Any) -> Any:
    """One solid of the kind its numbers tell it is, or - should a number be missing - the
    numbers kept and the solid refused by name rather than made from guessed ones."""
    found, fields = _KINDS[kind]
    values = [_field(entry, field) for field in fields]
    if any(value is None for value in values):
        return TGeoImportedShape(_field(entry, "fName") or "", kind, entry)
    shape = found(*(float(value) for value in values))
    if isinstance(shape, TGeoArb8):
        for index, (x, y) in enumerate(np.ravel(_field(entry, "fXY")).reshape(8, 2)):
            shape.SetVertex(index, float(x), float(y))
    if isinstance(shape, _SECTIONED):
        _define_sections(shape, entry)
    return shape


def _define_sections(shape: Any, entry: Any) -> None:
    """A polycone's sections, which the file keeps as three arrays rather than in the
    constructor's arguments."""
    columns = (np.ravel(_field(entry, name)) for name in ("fZ", "fRmin", "fRmax"))
    for index, (z, rmin, rmax) in enumerate(zip(*columns, strict=False)):
        shape.DefineSection(index, float(z), float(rmin), float(rmax))


def _field(entry: Any, name: str) -> Any:
    """One number of a solid, which a class deriving from another keeps in its base's part."""
    found = entry.get(name)
    if found is not None:
        return found
    for part in entry.values():
        if isinstance(part, dict):
            found = _field(part, name)
            if found is not None:
                return found
    return None
