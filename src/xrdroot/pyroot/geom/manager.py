"""``TGeoManager`` and ``gGeoManager``: the geometry being built, and everything in it.

Making a ``TGeoManager`` makes it the current geometry, ``gGeoManager``,
as ROOT's constructor does; making a material with no geometry yet makes a
default one first. The manager lists every material, medium, shape,
volume and matrix made while it is current, makes volumes of each shape
(``MakeBox``, ``MakeTube``...), and holds the settings drawing reads - the
visible depth, the visualisation option, the segments a circle is drawn
in. Closing it counts its nodes and levels and says so on standard error,
in ROOT's words; its navigation - where a point is, where a ray goes - is
not here, and is refused by name.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.collections import TList
from ..core.messages import Info
from ..core.objects import TNamed
from .makers import Makers
from .painting import walk

__all__ = ["TGeoManager", "gGeoManager", "current_manager", "TVirtualGeoPainter"]

#: The geometry being built: ``gGeoManager``, or nothing yet.
_CURRENT: list[TGeoManager] = []


class TVirtualGeoPainter(TNamed):
    """``GetGeomPainter()``: what draws the geometry, and the plugin it walks the tree with."""

    def __init__(self) -> None:
        super().__init__("", "")
        self.plugin: Any = None
        self.top: Any = None

    def SetIteratorPlugin(self, plugin: Any) -> None:
        self.plugin = plugin

    def ModifiedPad(self, update: bool = False) -> None:
        """``ModifiedPad``: the pad is repainted from the tree whenever it is drawn."""

    def SetRaytracing(self, flag: bool = True) -> None:
        """``SetRaytracing``: ROOT's ray-traced picture is not drawn here; the wireframe stays."""


class TGeoManager(Makers, TNamed):
    """``TGeoManager(name, title)``: a geometry, made current as it is made."""

    def __init__(self, name: Any = "Geometry", title: Any = "default geometry") -> None:
        super().__init__(str(name), str(title))
        self._lists: dict[str, TList] = {
            key: TList() for key in ("materials", "media", "shapes", "volumes", "matrices")
        }
        self._top: Any = None
        self._settings = {"nsegments": 20, "vislevel": 3, "visoption": 1, "topvisible": False}
        self._closed = False
        self._painter = TVirtualGeoPainter()
        _CURRENT[:] = [self]
        Info("TGeoManager::TGeoManager", f"Geometry {self.GetName()}, {self.GetTitle()} created")

    # -- what it lists ------------------------------------------------------------

    def _add(self, kind: str, obj: Any) -> int:
        listed = self._lists[kind]
        listed.Add(obj)
        return listed.GetEntries() - 1

    def AddMaterial(self, material: Any) -> int:
        return self._add("materials", material)

    def AddMedium(self, medium: Any) -> int:
        return self._add("media", medium)

    def AddShape(self, shape: Any) -> int:
        return self._add("shapes", shape)

    def AddVolume(self, volume: Any) -> int:
        return self._add("volumes", volume)

    def AddTransformation(self, matrix: Any) -> int:
        return self._add("matrices", matrix)

    def GetListOfMaterials(self) -> TList:
        return self._lists["materials"]

    def GetListOfMedia(self) -> TList:
        return self._lists["media"]

    def GetListOfShapes(self) -> TList:
        return self._lists["shapes"]

    def GetListOfVolumes(self) -> TList:
        return self._lists["volumes"]

    def GetListOfMatrices(self) -> TList:
        return self._lists["matrices"]

    def _named(self, kind: str, name: Any) -> Any:
        if isinstance(name, int):
            listed = self._lists[kind]
            return listed.At(name) if 0 <= name < listed.GetEntries() else None
        return next((obj for obj in self._lists[kind] if obj.GetName() == str(name)), None)

    def GetMaterial(self, name: Any) -> Any:
        return self._named("materials", name)

    def GetMedium(self, name: Any) -> Any:
        return self._named("media", name)

    def GetVolume(self, name: Any) -> Any:
        return self._named("volumes", name)

    def GetShape(self, name: Any) -> Any:
        """The first shape of that name, as ROOT's list search finds it."""
        return self._named("shapes", name)

    def GetGeomPainter(self) -> TVirtualGeoPainter:
        return self._painter

    # -- the tree -----------------------------------------------------------------

    def SetTopVolume(self, volume: Any) -> None:
        self._top = volume
        Info("TGeoManager::SetTopVolume",
             f"Top volume is {volume.GetName()}. Master volume is {volume.GetName()}")  # fmt: skip

    def GetTopVolume(self) -> Any:
        return self._top

    def GetMasterVolume(self) -> Any:
        return self._top

    def CloseGeometry(self, option: str = "d") -> None:
        """``CloseGeometry``: the tree counted - nodes, volumes and depth - as ROOT reports it."""
        if self._top is None:
            self.Error("CloseGeometry", "you have to define the top volume first")
            return
        nodes = list(walk(self._top))
        depth = max((level for *_, level, _ in nodes), default=0)
        widest = max((volume.GetNdaughters() for volume in self._lists["volumes"]), default=0)
        for place, said in (
            ("CheckGeometry", "Fixing runtime shapes..."), ("CheckGeometry", "...Nothing to fix"),
            ("CloseGeometry", "Counting nodes..."), ("Voxelize", "Voxelizing..."),
            ("CloseGeometry", "Building cache..."),
            ("CountLevels", f"max level = {depth}, max placements = {widest}"),
            ("CloseGeometry", f"{len(nodes) + 1} nodes/ {self._unique_volumes()} "
                              f"volume UID's in {self.GetTitle()}"),
            ("CloseGeometry", "----------------modeler ready----------------"),
        ):  # fmt: skip
            Info(f"TGeoManager::{place}", said)
        self._closed = True

    def _unique_volumes(self) -> int:
        """How many volume UIDs there are: volumes of one name share one, as ROOT's do."""
        return len({volume.GetName() for volume in self._lists["volumes"]})

    def IsClosed(self) -> bool:
        return self._closed

    # -- how it is drawn ----------------------------------------------------------

    def SetVisLevel(self, level: int = 3) -> None:
        if int(level) > 0:
            self._settings["vislevel"] = int(level)
            Info("TGeoManager::SetVisLevel", "Automatic visible depth disabled")

    def GetVisLevel(self) -> int:
        return int(self._settings["vislevel"])

    def SetVisOption(self, option: int = 0) -> None:
        self._settings["visoption"] = int(option)

    def GetVisOption(self) -> int:
        return int(self._settings["visoption"])

    def SetNsegments(self, count: int) -> None:
        self._settings["nsegments"] = max(int(count), 3)

    def GetNsegments(self) -> int:
        return int(self._settings["nsegments"])

    def SetTopVisible(self, visible: bool = True) -> None:
        self._settings["topvisible"] = bool(visible)

    def GetTopVisible(self) -> bool:
        return bool(self._settings["topvisible"])

    IsTopVisible = GetTopVisible

    def SetMaxVisNodes(self, count: int = 10000) -> None:
        """``SetMaxVisNodes``: every node down to the visible depth is drawn, however many."""

    def DefaultColors(self) -> None:
        """``DefaultColors``: each volume coloured by its material's number, as ROOT does."""
        materials = self._lists["materials"]
        for volume in self._lists["volumes"]:
            material = volume.GetMaterial()
            if material is not None:
                volume.SetLineColor(2 + (1 + materials.IndexOf(material)) % 6)

    # -- what it does not do ------------------------------------------------------

    def _navigation(self, what: str) -> Any:
        raise UnsupportedFeatureError(
            f"TGeoManager::{what} needs ROOT's geometry navigator (TGeoNavigator), which "
            f"xrdroot.pyroot does not have; the geometry can be built and drawn, not navigated"
        )

    def CreateParallelWorld(self, name: str) -> Any:
        return self._navigation("CreateParallelWorld")

    def MakePhysicalNode(self, path: str = "") -> Any:
        return self._navigation("MakePhysicalNode")

    def CheckOverlaps(self, *args: Any) -> Any:
        return self._navigation("CheckOverlaps")

    def FindNode(self, *args: Any) -> Any:
        return self._navigation("FindNode")

    def Export(self, filename: str, name: str = "", option: str = "vg") -> Any:
        raise UnsupportedFeatureError(
            f"TGeoManager::Export writes the geometry as a ROOT file or GDML, neither of which "
            f"xrdroot.pyroot writes a geometry as; {filename!r} was not written"
        )


def current_manager() -> TGeoManager:
    """``gGeoManager``, made - as a default geometry - if there is none yet."""
    if not _CURRENT:
        TGeoManager()
    return _CURRENT[0]


class _Current:
    """``gGeoManager``: the geometry made last, whenever it is asked; false when there is none."""

    def __getattr__(self, name: str) -> Any:
        if not _CURRENT:
            raise AttributeError(f"gGeoManager is null - no geometry has been made - so it "
                                 f"has no {name}")  # fmt: skip
        return getattr(_CURRENT[0], name)

    def __bool__(self) -> bool:
        return bool(_CURRENT)

    def __eq__(self, other: object) -> bool:
        return (_CURRENT[0] if _CURRENT else None) is other

    def __hash__(self) -> int:
        return id(self)


gGeoManager: Any = _Current()
