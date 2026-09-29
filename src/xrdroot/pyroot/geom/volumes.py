"""``TGeoVolume`` and ``TGeoNode``: a shape of a medium, and each place it is put.

A volume holds its daughters as nodes, each a volume placed by a matrix
with a copy number, named ``<volume>_<copy>``. An assembly
(``TGeoVolumeAssembly``) is a volume with no shape of its own, only its
daughters. Drawing a volume draws the tree below it as ``TGeoPainter``
does: a wireframe of every visible volume down to the geometry's visible
depth, in each volume's line colour, style and width.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.objects import TAttFill, TAttLine, TNamed
from .matrices import TGeoMatrix, gGeoIdentity

__all__ = ["TGeoVolume", "TGeoVolumeAssembly", "TGeoNode", "TGeoNodeMatrix"]


def _manager() -> Any:
    from .manager import current_manager

    return current_manager()


class TGeoNode(TNamed):
    """``TGeoNode``: a volume placed in its mother by a matrix, with its copy number."""

    def __init__(self, volume: Any, number: int, mother: Any, matrix: Any,
                 overlapping: bool = False) -> None:  # fmt: skip
        super().__init__(f"{volume.GetName()}_{int(number)}", "")
        self._volume, self._number, self._mother = volume, int(number), mother
        self._matrix = matrix if matrix is not None else gGeoIdentity
        self._overlapping = overlapping

    def GetVolume(self) -> Any:
        return self._volume

    def GetMotherVolume(self) -> Any:
        return self._mother

    def GetMatrix(self) -> TGeoMatrix:
        return self._matrix

    def SetMatrix(self, matrix: TGeoMatrix) -> None:
        self._matrix = matrix

    def GetNumber(self) -> int:
        return self._number

    def IsOverlapping(self) -> bool:
        return self._overlapping

    def GetNdaughters(self) -> int:
        return int(self._volume.GetNdaughters())

    def GetDaughter(self, index: int) -> Any:
        return self._volume.GetNode(index)

    def GetMedium(self) -> Any:
        return self._volume.GetMedium()

    def IsVisible(self) -> bool:
        return bool(self._volume.IsVisible())

    def SetVisibility(self, visible: bool = True) -> None:
        self._volume.SetVisibility(visible)

    def Draw(self, option: str = "") -> None:
        self._volume.Draw(option)


TGeoNodeMatrix = TGeoNode


class TGeoVolume(TNamed, TAttLine, TAttFill):
    """``TGeoVolume(name, shape[, medium])``: a shape of a medium, and its daughters."""

    def __init__(self, name: Any = "", shape: Any = None, medium: Any = None) -> None:
        super().__init__(str(name), "")
        self._shape, self._medium = shape, medium
        self._nodes: list[TGeoNode] = []
        self._visible, self._daughters_visible = True, True
        self._transparency = 0
        _manager().AddVolume(self)

    # -- what it is ---------------------------------------------------------------

    def GetShape(self) -> Any:
        return self._shape

    def SetShape(self, shape: Any) -> None:
        self._shape = shape

    def GetMedium(self) -> Any:
        return self._medium

    def SetMedium(self, medium: Any) -> None:
        self._medium = medium

    def GetMaterial(self) -> Any:
        return self._medium.GetMaterial() if self._medium is not None else None

    def IsAssembly(self) -> bool:
        return False

    # -- its daughters ------------------------------------------------------------

    def AddNode(self, volume: Any, number: int = 0, matrix: Any = None, option: str = "") -> Any:
        """``AddNode``: place ``volume`` in this one, by ``matrix`` (none: at the centre)."""
        node = TGeoNode(volume, number, self, matrix)
        self._nodes.append(node)
        return node

    def AddNodeOverlap(self, volume: Any, number: int = 0, matrix: Any = None,
                       option: str = "") -> Any:  # fmt: skip
        node = TGeoNode(volume, number, self, matrix, overlapping=True)
        self._nodes.append(node)
        return node

    def GetNdaughters(self) -> int:
        return len(self._nodes)

    def GetNodes(self) -> list[TGeoNode]:
        return list(self._nodes)

    def GetNode(self, which: Any) -> Any:
        """The daughter at ``which``, or the one named ``which``; ``None`` if there is none."""
        if isinstance(which, str):
            return next((node for node in self._nodes if node.GetName() == which), None)
        index = int(which)
        return self._nodes[index] if 0 <= index < len(self._nodes) else None

    def GetNtotal(self) -> int:
        """Every node below this volume, counting itself, as ``CountNodes`` does."""
        return 1 + sum(int(node.GetVolume().GetNtotal()) for node in self._nodes)

    def Divide(self, *args: Any) -> Any:
        raise UnsupportedFeatureError(
            "TGeoVolume::Divide - slicing a volume into divisions - is not supported by "
            "xrdroot.pyroot; place the slices with AddNode instead"
        )

    # -- how it looks ------------------------------------------------------------

    def SetVisibility(self, visible: bool = True) -> None:
        self._visible = bool(visible)

    def IsVisible(self) -> bool:
        return self._visible

    def VisibleDaughters(self, visible: bool = True) -> None:
        self._daughters_visible = bool(visible)

    def IsVisDaughters(self) -> bool:
        return self._daughters_visible

    def InvisibleAll(self, flag: bool = True) -> None:
        """``InvisibleAll``: hide this volume and every one below it (or show them)."""
        self.SetVisibility(not flag)
        for node in self._nodes:
            node.GetVolume().InvisibleAll(flag)

    def SetTransparency(self, transparency: int = 0) -> None:
        self._transparency = int(transparency)

    def GetTransparency(self) -> int:
        material = self.GetMaterial()
        return self._transparency or (material.GetTransparency() if material else 0)

    def SetVisContainers(self, flag: bool = True) -> None:
        _manager().SetVisOption(0 if flag else 1)

    def SetVisLeaves(self, flag: bool = True) -> None:
        _manager().SetVisOption(1 if flag else 0)

    def SetVisOnly(self, flag: bool = True) -> None:
        _manager().SetVisOption(2 if flag else 0)

    @property
    def _xrd(self) -> Any:
        """What a pad paints of this volume: the solids of the tree below it, as they are now."""
        from .painting import primitive

        return primitive(self)

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad - cleared first unless ``same`` - as a wireframe."""
        from .painting import draw_volume

        draw_volume(self, str(option))

    def Raytrace(self, flag: bool = True) -> None:
        """``Raytrace``: ROOT's ray-traced picture is not drawn here; the wireframe stays."""

    def RandomRays(self, *args: Any) -> None:
        raise UnsupportedFeatureError(
            "TGeoVolume::RandomRays tracks rays through the geometry with TGeoNavigator, which "
            "xrdroot.pyroot does not have; the geometry can be built and drawn, not navigated"
        )

    def CheckOverlaps(self, *args: Any) -> None:
        raise UnsupportedFeatureError(
            "TGeoVolume::CheckOverlaps needs TGeoNavigator's distance and inside tests, which "
            "xrdroot.pyroot does not have; the geometry can be built and drawn, not navigated"
        )


class TGeoVolumeAssembly(TGeoVolume):
    """``TGeoVolumeAssembly(name)``: daughters grouped, with no shape of its own."""

    def __init__(self, name: Any = "") -> None:
        super().__init__(name, None, None)

    def IsAssembly(self) -> bool:
        return True
