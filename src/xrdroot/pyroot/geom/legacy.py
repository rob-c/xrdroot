"""ROOT's first geometry package: ``TGeometry``, ``TNode``, ``TMaterial`` and ``TRotMatrix``.

A ``TGeometry`` lists materials, rotation matrices, shapes and nodes; the
last made is ``gGeometry``. A ``TNode`` places a shape at ``x, y, z``
turned by a matrix, inside whichever node was ``cd()``-ed to last - or at
the top, and ``cd()``-ed to, if none was. Drawing a node paints the tree
below it through the pad's perspective, as the geometry package does:
each visible node's shape, in the node's own colour, the drawn node at the
origin. A geometry read from a file is made one of these again, nodes,
shapes and all. Writing one is refused: its streamer is ROOT's own code.
"""

from __future__ import annotations

from typing import Any

from ...canvas import Primitive
from ...errors import UnsupportedFeatureError
from ...geom import Matrix
from ..core.collections import THashList, TList
from ..core.objects import TAttFill, TAttLine, TNamed

__all__ = ["TGeometry", "gGeometry", "TMaterial", "TMixture", "TRotMatrix", "TNode",
           "current_geometry"]  # fmt: skip

#: The geometry being built: ``gGeometry``, or nothing yet.
_CURRENT: list[TGeometry] = []


class TGeometry(TNamed):
    """``TGeometry(name, title)``: materials, matrices, shapes and nodes; ``gGeometry``."""

    def __init__(self, name: Any = "Geometry", title: Any = "Default Geometry") -> None:
        super().__init__(str(name), str(title))
        self._lists = {key: THashList() for key in ("materials", "matrices", "shapes")}
        self._nodes = TList()
        self._current: Any = None
        self._bomb = 1.0
        self.cd()
        from ..core.troot import gROOT

        gROOT.GetListOfGeometries().Add(self)

    def add(self, kind: str, obj: Any) -> int:
        listed = self._lists[kind]
        listed.Add(obj)
        return int(listed.GetEntries()) - 1

    def cd(self, path: Any = None) -> None:
        """Make this ``gGeometry``."""
        _CURRENT[:] = [self]

    def GetListOfMaterials(self) -> THashList:
        return self._lists["materials"]

    def GetListOfMatrices(self) -> THashList:
        return self._lists["matrices"]

    def GetListOfShapes(self) -> THashList:
        return self._lists["shapes"]

    def GetListOfNodes(self) -> TList:
        return self._nodes

    def GetShape(self, name: Any) -> Any:
        return self._lists["shapes"].FindObject(str(name))

    def GetMaterial(self, name: Any) -> Any:
        return self._lists["materials"].FindObject(str(name))

    def GetRotMatrix(self, name: Any) -> Any:
        return self._lists["matrices"].FindObject(str(name))

    def GetNode(self, name: Any) -> Any:
        """The node of that name anywhere below the first top node."""
        first = self._nodes.First() if self._nodes.GetEntries() else None
        return first.GetNode(name) if first is not None else None

    def FindObject(self, name: Any) -> Any:
        for listed in (*self._lists.values(), self._nodes):
            found = listed.FindObject(name)
            if found is not None:
                return found
        return self.GetNode(name)

    def GetCurrentNode(self) -> Any:
        return self._current

    def SetCurrentNode(self, node: Any) -> None:
        self._current = node

    def SetBomb(self, bomb: float = 1.4) -> None:
        """``SetBomb``: the factor an exploded view spreads nodes by, kept, not drawn."""
        self._bomb = float(bomb)

    def GetBomb(self) -> float:
        return self._bomb

    def Draw(self, option: str = "") -> None:
        """``Draw``: the first top node and everything below it."""
        if self._nodes.GetEntries():
            self._nodes.First().Draw(option)

    def Write(self, name: Any = None, option: int = 0, bufsize: int = 0) -> int:
        raise UnsupportedFeatureError(
            "writing a TGeometry is not supported: its streamer is ROOT's own code, not a "
            "layout this writer carries; the geometry can be built, drawn, and read back from "
            "a file ROOT wrote"
        )


def current_geometry() -> TGeometry:
    """``gGeometry``, made - as a default geometry - if there is none yet."""
    if not _CURRENT:
        TGeometry()
    return _CURRENT[0]


class _Current:
    """``gGeometry``: the geometry made last, whenever it is asked; false when there is none."""

    def __getattr__(self, name: str) -> Any:
        return getattr(current_geometry(), name)

    def __bool__(self) -> bool:
        return bool(_CURRENT)


gGeometry: Any = _Current()


class TMaterial(TNamed, TAttFill):
    """``TMaterial(name, title, a, z, density[, radl, inter])``: a material of the old package."""

    def __init__(self, name: Any = "", title: Any = "", a: float = 0.0, z: float = 0.0,
                 density: float = 0.0, radl: float = 0.0, inter: float = 0.0) -> None:  # fmt: skip
        super().__init__(str(name), str(title))
        self._a, self._z, self._density = float(a), float(z), float(density)
        self._radl, self._inter = float(radl), float(inter)
        self._number = current_geometry().add("materials", self)

    def GetA(self) -> float:
        return self._a

    def GetZ(self) -> float:
        return self._z

    def GetDensity(self) -> float:
        return self._density

    def GetNumber(self) -> int:
        return self._number


class TMixture(TMaterial):
    """``TMixture(name, title, nmixt)``: elements by weight, given by ``DefineElement``."""

    def __init__(self, name: Any = "", title: Any = "", nmixt: int = 0) -> None:
        super().__init__(name, title)
        self._elements = [(0.0, 0.0, 0.0)] * abs(int(nmixt))

    def DefineElement(self, index: int, a: float, z: float, weight: float) -> None:
        self._elements[int(index)] = (float(a), float(z), float(weight))

    def GetNmixt(self) -> int:
        return len(self._elements)


def _rotation(args: tuple[Any, ...]) -> Matrix:
    """``TRotMatrix``'s turn: GEANT3's six angles, or the nine numbers row by row."""
    if len(args) == 1:
        rows = [float(v) for v in args[0][:9]]
        return Matrix([rows[0::3], rows[1::3], rows[2::3]])
    if len(args) >= 6:
        return Matrix.geant(*(float(v) for v in args[:6]))
    print("ERROR: This form of TRotMatrix constructor not implemented yet")
    return Matrix()


class TRotMatrix(TNamed):
    """``TRotMatrix(name, title, theta1, phi1, ..., phi3)``, or ``(name, title, matrix)``."""

    def __init__(self, name: Any = "", title: Any = "", *args: Any) -> None:
        super().__init__(str(name), str(title))
        self._xrd = _rotation(args) if args else Matrix()
        self._number = current_geometry().add("matrices", self)

    def GetMatrix(self) -> Any:
        """The nine numbers, each row a local axis in the mother, as ROOT keeps them."""
        return self._xrd.rotation.T.reshape(9)

    def GetNumber(self) -> int:
        return self._number

    def IsReflection(self) -> bool:
        return self._xrd.is_reflection()


def _found(kind: str, given: Any) -> Any:
    """A shape or matrix given by name, looked up in ``gGeometry``; one given itself, as it is."""
    if not isinstance(given, str):
        return given
    geometry = current_geometry()
    return geometry.GetShape(given) if kind == "shape" else geometry.GetRotMatrix(given)


class TNode(TNamed, TAttLine, TAttFill):
    """``TNode(name, title, shape[, x, y, z, matrix, option])``: a shape placed in its mother."""

    def __init__(self, name: Any = "", title: Any = "", shape: Any = "", x: float = 0.0,
                 y: float = 0.0, z: float = 0.0, matrix: Any = "",
                 option: str = "") -> None:  # fmt: skip
        super().__init__(str(name), str(title))
        geometry = current_geometry()
        self._shape = _found("shape", shape)
        self._at = (float(x), float(y), float(z))
        self._matrix = _found("matrix", matrix) if matrix else None
        self._nodes: list[TNode] = []
        self._visibility, self._bits, self._option = 1, 0, str(option)
        self._parent = geometry.GetCurrentNode()
        if self._shape is None:
            print(f"Error Referenced shape does not exist: {shape}")
            return
        self.ImportShapeAttributes()
        if self._parent is not None:
            self._parent._nodes.append(self)
        else:
            geometry.GetListOfNodes().Add(self)
            self.cd()

    def cd(self, path: Any = None) -> None:
        """Make this the node the next nodes are made inside."""
        current_geometry().SetCurrentNode(self)

    def ImportShapeAttributes(self) -> None:
        """Take the shape's line and fill, and do the same for every node below."""
        for member in ("LineColor", "LineStyle", "LineWidth", "FillColor", "FillStyle"):
            getattr(self, f"Set{member}")(getattr(self._shape, f"Get{member}")())
        for node in self._nodes:
            node.ImportShapeAttributes()

    def GetShape(self) -> Any:
        return self._shape

    def GetParent(self) -> Any:
        return self._parent

    def GetListOfNodes(self) -> list[TNode]:
        return self._nodes

    def GetX(self) -> float:
        return self._at[0]

    def GetY(self) -> float:
        return self._at[1]

    def GetZ(self) -> float:
        return self._at[2]

    def SetPosition(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self._at = (float(x), float(y), float(z))

    def GetMatrix(self) -> Any:
        return self._matrix

    def GetNode(self, name: Any) -> Any:
        """The node of that name: this one, or the first found below it."""
        if self.GetName() == str(name):
            return self
        for node in self._nodes:
            found = node.GetNode(name)
            if found is not None:
                return found
        return None

    def GetVisibility(self) -> int:
        return self._visibility

    def SetVisibility(self, visibility: int = 1) -> None:
        """``SetVisibility``: ROOT's codes, from -4 (only the daughters) to 3 (all of it)."""
        from .legacyvis import set_visibility

        set_visibility(self, int(visibility))

    def placement(self) -> Matrix:
        """Where the node sits in its mother: its matrix's turn, then its shift."""
        turn = self._matrix._xrd.rotation if self._matrix is not None else None
        return Matrix(turn, self._at)

    @property
    def _xrd(self) -> Primitive:
        """What a pad paints of this node: the solids of the tree below it, as it is now."""
        from .legacyvis import solids

        return Primitive("TGeoVolume", {"fName": self.GetName(), "solids": solids(self)})

    def Draw(self, option: str = "") -> None:
        """``Draw``: into the current pad - cleared first unless ``same`` - the drawn node at 0."""
        from .painting import draw_object

        draw_object(self, str(option))

    def DrawOnly(self, option: str = "") -> None:
        self.SetVisibility(2)
        self.Draw(option)
