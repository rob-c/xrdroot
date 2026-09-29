"""The ``TGeo`` solids built up after they are made: section by section, facet by facet.

A polycone or polygon (``TGeoPcon``, ``TGeoPgon``) is given its sections
with ``DefineSection``, an extruded polygon (``TGeoXtru``) its outline
with ``DefinePolygon`` and its sections after, a tessellated solid
(``TGeoTessellated``) its facets with ``AddFacet`` - or all at once from
a Wavefront ``.obj`` file - and a composite (``TGeoCompositeShape``) is a
Boolean expression over other shapes, drawn as its components are.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...geom import shapes as build
from ...geom.composite import leaves, parse
from ...geom.mesh import Mesh, merged
from ...geom.wavefront import ObjError, read_obj
from ..core.messages import Error
from .matrices import REGISTERED
from .shapes import TGeoShape

__all__ = ["TGeoPcon", "TGeoPgon", "TGeoXtru", "TGeoTessellated", "TGeoCompositeShape"]


def _point(given: Any) -> tuple[float, float, float]:
    """A vertex - three numbers, or a ``Vertex_t`` - as three floats."""
    return float(given[0]), float(given[1]), float(given[2])


class TGeoPcon(TGeoShape):
    """``TGeoPcon(phi, dphi, nz)``: a polycone, its ``nz`` sections given by ``DefineSection``."""

    PARAMETERS: ClassVar[tuple[str, ...]] = ("Phi1", "Dphi", "Nz")

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._sections = [[0.0, 0.0, 0.0] for _ in range(int(self._values[-1]))]

    def DefineSection(self, index: int, z: float, rmin: float, rmax: float) -> None:
        self._sections[int(index)] = [float(z), float(rmin), float(rmax)]

    def GetZ(self, index: int = 0) -> float:
        return self._sections[int(index)][0]

    def GetRmin(self, index: int = 0) -> float:
        return self._sections[int(index)][1]

    def GetRmax(self, index: int = 0) -> float:
        return self._sections[int(index)][2]

    def _mesh(self, steps: int) -> Mesh:
        return build.pcon(self._values[0], self._values[1], self._sections, steps)


class TGeoPgon(TGeoPcon):
    """``TGeoPgon(phi, dphi, nedges, nz)``: a polycone of ``nedges`` flat sides."""

    PARAMETERS = ("Phi1", "Dphi", "Nedges", "Nz")

    def _mesh(self, steps: int) -> Mesh:
        return build.pgon(self._values[0], self._values[1], int(self._values[2]), self._sections)


class TGeoXtru(TGeoShape):
    """``TGeoXtru(nz)``: a polygon (``DefinePolygon``) extruded through ``nz`` sections."""

    PARAMETERS = ("Nz",)

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
        self._polygon: list[tuple[float, float]] = []
        self._sections = [[0.0, 0.0, 0.0, 1.0] for _ in range(int(self._values[0]))]

    def DefinePolygon(self, count: int, x: Any, y: Any) -> bool:
        self._polygon = [(float(x[i]), float(y[i])) for i in range(int(count))]
        return True

    def DefineSection(self, index: int, z: float, x0: float = 0.0, y0: float = 0.0,
                      scale: float = 1.0) -> None:  # fmt: skip
        self._sections[int(index)] = [float(z), float(x0), float(y0), float(scale)]

    def GetNvert(self) -> int:
        return len(self._polygon)

    def _mesh(self, steps: int) -> Mesh:
        return build.xtru(self._polygon, self._sections)


class TGeoTessellated(TGeoShape):
    """``TGeoTessellated(name, nfacets)`` or ``(name, vertices)``: a solid of flat facets."""

    def __init__(self, name: Any = "", given: Any = 0) -> None:
        super().__init__(str(name))
        self._vertices: list[tuple[float, float, float]] = []
        if not isinstance(given, (int, float)):
            self._vertices = [_point(vertex) for vertex in given]
        self._facets: list[tuple[int, ...]] = []

    def AddFacet(self, *corners: Any) -> bool:
        """``AddFacet`` by three or four vertex indices, or by three or four points."""
        if corners and not isinstance(corners[0], (int, np.integer)):
            start = len(self._vertices)
            self._vertices += [_point(point) for point in corners]
            corners = tuple(range(start, len(self._vertices)))
        self._facets.append(tuple(int(i) for i in corners))
        return True

    def CloseShape(self, check: bool = False, fixFlipped: bool = True,
                   verbose: bool = True) -> None:  # fmt: skip
        """``CloseShape``: the box is found from the facets whenever it is asked for."""

    def GetNvertices(self) -> int:
        return len(self._vertices)

    def GetNfacets(self) -> int:
        return len(self._facets)

    def ResizeCenter(self, maxsize: float) -> None:
        """``ResizeCenter``: centred on the origin, the longest side of its box ``2 maxsize``."""
        half, origin = self._box()
        scale = float(maxsize) / float(max(half))
        points = (np.asarray(self._vertices) - origin) * scale
        self._vertices = [tuple(point) for point in points]

    def Print(self, option: str = "") -> None:
        print(f"=== Tessellated shape {self.GetName()} having {self.GetNvertices()} vertices "
              f"and {self.GetNfacets()} facets")  # fmt: skip

    def _mesh(self, steps: int) -> Mesh:
        return build.tessellated(self._vertices, self._facets)

    @staticmethod
    def ImportFromObjFormat(path: Any, check: bool = False, verbose: bool = False) -> Any:
        """A tessellated solid from a Wavefront ``.obj`` file, printed - ``None`` if refused."""
        try:
            vertices, faces = read_obj(str(path))
        except ObjError as refused:
            Error("TGeoTessellated::ImportFromObjFormat", str(refused))
            return None
        if verbose:
            print(f"Read {len(vertices)} vertices and {len(faces)} facets from {path}")
        name = str(path).rsplit(".", 1)[0]
        made = TGeoTessellated(name, vertices)
        for face in faces:
            made.AddFacet(*face)
        made.Print()
        return made


class TGeoCompositeShape(TGeoShape):
    """``TGeoCompositeShape([name,] expression)``: shapes joined by ``+``, ``-`` and ``*``."""

    def __init__(self, *args: Any) -> None:
        name, expression = (args[0], args[1]) if len(args) > 1 else ("", args[0] if args else "")
        super().__init__(str(name))
        self._expression = str(expression)
        self._tree = parse(self._expression) if self._expression else None

    def IsComposite(self) -> bool:
        return True

    def GetExpression(self) -> str:
        return self._expression

    def _mesh(self, steps: int) -> Mesh:
        from .manager import current_manager

        found = []
        for shape_name, matrix_name in leaves(self._tree) if self._tree else []:
            shape = current_manager().GetShape(shape_name)
            if shape is None:
                raise ValueError(f"the composite shape {self.GetName()!r} names a shape "
                                 f"{shape_name!r} that the geometry does not have")  # fmt: skip
            mesh = shape._mesh(steps)
            matrix = REGISTERED.get(matrix_name) if matrix_name else None
            found.append(mesh.transformed(matrix._xrd) if matrix is not None else mesh)
        return merged(found)
