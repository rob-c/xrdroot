"""ROOT's geometry package: ``TGeoManager``, the ``TGeo`` solids, volumes and matrices.

    >>> import xrdroot.pyroot as ROOT
    >>> geom = ROOT.TGeoManager("world", "a box in a box")           # doctest: +SKIP
    >>> vacuum = ROOT.TGeoMedium("Vacuum", 1, ROOT.TGeoMaterial("Vacuum", 0, 0, 0))
    >>> top = geom.MakeBox("TOP", vacuum, 100, 100, 100)             # doctest: +SKIP
    >>> geom.SetTopVolume(top); geom.CloseGeometry(); top.Draw()     # doctest: +SKIP

A geometry is built here as ROOT builds it, with ROOT's names and
arguments, and drawn in a pad as ROOT draws one there in batch - a
wireframe in the pad's perspective - or handed to plotly or pyvista as a
scene (:mod:`xrdroot.geom.backends`). What needs ROOT's navigator - where a
point is, where a ray goes, which volumes overlap - is refused by name.
The older geometry classes, ``TGeometry``, ``TNode`` and ``TBRIK`` and
theirs, are in :mod:`.legacy`.
"""

from __future__ import annotations

from . import legacyread  # noqa: F401 - a TGeometry read from a file is made again
from .iterators import TGeoIterator, TGeoIteratorPlugin
from .manager import TGeoManager, TVirtualGeoPainter, gGeoManager
from .materials import TGeoElement, TGeoMaterial, TGeoMedium, TGeoMixture
from .matrices import (
    TGeoCombiTrans,
    TGeoHMatrix,
    TGeoIdentity,
    TGeoMatrix,
    TGeoRotation,
    TGeoScale,
    TGeoTranslation,
    gGeoIdentity,
)
from .polyshapes import TGeoCompositeShape, TGeoPcon, TGeoPgon, TGeoTessellated, TGeoXtru
from .shapes import (
    TGeoArb8,
    TGeoBBox,
    TGeoCone,
    TGeoConeSeg,
    TGeoCtub,
    TGeoEltu,
    TGeoGtra,
    TGeoHype,
    TGeoPara,
    TGeoParaboloid,
    TGeoShape,
    TGeoSphere,
    TGeoTorus,
    TGeoTrap,
    TGeoTrd1,
    TGeoTrd2,
    TGeoTube,
    TGeoTubeSeg,
)  # fmt: skip
from .volumes import TGeoNode, TGeoNodeMatrix, TGeoVolume, TGeoVolumeAssembly

__all__ = [
    "TGeoManager", "gGeoManager", "TVirtualGeoPainter", "TGeoIterator", "TGeoIteratorPlugin",
    "TGeoElement", "TGeoMaterial", "TGeoMixture", "TGeoMedium",
    "TGeoMatrix", "TGeoIdentity", "TGeoTranslation", "TGeoRotation", "TGeoCombiTrans",
    "TGeoHMatrix", "TGeoScale", "gGeoIdentity",
    "TGeoShape", "TGeoBBox", "TGeoArb8", "TGeoTrd1", "TGeoTrd2", "TGeoPara", "TGeoTrap",
    "TGeoGtra", "TGeoTube", "TGeoTubeSeg", "TGeoCtub", "TGeoCone", "TGeoConeSeg", "TGeoSphere",
    "TGeoTorus", "TGeoEltu", "TGeoParaboloid", "TGeoHype", "TGeoPcon", "TGeoPgon", "TGeoXtru",
    "TGeoTessellated", "TGeoCompositeShape",
    "TGeoVolume", "TGeoVolumeAssembly", "TGeoNode", "TGeoNodeMatrix",
]  # fmt: skip
