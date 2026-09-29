"""Detector geometry: ROOT's ``TGeo`` solids, placements and pictures, over NumPy.

A geometry is a tree of volumes, each a shape of a material, placed in its
mother by a rotation and a translation. What of that is arithmetic is here:
:mod:`.matrix` the placements, :mod:`.shapes` every solid ROOT has as a
surface of points and faces (:mod:`.mesh`), :mod:`.wavefront` the ``.obj``
files ``TGeoTessellated`` imports. What of it is a picture is here too: the
pad's wireframe in ``TView3D``'s perspective (:mod:`.view`, :mod:`.paint`),
and a scene to turn round in plotly or pyvista (:mod:`.backends`).

:mod:`xrdroot.pyroot.geom` is ROOT's classes over these - ``TGeoManager``
and ``gGeoManager``, the ``TGeo`` shapes, volumes, media and matrices, and
the older ``TGeometry``, ``TNode`` and ``TBRIK`` family.
"""

from __future__ import annotations

from .matrix import IDENTITY, Matrix
from .mesh import Mesh
from .scene import Solid

__all__ = ["Matrix", "IDENTITY", "Mesh", "Solid"]
