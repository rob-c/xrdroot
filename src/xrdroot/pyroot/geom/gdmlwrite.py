"""The geometry being built, as plain GDML data: what ``TGeoManager::Export`` writes.

``TGeoManager::Export`` hands a geometry to ``TGDMLWrite`` when the file
asked for is a ``.gdml`` one, and to the ROOT file writer otherwise. This
module is the first half of that: it reads the manager's own tables -
materials, media, solids and the nodes that place volumes - into the plain
:class:`~xrdroot.geom.gdml.Geometry` the GDML reader of ``geom.gdml`` takes
back, so a geometry written here is one this library reads again, and one
Geant4's GDML reader takes as ROOT's writer states it.

A solid GDML has no element for is refused by name rather than written as
something else: a torus or an extruded polygon written as its bounding box
would be a plausible-looking file that is not the geometry.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ...geom.gdml import Element, Geometry, Material, Placement, Solid, Volume

__all__ = ["geometry_of", "written_shapes"]

#: Each shape GDML states exactly as ROOT keeps it: the class, the GDML element, and a
#: reader of the shape's numbers - ``(shape)`` giving ``params`` in the reader's order.
_SHAPES: dict[str, tuple[str, Any]] = {
    "TGeoBBox": ("box", lambda s: (s.GetDX(), s.GetDY(), s.GetDZ())),
    "TGeoTube": ("tube", lambda s: (s.GetRmin(), s.GetRmax(), s.GetDz(), 0.0, 360.0)),
    "TGeoTubeSeg": ("tube", lambda s: (s.GetRmin(), s.GetRmax(), s.GetDz(), s.GetPhi1(),
                                       s.GetPhi2() - s.GetPhi1())),
    "TGeoCone": ("cone", lambda s: (s.GetRmin1(), s.GetRmax1(), s.GetRmin2(), s.GetRmax2(),
                                    s.GetDz(), 0.0, 360.0)),
    "TGeoConeSeg": ("cone", lambda s: (s.GetRmin1(), s.GetRmax1(), s.GetRmin2(), s.GetRmax2(),
                                       s.GetDz(), s.GetPhi1(), s.GetPhi2() - s.GetPhi1())),
    "TGeoTrd1": ("trd", lambda s: (s.GetDx1(), s.GetDx2(), s.GetDy(), s.GetDy(), s.GetDz())),
    "TGeoTrd2": ("trd", lambda s: (s.GetDx1(), s.GetDx2(), s.GetDy1(), s.GetDy2(), s.GetDz())),
    "TGeoSphere": ("sphere", lambda s: (s.GetRmin(), s.GetRmax(), s.GetPhi1(),
                                        s.GetPhi2() - s.GetPhi1(), s.GetTheta1(),
                                        s.GetTheta2() - s.GetTheta1())),
}
#: What ``written_shapes`` answers: the ROOT class names GDML has a solid for.
written_shapes = tuple(_SHAPES)


def geometry_of(manager: Any) -> Geometry:
    """The manager's tables as ``geom.gdml``'s plain data, ready to be written."""
    geometry = Geometry(setup=manager.GetName(), world="")
    _materials(manager, geometry)
    volumes = list(manager.GetListOfVolumes())
    _solids(volumes, geometry)
    for volume in volumes:
        geometry.volumes[volume.GetName()] = _volume(volume)
    top = manager.GetTopVolume()
    if top is None:
        raise UnsupportedFeatureError(
            "TGeoManager::Export writes a geometry that has been closed, and this one has no "
            "top volume; call CloseGeometry, or SetTopVolume, before exporting it"
        )
    geometry.world = top.GetName()
    return geometry


def _materials(manager: Any, geometry: Geometry) -> None:
    """Every material in the manager, with the elements a mixture is made of."""
    for material in manager.GetListOfMaterials():
        name = str(material.GetName())
        parts = _parts(name, material, geometry)
        geometry.materials[name] = Material(name=name,
                                            density=float(material.GetDensity()), parts=parts)


def _parts(name: str, material: Any, geometry: Geometry) -> tuple[tuple[str, float], ...]:
    """A mixture's elements by weight, each written as an ``<element>`` of its own.

    ROOT keeps a mixture as atomic masses and numbers, not as named elements,
    so each is named after the material and the place it holds in it; its
    ``Z`` and mass state it exactly, which is what a reader builds it from.
    """
    if not material.IsMixture():
        element = f"{name}_el"
        geometry.elements[element] = Element(name=element, formula=f"Z{material.GetZ():g}",
                                             z=round(float(material.GetZ())),
                                             a=float(material.GetA()))
        return ((element, 1.0),)
    found = []
    for index, (a, z, weight) in enumerate(getattr(material, "_elements", ())):
        element = f"{name}_el{index}"
        geometry.elements[element] = Element(name=element, formula=f"Z{z:g}",
                                             z=round(float(z)), a=float(a))
        found.append((element, float(weight)))
    return tuple(found)


def _solids(volumes: list[Any], geometry: Geometry) -> None:
    """A solid per volume's shape, named as ROOT names a shape - the volume's own name."""
    for volume in volumes:
        shape = volume.GetShape()
        kind = type(shape).__name__
        found = _SHAPES.get(kind)
        if found is None:
            raise UnsupportedFeatureError(
                f"TGeoManager::Export writes GDML, which has no solid for ROOT's {kind}, so the "
                f"geometry volume {volume.GetName()!r} is of cannot be written; build the "
                f"geometry of boxes, tubes, cones, trapezoids or spheres, or write it as a ROOT "
                f"file instead"
            )
        element, params = found
        geometry.solids[str(volume.GetName())] = Solid(
            name=str(volume.GetName()), shape=element,
            params=tuple(float(value) for value in params(shape)))


def _volume(volume: Any) -> Volume:
    """One volume: its material, its solid, and a placement for each node inside it."""
    medium = volume.GetMedium()
    material = "" if medium is None else str(medium.GetMaterial().GetName())
    placements = [_placement(volume.GetNode(index))
                  for index in range(int(volume.GetNdaughters()))]
    return Volume(name=str(volume.GetName()), solid=str(volume.GetName()),
                  material=material, placements=placements)


def _placement(node: Any) -> Placement:
    """One node's volume, where it is put and how it is turned, in the mother's frame."""
    matrix = node.GetMatrix()._xrd
    if matrix.is_reflection():
        raise UnsupportedFeatureError(
            f"TGeoManager::Export writes GDML, which places a volume by a rotation and a "
            f"translation, and the node {node.GetName()!r} is placed by a reflection; place it "
            f"with a rotation instead, or write the geometry as a ROOT file"
        )
    where = matrix.translation
    turned = matrix.euler_angles()
    return Placement(volume=str(node.GetVolume().GetName()),
                     position=(float(where[0]), float(where[1]), float(where[2])),
                     rotation=(float(turned[0]), float(turned[1]), float(turned[2])))
