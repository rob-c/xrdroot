"""A geometry read from a GDML file, made into a ``TGeoManager`` as ``Import`` makes one.

:mod:`xrdroot.geom.gdml` reads the file into plain tables - elements,
materials, solids, volumes and their placements - because the engine knows no
ROOT class. This module is the other half: each table entry is made again as
ROOT's class for it, through the very constructors a macro would use, so an
imported geometry is laid out, counted and drawn exactly as a built one. The
names, the medium of a volume and the order of the shapes are what ROOT's
``TGDMLParse`` makes of the same file.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ...geom import gdml
from .manager import TGeoManager, _said
from .materials import TGeoElement, TGeoMaterial, TGeoMedium, TGeoMixture
from .matrices import TGeoCombiTrans, TGeoRotation, TGeoTranslation
from .polyshapes import TGeoCompositeShape
from .shapes import (
    TGeoBBox,
    TGeoCone,
    TGeoConeSeg,
    TGeoSphere,
    TGeoTrd2,
    TGeoTube,
    TGeoTubeSeg,
)
from .volumes import TGeoVolume

__all__ = ["from_gdml"]


def from_gdml(filename: str, title: str = "Geometry imported from GDML") -> TGeoManager:
    """The ``TGeoManager`` the GDML file at ``filename`` describes, closed and current."""
    geometry = gdml.read(filename)
    if geometry.world not in geometry.volumes:
        raise gdml.GdmlError(f"the <setup> names the volume {geometry.world!r}, which the file "
                             f"does not define")
    manager = TGeoManager("GDMLImport", title)
    manager.SetName(geometry.setup or manager.GetName())
    materials = _materials(geometry)
    media = {name: TGeoMedium(name, number, material)
             for number, (name, material) in enumerate(materials.items())}
    shapes = _solids(geometry)
    volumes = {name: TGeoVolume(name, shapes[volume.solid], media[volume.material])
               for name, volume in geometry.volumes.items()}
    for volume in geometry.volumes.values():
        for number, placed in enumerate(volume.placements):
            volumes[volume.name].AddNode(volumes[placed.volume], number,
                                         _placed(f"{volume.name}_{number}", placed))
    manager.SetTopVolume(volumes[geometry.world])
    _said("TGeoNavigator::BuildCache", "--- Maximum geometry depth set to 100")
    manager.CloseGeometry()
    return manager


def _elements(geometry: gdml.Geometry) -> dict[str, TGeoElement]:
    """``TGeoElement`` per ``<element>``, in the order the file lists them."""
    return {name: TGeoElement(name, element.formula, element.z, element.a)
            for name, element in geometry.elements.items()}


def _materials(geometry: gdml.Geometry) -> dict[str, TGeoMaterial]:
    """``TGeoMaterial`` per ``<material>``: one element's numbers, or a mixture of parts."""
    elements = _elements(geometry)
    made: dict[str, TGeoMaterial] = {}
    for name, material in geometry.materials.items():
        parts = material.parts
        if len(parts) == 1 and parts[0][0] in elements:
            element = elements[parts[0][0]]
            made[name] = TGeoMaterial(name, element.A(), element.Z(), material.density)
            continue
        mixture = TGeoMixture(name, len(parts), material.density)
        for part, fraction in parts:
            mixture.AddElement(*_numbers(part, elements, made), fraction)
        made[name] = mixture
    return made


def _numbers(name: str, elements: dict[str, TGeoElement],
             made: dict[str, TGeoMaterial]) -> tuple[float, float]:
    """The ``(a, z)`` of a mixture's part: its element's, or an earlier material's."""
    if name in elements:
        return elements[name].A(), elements[name].Z()
    if name in made:
        return made[name].GetA(), made[name].GetZ()
    raise gdml.GdmlError(f"a <material> is part {name!r}, which the file defines as neither "
                         f"an element nor a material")


def _solids(geometry: gdml.Geometry) -> dict[str, Any]:
    """Every ``<solid>`` as ROOT's shape for it, each added to the manager's list."""
    made: dict[str, Any] = {}
    for name, solid in geometry.solids.items():
        made[name] = _solid(solid, made)
    return made


def _solid(solid: gdml.Solid, made: dict[str, Any]) -> Any:
    """One solid: a shape of its kind, or the two operands of a ``<subtraction>``."""
    name, p = solid.name, solid.params
    if solid.shape == "box":
        return TGeoBBox(name, *p)
    if solid.shape == "tube":
        return _tube(name, p)
    if solid.shape == "cone":
        return _cone(name, p)
    if solid.shape == "trd":
        return TGeoTrd2(name, *p)
    if solid.shape == "sphere":
        rmin, rmax, phi1, dphi, theta1, dtheta = p
        return TGeoSphere(name, rmin, rmax, theta1, theta1 + dtheta, phi1, phi1 + dphi)
    if solid.shape == "subtraction":
        return _subtraction(solid, made)
    raise UnsupportedFeatureError(
        f"the GDML solid {name!r} is a <{solid.shape}>, which xrdroot's GDML reader does not "
        f"build; build it with TGeoManager's own Make methods instead"
    )


def _tube(name: str, p: tuple[float, ...]) -> Any:
    """A tube, a segment of one when it does not go right round as GDML's default does."""
    rmin, rmax, dz, start, delta = p
    if abs(delta) >= 360.0:
        return TGeoTube(name, rmin, rmax, dz)
    return TGeoTubeSeg(name, rmin, rmax, dz, start, start + delta)


def _cone(name: str, p: tuple[float, ...]) -> Any:
    rmin1, rmax1, rmin2, rmax2, dz, start, delta = p
    if abs(delta) >= 360.0:
        return TGeoCone(name, dz, rmin1, rmax1, rmin2, rmax2)
    return TGeoConeSeg(name, dz, rmin1, rmax1, rmin2, rmax2, start, start + delta)


def _subtraction(solid: gdml.Solid, made: dict[str, Any]) -> TGeoCompositeShape:
    """A composite of ``first`` less ``second``, the second where the file puts it."""
    first, second = solid.operands
    origin, angles = solid.params[:3], solid.params[3:]
    expression = f"{first}-{second}"
    if any(origin) or any(angles):
        matrix = _named_matrix(f"{solid.name}_operand", origin, angles)
        expression = f"{first}-{second}:{matrix.GetName()}"
    return TGeoCompositeShape(solid.name, expression)


def _named_matrix(name: str, origin: Any, angles: Any) -> Any:
    """A matrix under a name, registered so that a composite's expression can cite it."""
    turn = TGeoRotation(f"{name}_rot", *angles)
    matrix = TGeoCombiTrans(name, *origin, turn)
    matrix.RegisterYourself()
    return matrix


def _placed(name: str, placed: gdml.Placement) -> Any:
    """Where a ``<physvol>`` puts its volume: none when it is at the centre, unturned."""
    origin, angles = placed.position, placed.rotation
    if not any(origin):
        return TGeoRotation(f"{name}_rot", *angles) if any(angles) else None
    if not any(angles):
        return TGeoTranslation(name, *origin)
    return TGeoCombiTrans(name, *origin, TGeoRotation(f"{name}_rot", *angles))
