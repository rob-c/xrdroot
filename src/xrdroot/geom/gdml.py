"""GDML, the XML geometry description ``TGDMLParse`` reads into a ``TGeoManager``.

A GDML file is five tables and one tree: ``<element>`` and ``<material>`` say
what matter is made of, ``<position>`` and ``<rotation>`` name the placements,
``<solid>`` gives each shape's parameters, ``<volume>`` pairs a solid with a
material, and ``<physvol>`` inside a volume places another volume in it, at a
named or an inline position and rotation. ``<setup>`` names the volume the
tree hangs from. This module reads those into plain data so that the
``TGeoManager`` wrapper can build its volumes and nodes from them, without
this side knowing any ROOT class.

Two conventions of ROOT's reader are kept: lengths are converted to
centimetres (the unit ``TGeoBBox`` and ``TGeoTube`` work in, whatever
``unit`` the file states) and a solid's half lengths, radii and z extent are
what the file gives, since GDML states a box's half width too.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["read", "Geometry", "GdmlError", "Counts"]

#: How many centimetres one of GDML's length units is.
LENGTH_UNITS = {
    "mm": 0.1,
    "cm": 1.0,
    "m": 100.0,
    "km": 1.0e5,
    "in": 2.54,
    "ft": 30.48,
    "mi": 1.609344e5,
}
#: The same for angles, which ROOT stores in degrees.
ANGLE_UNITS = {"deg": 1.0, "rad": 57.29577951308232, "mrad": 0.05729577951308232}
#: And for a density, which ``TGeoMaterial`` keeps in g/cm3.
DENSITY_UNITS = {"g/cm3": 1.0, "kg/m3": 1.0e-3, "g/m3": 1.0e-6}


class GdmlError(ValueError):
    """The file is not one ROOT's reader accepts: the message says what it lacks."""


def _tag(element: ET.Element) -> str:
    """An element's name without the schema's ``{namespace}`` prefix."""
    return element.tag.rsplit("}", 1)[-1]


def _scaled(element: ET.Element, name: str, units: dict[str, float], default: str) -> float:
    """One attribute as a number, times its ``unit`` (``default`` when it states none)."""
    text = element.get(name)
    if text is None:
        raise GdmlError(f"<{_tag(element)}> has no '{name}' attribute")
    #: A length container states ``unit`` or ``lunit``, an angle one ``aunit`` or ``unit``.
    stated = ("aunit", "unit") if units is ANGLE_UNITS else ("unit", "lunit")
    unit = next((element.get(key) for key in stated if element.get(key) is not None), default)
    if unit not in units:
        raise GdmlError(f"<{_tag(element)}> states the unknown unit {unit!r}")
    return float(text) * units[unit]


def _triple(element: ET.Element, units: dict[str, float], default: str,
            names: tuple[str, str, str] = ("x", "y", "z")) -> tuple[float, float, float]:
    return tuple(_scaled(element, name, units, default) for name in names)  # type: ignore[return-value]


@dataclass(frozen=True)
class Element:
    """A chemical element: what ``TGeoElement`` is made from."""

    name: str
    formula: str
    z: int
    a: float


@dataclass(frozen=True)
class Material:
    """A material: a density in g/cm3 and its parts, each a fraction by name."""

    name: str
    density: float
    parts: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class Solid:
    """A shape: ``shape`` names the GDML solid, ``params`` its numbers in cm and degrees.

    A ``subtraction`` states its two operands' names in ``operands`` and, in
    ``params``, the second one's position and rotation relative to the first.
    """

    name: str
    shape: str
    params: tuple[float, ...] = ()
    operands: tuple[str, ...] = ()


@dataclass(frozen=True)
class Placement:
    """One ``<physvol>``: the volume placed, where, and turned how."""

    volume: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Volume:
    """A volume: its solid's and material's names, and the volumes placed inside it."""

    name: str
    solid: str
    material: str
    placements: list[Placement] = field(default_factory=list)


@dataclass(frozen=True)
class Counts:
    """What ``TGeoManager::CountLevels`` counts once the tree is built."""

    nodes: int
    volumes: int
    max_level: int
    levels: tuple[int, ...]


@dataclass
class Geometry:
    """A whole file: the tables, the volumes, and the volume the tree hangs from."""

    elements: dict[str, Element] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    solids: dict[str, Solid] = field(default_factory=dict)
    volumes: dict[str, Volume] = field(default_factory=dict)
    positions: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    rotations: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    world: str = ""

    def counts(self) -> Counts:
        """What ``TGeoManager::CountLevels`` counts: the nodes of the tree, level by level."""
        if self.world not in self.volumes:
            raise GdmlError(f"the <setup> names the volume {self.world!r}, which the file "
                            f"does not define")
        levels: dict[int, int] = {}
        stack = [(self.world, 0)]
        while stack:
            name, level = stack.pop()
            levels[level] = levels.get(level, 0) + 1
            placed = self.volumes.get(name)
            if placed is None:
                raise GdmlError(f"a <physvol> places the volume {name!r}, which the file "
                                f"does not define")
            stack.extend((child.volume, level + 1) for child in placed.placements)
        return Counts(nodes=sum(levels.values()), volumes=len(self.volumes),
                      max_level=max(levels),
                      levels=tuple(levels[index] for index in range(max(levels) + 1)))


def read(path: str | Path) -> Geometry:
    """Every table of the GDML file at ``path``, its volumes and the world volume."""
    geometry = Geometry()
    root = ET.parse(str(path)).getroot()
    for section in root:
        _section(geometry, section)
    if not geometry.world:
        raise GdmlError(f"{path} has no <setup> naming the world volume")
    return geometry


def _section(geometry: Geometry, section: ET.Element) -> None:
    """Read one top-level section: a table, or the solid definitions."""
    reader = _SECTIONS.get(_tag(section))
    if reader is None:
        return
    reader(geometry, section)


def _define(geometry: Geometry, section: ET.Element) -> None:
    for element in section:
        name = element.get("name", "")
        if _tag(element) == "position":
            geometry.positions[name] = _triple(element, LENGTH_UNITS, "cm")
        elif _tag(element) == "rotation":
            geometry.rotations[name] = _triple(element, ANGLE_UNITS, "deg",
                                               ("x", "y", "z"))


def _element(geometry: Geometry, element: ET.Element) -> None:
    atom = element.find("atom")
    mass = 1.0 if atom is None else float(atom.get("value", "1"))
    geometry.elements[element.get("name", "")] = Element(
        name=element.get("name", ""), formula=element.get("formula", ""),
        z=int(element.get("Z", "0")), a=mass)


def _fraction(part: ET.Element) -> tuple[str, float]:
    return part.get("ref", ""), float(part.get("n", part.get("value", "0")))


def _materials(geometry: Geometry, section: ET.Element) -> None:
    """``<element>`` and ``<material>`` share the section; ``<composite>`` mixes materials."""
    for material in section:
        if _tag(material) == "element":
            _element(geometry, material)
            continue
        density = material.find("D")
        if density is None:
            raise GdmlError(f"the material {material.get('name', '')!r} states no <D> density")
        parts = tuple(_fraction(part) for part in material
                      if _tag(part) in ("fraction", "composite", "materialref"))
        geometry.materials[material.get("name", "")] = Material(
            name=material.get("name", ""),
            density=_scaled(density, "value", DENSITY_UNITS, "g/cm3"),
            parts=parts)


#: Each shape's parameters in ROOT's constructor order, with the value GDML gives
#: one the file leaves out: a tube with no inner radius is solid, and a
#: segment with no angles goes right round.
_SHAPES = {
    "box": (("x", None, LENGTH_UNITS), ("y", None, LENGTH_UNITS), ("z", None, LENGTH_UNITS)),
    "tube": (("rmin", 0.0, LENGTH_UNITS), ("rmax", None, LENGTH_UNITS),
             ("z", None, LENGTH_UNITS), ("startphi", 0.0, ANGLE_UNITS),
             ("deltaphi", 360.0, ANGLE_UNITS)),
    "cone": (("rmin1", 0.0, LENGTH_UNITS), ("rmax1", None, LENGTH_UNITS),
             ("rmin2", 0.0, LENGTH_UNITS), ("rmax2", None, LENGTH_UNITS),
             ("z", None, LENGTH_UNITS), ("startphi", 0.0, ANGLE_UNITS),
             ("deltaphi", 360.0, ANGLE_UNITS)),
    "trd": (("x1", None, LENGTH_UNITS), ("x2", None, LENGTH_UNITS), ("y1", None, LENGTH_UNITS),
            ("y2", None, LENGTH_UNITS), ("z", None, LENGTH_UNITS)),
    "sphere": (("rmin", 0.0, LENGTH_UNITS), ("rmax", None, LENGTH_UNITS),
               ("startphi", 0.0, ANGLE_UNITS), ("deltaphi", 360.0, ANGLE_UNITS),
               ("starttheta", 0.0, ANGLE_UNITS), ("deltatheta", 180.0, ANGLE_UNITS)),
}


def _given(solid: ET.Element, key: str, default: float | None,
           units: dict[str, float]) -> float:
    """One of a shape's numbers, or GDML's default when the file states none."""
    if solid.get(key) is None and default is not None:
        return default
    return _scaled(solid, key, units, "mm" if units is LENGTH_UNITS else "deg")


def _shape(solid: ET.Element, geometry: Geometry) -> Solid:
    shape = _tag(solid)
    name = solid.get("name", "")
    if shape == "subtraction":
        return Solid(name=name, shape=shape, params=_operand_place(solid, geometry),
                     operands=(_ref(solid, "first"), _ref(solid, "second")))
    return Solid(name=name, shape=shape, params=tuple(
        _given(solid, key, default, units)
        for key, default, units in _SHAPES.get(shape, ())))


def _ref(solid: ET.Element, tag: str) -> str:
    """The solid a subtraction's ``<first>``/``<second>`` names."""
    operand = solid.find(tag)
    if operand is None or operand.get("ref") is None:
        raise GdmlError(f"a <subtraction> states no <{tag} ref=...>")
    return operand.get("ref", "")


def _operand_place(solid: ET.Element, geometry: Geometry) -> tuple[float, ...]:
    """The second operand's position, then its rotation's angles, in the first's frame."""
    origin = _at(solid, geometry.positions, "position", LENGTH_UNITS)
    angles = _at(solid, geometry.rotations, "rotation", ANGLE_UNITS)
    return origin + angles


def _solids(geometry: Geometry, section: ET.Element) -> None:
    for solid in section:
        built = _shape(solid, geometry)
        geometry.solids[built.name] = built


def _placements(volume: ET.Element, geometry: Geometry) -> list[Placement]:
    """Every ``<physvol>`` in the volume, at its named or inline placement."""
    found = []
    for physvol in volume:
        if _tag(physvol) != "physvol":
            continue
        found.append(Placement(volume=_named(physvol, "volumeref"),
                               position=_at(physvol, geometry.positions, "position", LENGTH_UNITS),
                               rotation=_at(physvol, geometry.rotations, "rotation", ANGLE_UNITS)))
    return found


def _named(physvol: ET.Element, tag: str) -> str:
    child = physvol.find(tag)
    if child is None:
        raise GdmlError(f"a <physvol> has no <{tag}>")
    return child.get("ref", "")


def _at(physvol: ET.Element, table: dict[str, tuple[float, float, float]], tag: str,
        units: dict[str, float]) -> tuple[float, float, float]:
    """Where a ``<physvol>`` puts the volume: a named entry, or one stated inline."""
    inline = physvol.find(tag)
    if inline is None:
        inline = physvol.find(f"{tag}ref")
    if inline is None:
        return (0.0, 0.0, 0.0)
    if inline.get("ref") is None:
        return _triple(inline, units, "mm" if units is LENGTH_UNITS else "deg")
    if inline.get("ref") not in table:
        raise GdmlError(f"a <physvol> names the {tag} {inline.get('ref')!r}, which "
                        f"<define> does not state")
    return table[inline.get("ref", "")]


def _volumes(geometry: Geometry, section: ET.Element) -> None:
    for volume in section:
        name = volume.get("name", "")
        geometry.volumes[name] = Volume(
            name=name, solid=_named(volume, "solidref"),
            material=_named(volume, "materialref"), placements=_placements(volume, geometry))


def _setup(geometry: Geometry, section: ET.Element) -> None:
    world = section.find("world")
    geometry.world = "" if world is None else world.get("ref", "")


_SECTIONS = {
    "define": _define,
    "materials": _materials,
    "solids": _solids,
    "structure": _volumes,
    "setup": _setup,
}


