"""GDML written back: ``geom.gdml``'s writer, and ``TGeoManager::Export`` writing it.

A geometry written here is one this library reads again with the same shapes
and placements, and one ROOT's own ``TGDMLParse`` reads, which is the only
reason to write GDML at all: ROOT 6.40.04 reads the files these tests write,
naming each solid by ROOT's class for it (a ``<tube>`` with angles is a
``TGeoTubeSeg``) and placing each node at the matrix it was given.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.geom.gdml import (
    Element,
    GdmlError,
    Geometry,
    Material,
    Placement,
    Solid,
    Volume,
    read,
    text,
    write,
)


def _geometry() -> Geometry:
    """A small geometry with both kinds of placement, a shared volume and a stray one."""
    return Geometry(
        elements={"C": Element(name="C", formula="C", z=6, a=12.011)},
        materials={"Carbon": Material(name="Carbon", density=2.2, parts=(("C", 1.0),))},
        solids={
            "WORLD": Solid(name="WORLD", shape="box", params=(50.0, 50.0, 50.0)),
            "INNER": Solid(name="INNER", shape="tube", params=(0.0, 5.0, 10.0, 0.0, 360.0)),
            "LOST": Solid(name="LOST", shape="trd", params=(1.0, 2.0, 1.0, 2.0, 3.0)),
        },
        volumes={
            "WORLD": Volume(name="WORLD", solid="WORLD", material="Carbon", placements=[
                Placement(volume="INNER", position=(1.0, 0.0, 0.0)),
                Placement(volume="INNER", position=(0.0, 2.0, 0.0),
                          rotation=(30.0, 0.0, 0.0)),
            ]),
            "INNER": Volume(name="INNER", solid="INNER", material="Carbon"),
            "LOST": Volume(name="LOST", solid="LOST", material="Carbon"),
        },
        positions={"far": (0.0, 0.0, 9.0)},
        world="WORLD",
        setup="written",
    )


def test_a_geometry_written_as_gdml_reads_back_with_the_same_shapes_and_places(
    tmp_path: Path,
) -> None:
    written = tmp_path / "written.gdml"
    write(written, _geometry())
    back = read(written)
    assert back.solids == _geometry().solids
    assert back.volumes["WORLD"].placements == _geometry().volumes["WORLD"].placements
    assert back.materials["Carbon"].density == pytest.approx(2.2)
    assert back.elements["C"].z == 6
    assert back.world == "WORLD" and back.setup == "written"


def test_a_volume_the_world_does_not_hold_is_written_after_the_worlds_tree() -> None:
    order = [line for line in text(_geometry()).splitlines() if "<volume name=" in line]
    assert order == ['    <volume name="INNER">', '    <volume name="WORLD">',
                     '    <volume name="LOST">']


def test_a_subtraction_is_written_as_its_two_operands_and_their_placement() -> None:
    geometry = _geometry()
    geometry.solids["CUT"] = Solid(name="CUT", shape="subtraction", operands=("WORLD", "INNER"),
                                   params=(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
    back = read_text(text(geometry))
    assert back.solids["CUT"].operands == ("WORLD", "INNER")
    assert back.solids["CUT"].params == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def read_text(written: str) -> Geometry:
    """A geometry read back out of text, through the file the writer would make of it."""
    with tempfile.TemporaryDirectory() as holding:
        path = Path(holding) / "read.gdml"
        path.write_text(written, encoding="utf-8")
        return read(path)


def test_a_shape_gdml_has_no_element_for_is_refused_by_name() -> None:
    geometry = _geometry()
    geometry.solids["ODD"] = Solid(name="ODD", shape="torus", params=(1.0, 2.0, 3.0))
    with pytest.raises(GdmlError, match="'torus'"):
        text(geometry)


#: The geometry ``TGeoManager::Export`` is given in the tests below.
def _built(manager: Any) -> Any:
    mixture = ROOT.TGeoMixture("Air", 2, 0.0012)
    mixture.AddElement(14.0, 7.0, 0.75)
    mixture.AddElement(16.0, 8.0, 0.25)
    medium = ROOT.TGeoMedium("Air", 1, mixture)
    plain = ROOT.TGeoMedium("Water", 2, ROOT.TGeoMaterial("Water", 1.0, 1.0, 1.0))
    world = manager.MakeBox("WORLD", medium, 100.0, 100.0, 100.0)
    manager.SetTopVolume(world)
    world.AddNode(manager.MakeTubs("TUBE", medium, 0.0, 20.0, 30.0, 0.0, 90.0), 1,
                  ROOT.TGeoTranslation("where", 10.0, 0.0, -5.0))
    world.AddNode(manager.MakeTrd2("BAR", plain, 1.0, 2.0, 3.0, 4.0, 5.0), 1)
    return world


def test_export_writes_the_geometry_as_gdml_which_reads_back(tmp_path: Path) -> None:
    manager = ROOT.TGeoManager("exported", "written by Export")
    _built(manager)
    target = tmp_path / "exported.gdml"
    assert manager.Export(str(target)) == 1
    back = read(target)
    assert sorted(back.volumes) == ["BAR", "TUBE", "WORLD"]
    assert back.solids["TUBE"].params == (0.0, 20.0, 30.0, 0.0, 90.0)
    assert back.solids["BAR"].params == (1.0, 2.0, 3.0, 4.0, 5.0)
    assert back.volumes["WORLD"].placements[0].position == (10.0, 0.0, -5.0)
    assert back.setup == "exported"


def test_export_states_a_mixtures_elements_by_weight(tmp_path: Path) -> None:
    manager = ROOT.TGeoManager("exported", "a mixture")
    _built(manager)
    target = tmp_path / "mixture.gdml"
    manager.Export(str(target))
    back = read(target)
    assert [name for name, _ in back.materials["Air"].parts] == ["Air_el0", "Air_el1"]
    assert back.elements["Air_el0"].z == 7 and back.elements["Air_el0"].a == 14.0


def test_a_solid_gdml_cannot_state_is_refused_by_name() -> None:
    manager = ROOT.TGeoManager("odd", "a torus")
    medium = ROOT.TGeoMedium("Air", 1, ROOT.TGeoMaterial("Air", 14.0, 7.0, 0.0012))
    world = manager.MakeBox("WORLD", medium, 100.0, 100.0, 100.0)
    manager.SetTopVolume(world)
    world.AddNode(manager.MakeTorus("RING", medium, 10.0, 2.0, 1.0), 1)
    with pytest.raises(UnsupportedFeatureError, match=r"TGeoBBox|torus|no solid for"):
        manager.Export("/dev/null.gdml")


def test_a_node_placed_by_a_reflection_is_refused() -> None:
    manager = ROOT.TGeoManager("mirrored", "a reflection")
    medium = ROOT.TGeoMedium("Air", 1, ROOT.TGeoMaterial("Air", 14.0, 7.0, 0.0012))
    world = manager.MakeBox("WORLD", medium, 100.0, 100.0, 100.0)
    manager.SetTopVolume(world)
    mirrored = ROOT.TGeoHMatrix("mirrored")
    mirrored.ReflectX(True)
    world.AddNode(manager.MakeBox("MIRROR", medium, 1.0, 1.0, 1.0), 1, mirrored)
    with pytest.raises(UnsupportedFeatureError, match="reflection"):
        manager.Export("/dev/null.gdml")


def test_exporting_a_geometry_with_no_top_volume_is_refused() -> None:
    manager = ROOT.TGeoManager("bare", "no top volume")
    with pytest.raises(UnsupportedFeatureError, match="no top volume"):
        manager.Export("/dev/null.gdml")


def test_a_root_file_name_is_refused_with_what_to_do_instead() -> None:
    manager = ROOT.TGeoManager("exported", "a root file")
    _built(manager)
    with pytest.raises(UnsupportedFeatureError, match=r"not written.*\.gdml instead"):
        manager.Export("/tmp/exported.export.root")


def test_a_cxx_script_name_is_refused_with_what_to_do_instead() -> None:
    manager = ROOT.TGeoManager("exported", "a script")
    _built(manager)
    with pytest.raises(UnsupportedFeatureError, match=r"\.gdml instead"):
        manager.Export("/tmp/exported.C")
