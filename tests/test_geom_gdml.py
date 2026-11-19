"""``xrdroot.geom.gdml``: the tables and the tree of a GDML file, as ``TGDMLParse`` reads them.

The numbers are ROOT's: ``CountLevels``' node count, level by level, and the
shape parameters the file states in its own ``lunit``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xrdroot.geom import gdml

#: A file using every table: an element, a material, a box, a tube with its
#: defaults left out, a subtraction, named and inline placements, two levels.
DOCUMENT = """<?xml version="1.0"?>
<gdml xmlns:gdml="http://cern.ch/2001/Schemas/GDML">
 <define>
  <position name="posUp" unit="cm" x="0" y="0" z="10"/>
  <rotation name="rX" unit="deg" x="90" y="0" z="0"/>
  <constant name="notAPlacement" value="1"/>
 </define>
 <materials>
  <element name="argon" formula="Ar" Z="18"><atom value="39.948"/></element>
  <material name="LAr" formula="LAr">
   <D value="1.396" unit="g/cm3"/>
   <fraction n="1.0" ref="argon"/>
  </material>
  <material name="Mixture" formula="Mixture">
   <D value="2.0" unit="g/cm3"/>
   <composite n="0.5" ref="LAr"/>
  </material>
 </materials>
 <solids>
  <box name="box1" lunit="cm" x="2" y="4" z="6"/>
  <tube name="tube1" lunit="cm" rmax="5" z="3"/>
  <tube name="tube2" lunit="mm" rmin="1" rmax="5" z="3" startphi="10" deltaphi="90"/>
  <subtraction name="cut">
   <first ref="box1"/>
   <second ref="tube1"/>
   <positionref ref="posUp"/>
   <rotationref ref="rX"/>
  </subtraction>
 </solids>
 <structure>
  <volume name="world">
   <solidref ref="box1"/>
   <materialref ref="LAr"/>
   <physvol>
    <volumeref ref="inner"/>
    <positionref ref="posUp"/>
    <rotationref ref="rX"/>
   </physvol>
   <physvol>
    <volumeref ref="inner"/>
    <position unit="m" x="1" y="2" z="3"/>
    <rotation unit="deg" x="0" y="0" z="45"/>
   </physvol>
   <physvol><volumeref ref="inner"/></physvol>
  </volume>
  <volume name="inner">
   <solidref ref="cut"/>
   <materialref ref="LAr"/>
  </volume>
 </structure>
 <setup name="Default"><world ref="world"/></setup>
</gdml>
"""


def written(tmp_path: Path, text: str = DOCUMENT, name: str = "geometry.gdml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_every_table_of_the_file_is_read(tmp_path: Path) -> None:
    geometry = gdml.read(written(tmp_path))
    assert geometry.elements["argon"] == gdml.Element("argon", "Ar", 18, 39.948)
    assert geometry.materials["LAr"] == gdml.Material("LAr", 1.396, (("argon", 1.0),))
    assert geometry.materials["Mixture"].parts == (("LAr", 0.5),)
    assert geometry.positions["posUp"] == (0.0, 0.0, 10.0)
    assert geometry.rotations["rX"] == (90.0, 0.0, 0.0)
    assert geometry.world == "world"


def test_a_solid_is_read_in_the_unit_the_file_states(tmp_path: Path) -> None:
    geometry = gdml.read(written(tmp_path))
    assert geometry.solids["box1"].params == (2.0, 4.0, 6.0)
    assert geometry.solids["tube2"].params == pytest.approx((0.1, 0.5, 0.3, 10.0, 90.0))


def test_a_tube_with_no_inner_radius_is_solid_and_goes_right_round(tmp_path: Path) -> None:
    geometry = gdml.read(written(tmp_path))
    assert geometry.solids["tube1"].params == (0.0, 5.0, 3.0, 0.0, 360.0)


def test_a_subtraction_names_its_operands_and_the_second_ones_place(tmp_path: Path) -> None:
    geometry = gdml.read(written(tmp_path))
    assert geometry.solids["cut"].operands == ("box1", "tube1")
    assert geometry.solids["cut"].params == (0.0, 0.0, 10.0, 90.0, 0.0, 0.0)


def test_a_placement_is_named_or_stated_in_place(tmp_path: Path) -> None:
    geometry = gdml.read(written(tmp_path))
    placed = geometry.volumes["world"].placements
    assert placed[0] == gdml.Placement("inner", (0.0, 0.0, 10.0), (90.0, 0.0, 0.0))
    assert placed[1] == gdml.Placement("inner", (100.0, 200.0, 300.0), (0.0, 0.0, 45.0))
    assert placed[2] == gdml.Placement("inner")


def test_the_counts_are_the_nodes_of_the_tree_level_by_level(tmp_path: Path) -> None:
    counts = gdml.read(written(tmp_path)).counts()
    assert (counts.nodes, counts.volumes, counts.max_level) == (4, 2, 1)
    assert counts.levels == (1, 3)


def test_a_file_without_a_setup_names_no_world_volume(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<setup name="Default"><world ref="world"/></setup>', "")
    with pytest.raises(gdml.GdmlError, match="has no <setup>"):
        gdml.read(written(tmp_path, text))


def test_a_setup_naming_a_volume_the_file_lacks_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<world ref="world"/>', '<world ref="nowhere"/>')
    with pytest.raises(gdml.GdmlError, match="names the volume 'nowhere'"):
        gdml.read(written(tmp_path, text)).counts()


def test_a_physvol_placing_a_volume_the_file_lacks_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<volumeref ref="inner"/>', '<volumeref ref="ghost"/>', 1)
    with pytest.raises(gdml.GdmlError, match="places the volume 'ghost'"):
        gdml.read(written(tmp_path, text)).counts()


def test_a_physvol_with_no_volumeref_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace("<physvol><volumeref ref=\"inner\"/></physvol>", "<physvol/>")
    with pytest.raises(gdml.GdmlError, match="has no <volumeref>"):
        gdml.read(written(tmp_path, text))


def test_a_placement_naming_a_table_entry_the_file_lacks_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<positionref ref="posUp"/>', '<positionref ref="posDown"/>')
    with pytest.raises(gdml.GdmlError, match="names the position 'posDown'"):
        gdml.read(written(tmp_path, text))


def test_a_solid_with_no_attribute_for_a_parameter_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<box name="box1" lunit="cm" x="2" y="4" z="6"/>',
                            '<box name="box1" lunit="cm" y="4" z="6"/>')
    with pytest.raises(gdml.GdmlError, match="<box> has no 'x' attribute"):
        gdml.read(written(tmp_path, text))


def test_a_unit_the_reader_does_not_know_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('unit="cm" x="0" y="0" z="10"', 'unit="furlong" x="0" y="0" z="10"')
    with pytest.raises(gdml.GdmlError, match="unknown unit 'furlong'"):
        gdml.read(written(tmp_path, text))


def test_a_material_stating_no_density_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<D value="1.396" unit="g/cm3"/>', "")
    with pytest.raises(gdml.GdmlError, match="states no <D> density"):
        gdml.read(written(tmp_path, text))


def test_a_subtraction_with_no_second_operand_is_refused(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<second ref="tube1"/>', "")
    with pytest.raises(gdml.GdmlError, match="states no <second ref="):
        gdml.read(written(tmp_path, text))


def test_a_section_the_reader_does_not_know_is_left_alone(tmp_path: Path) -> None:
    text = DOCUMENT.replace("<structure>", "<extension><x/></extension>\n<structure>")
    assert gdml.read(written(tmp_path, text)).world == "world"


def test_a_shape_the_reader_does_not_know_carries_no_parameters(tmp_path: Path) -> None:
    text = DOCUMENT.replace('<box name="box1" lunit="cm" x="2" y="4" z="6"/>',
                            '<box name="box1" lunit="cm" x="2" y="4" z="6"/>\n'
                            '  <trapezoid name="odd" lunit="cm" z="1"/>')
    assert gdml.read(written(tmp_path, text)).solids["odd"].params == ()
