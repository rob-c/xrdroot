"""``TGeoManager::Import``: a GDML file made into a geometry, as ROOT's ``TGDMLParse`` makes it.

The file read here is the shape of ROOT's own ``roottest/root/io/geo``
geometry in miniature: named placements, an element and a mixture of it,
every solid kind xrdroot's GDML reader knows, and placements of all four
matrix kinds. What ROOT says as it reads and closes is compared line for
line, since a test of ROOT's reads the same lines.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.geom import TGeoCombiTrans, TGeoMixture, TGeoRotation, TGeoTranslation

DOCUMENT = """<?xml version="1.0" encoding="UTF-8" ?>
<gdml>
 <define>
  <position name="posUp" unit="cm" x="0" y="0" z="10"/>
  <rotation name="rX" unit="deg" x="90" y="0" z="0"/>
 </define>
 <materials>
  <element name="argon" formula="Ar" Z="18"><atom value="39.948"/></element>
  <material name="LAr" formula="Ar">
   <D value="1.396" unit="g/cm3"/>
   <fraction n="1.0" ref="argon"/>
  </material>
  <material name="Mix" formula="Mix">
   <D value="2.0" unit="g/cm3"/>
   <composite n="0.5" ref="LAr"/>
   <composite n="0.5" ref="argon"/>
  </material>
 </materials>
 <solids>
  <box name="box1" lunit="cm" x="2" y="4" z="6"/>
  <tube name="tube1" lunit="cm" rmax="5" z="3"/>
  <tube name="tube2" lunit="cm" rmin="1" rmax="5" z="3" startphi="10" deltaphi="90"/>
  <cone name="cone1" lunit="cm" rmax1="5" rmax2="2" z="3"/>
  <cone name="cone2" lunit="cm" rmax1="5" rmax2="2" z="3" startphi="0" deltaphi="90"/>
  <trd name="trd1" lunit="cm" x1="2" x2="4" y1="3" y2="5" z="6"/>
  <sphere name="sphere1" lunit="cm" rmax="4"/>
  <subtraction name="cut">
   <first ref="box1"/>
   <second ref="tube1"/>
   <positionref ref="posUp"/>
   <rotationref ref="rX"/>
  </subtraction>
  <subtraction name="cut2">
   <first ref="box1"/>
   <second ref="tube1"/>
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
    <position unit="cm" x="1" y="2" z="3"/>
   </physvol>
   <physvol>
    <volumeref ref="inner"/>
    <rotation unit="deg" x="0" y="0" z="45"/>
   </physvol>
   <physvol><volumeref ref="inner"/></physvol>
  </volume>
  <volume name="inner"><solidref ref="cut"/><materialref ref="Mix"/></volume>
  <volume name="cutof"><solidref ref="cut2"/><materialref ref="LAr"/></volume>
  <volume name="sphere"><solidref ref="sphere1"/><materialref ref="LAr"/></volume>
  <volume name="cones"><solidref ref="cone1"/><materialref ref="LAr"/></volume>
  <volume name="conesseg"><solidref ref="cone2"/><materialref ref="LAr"/></volume>
  <volume name="trds"><solidref ref="trd1"/><materialref ref="LAr"/></volume>
  <volume name="tubes"><solidref ref="tube2"/><materialref ref="LAr"/></volume>
 </structure>
 <setup name="Default"><world ref="world"/></setup>
</gdml>
"""


def written(tmp_path: Path, text: str = DOCUMENT, name: str = "geometry.gdml") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_importing_a_gdml_file_says_what_root_says_reading_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    geometry = ROOT.TGeoManager.Import(str(written(tmp_path)))
    said = capsys.readouterr().err.splitlines()
    expect((said[0], f"Info in <TGeoManager::Import>: Reading geometry from file: "
                     f"{tmp_path / 'geometry.gdml'}"),
           (said[1], "Info in <TGeoManager::TGeoManager>: Geometry GDMLImport, Geometry "
                     "imported from GDML created"),
           (said[2], "Info in <TGeoManager::SetTopVolume>: Top volume is world. Master volume "
                     "is world"),
           (said[3], "Info in <TGeoNavigator::BuildCache>: --- Maximum geometry depth set to "
                     "100"),
           (said[-2], "Info in <TGeoManager::CloseGeometry>: 5 nodes/ 8 volume UID's in "
                      "Geometry imported from GDML"),
           (said[-1], "Info in <TGeoManager::CloseGeometry>: ----------------modeler ready"
                      "----------------"),
           (said[-3], "Info in <TGeoManager::CountLevels>: max level = 1, max placements = 4"),
           (geometry.GetTitle(), "Geometry imported from GDML"),
           (geometry.GetTopVolume().GetName(), "world"),
           (geometry.IsClosed(), True))  # fmt: skip


def test_importing_makes_the_geometry_current_as_gcurrent_do_does(tmp_path: Path) -> None:
    geometry = ROOT.TGeoManager.Import(str(written(tmp_path)))
    expect((ROOT.gGeoManager == geometry, True),
           (ROOT.gGeoManager.GetTopVolume().GetName(), "world"),
           (ROOT.gGeoManager.GetListOfVolumes().GetEntries(), 8),
           (ROOT.gGeoManager.GetListOfShapes().GetEntries(), 9))  # fmt: skip


def test_the_setup_names_the_geometry_as_root_names_it(tmp_path: Path) -> None:
    """ROOT names the manager after the file's ``<setup>``, whatever ``Import`` is passed."""
    geometry = ROOT.TGeoManager.Import(str(written(tmp_path)), "dune")
    expect((geometry.GetName(), "Default"), (geometry.GetTitle(), "Geometry imported from GDML"))


def test_every_solid_kind_is_made_as_root_makes_it(tmp_path: Path) -> None:
    ROOT.TGeoManager.Import(str(written(tmp_path)))
    shapes = ROOT.gGeoManager.GetListOfShapes()
    made = {(name := shapes.At(index).GetName()): type(shapes.At(index)).__name__
            for index in range(shapes.GetEntries())}
    expect((made["box1"], "TGeoBBox"), (made["tube1"], "TGeoTube"),
           (made["tube2"], "TGeoTubeSeg"), (made["cone1"], "TGeoCone"),
           (made["cone2"], "TGeoConeSeg"), (made["trd1"], "TGeoTrd2"),
           (made["sphere1"], "TGeoSphere"), (made["cut"], "TGeoCompositeShape"),
           (name, "cut2"))  # fmt: skip


def test_a_subtraction_places_its_second_operand_where_the_file_puts_it(tmp_path: Path) -> None:
    ROOT.TGeoManager.Import(str(written(tmp_path)))
    placed = ROOT.gGeoManager.GetShape("cut").GetExpression()
    expect((placed, "box1-tube1:cut_operand"),
           (ROOT.gGeoManager.GetShape("cut2").GetExpression(), "box1-tube1"),
           (ROOT.gGeoManager.GetShape("cut").GetExpression() != placed, False))  # fmt: skip


def test_a_material_is_one_element_or_a_mixture_of_parts(tmp_path: Path) -> None:
    ROOT.TGeoManager.Import(str(written(tmp_path)))
    argon = ROOT.gGeoManager.GetMaterial("LAr")
    mixture = ROOT.gGeoManager.GetMaterial("Mix")
    expect((argon.GetA(), 39.948), (argon.GetZ(), 18.0), (argon.GetDensity(), 1.396),
           (isinstance(mixture, TGeoMixture), True), (mixture.GetNelements(), 2),
           (mixture.GetDensity(), 2.0), (mixture.GetA(), 39.948))  # fmt: skip


def test_a_placement_is_a_matrix_of_the_kind_the_file_states(tmp_path: Path) -> None:
    ROOT.TGeoManager.Import(str(written(tmp_path)))
    world = ROOT.gGeoManager.GetTopVolume()
    expect((type(world.GetNode(0).GetMatrix()), TGeoCombiTrans),
           (type(world.GetNode(1).GetMatrix()), TGeoTranslation),
           (type(world.GetNode(2).GetMatrix()), TGeoRotation),
           (world.GetNode(3).GetMatrix().IsIdentity(), True),
           (world.GetNode(1).GetMatrix().GetTranslation().tolist(), [1.0, 2.0, 3.0]),
           (tuple(world.GetNode(0).GetMatrix().GetTranslation()), (0.0, 0.0, 10.0)))  # fmt: skip


def test_a_solid_kind_the_reader_does_not_build_is_refused_by_name(tmp_path: Path) -> None:
    document = DOCUMENT.replace(
        '<box name="box1" lunit="cm" x="2" y="4" z="6"/>',
        '<box name="box1" lunit="cm" x="2" y="4" z="6"/>\n'
        '  <ellipsoid name="ell1" lunit="cm" ax="2" by="3" cz="4"/>',
    ).replace('<solidref ref="sphere1"/>', '<solidref ref="ell1"/>')
    with pytest.raises(UnsupportedFeatureError, match="<ellipsoid>, which xrdroot's GDML"):
        ROOT.TGeoManager.Import(str(written(tmp_path, document)))


def test_a_mixtures_part_the_file_never_defines_is_refused_by_name(tmp_path: Path) -> None:
    document = DOCUMENT.replace('<composite n="0.5" ref="argon"/>',
                                '<composite n="0.5" ref="nothing"/>')
    with pytest.raises(ValueError, match="part 'nothing', which the file defines as neither"):
        ROOT.TGeoManager.Import(str(written(tmp_path, document)))


def test_the_world_volume_is_one_the_file_defines(tmp_path: Path) -> None:
    document = DOCUMENT.replace('<world ref="world"/>', '<world ref="nowhere"/>')
    with pytest.raises(ValueError, match="names the volume 'nowhere'"):
        ROOT.TGeoManager.Import(str(written(tmp_path, document)))


def test_a_file_that_is_neither_gdml_nor_a_root_file_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "geometry.geo"
    path.write_text(DOCUMENT)
    with pytest.raises(UnsupportedFeatureError, match="neither a GDML document nor a ROOT"):
        ROOT.TGeoManager.Import(str(path))
