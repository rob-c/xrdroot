"""A ``TGeo`` geometry's tree: volumes placed in volumes, walked, and what a drawing paints.

The tree is a small version of ROOT's ``assembly.C`` and ``rootgeom.C``:
a world, a container, the leaves in it, one assembly.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.geom.painting import solids


def world() -> dict[str, Any]:
    """A world with a box in it holding two leaves, and an assembly holding one more."""
    geom = ROOT.TGeoManager("world", "a small world")
    medium = ROOT.TGeoMedium("Vacuum", 1, ROOT.TGeoMaterial("Vacuum", 0, 0, 0))
    top = geom.MakeBox("TOP", medium, 100, 100, 100)
    geom.SetTopVolume(top)
    box = geom.MakeBox("BOX", medium, 50, 50, 50)
    leaf = geom.MakeTube("LEAF", medium, 0, 5, 10)
    cone = geom.MakeCone("CONE", medium, 5, 0, 1, 0, 2)
    group = ROOT.TGeoVolumeAssembly("GROUP")
    box.AddNode(leaf, 1, ROOT.TGeoTranslation(10, 0, 0))
    box.AddNodeOverlap(leaf, 2, ROOT.TGeoTranslation(-10, 0, 0))
    group.AddNode(cone, 1)
    top.AddNode(box, 1)
    top.AddNode(group, 1, ROOT.TGeoTranslation(0, 80, 0))
    geom.CloseGeometry()
    return {"geom": geom, "top": top, "box": box, "leaf": leaf, "cone": cone, "group": group}


def names(top: Any) -> list[str]:
    return [solid.name for solid in solids(top)]


def test_the_tree_is_walked_and_counted_as_root_counts_it() -> None:
    made = world()
    top, box = made["top"], made["box"]
    expect((top.GetNtotal(), 6), (top.GetNdaughters(), 2), (box.GetNode(1).GetName(), "LEAF_2"),
           (box.GetNode("LEAF_1").GetNumber(), 1), (box.GetNode(5), None),
           (box.GetNode(0).GetMotherVolume(), box), (box.GetNode(1).IsOverlapping(), True),
           (top.GetNode(1).GetDaughter(0).GetVolume(), made["cone"]),
           (made["group"].IsAssembly(), True), (top.GetNode(0).GetNdaughters(), 2))  # fmt: skip


def test_the_leaves_are_painted_by_default_and_every_level_with_containers_shown() -> None:
    made = world()
    assert names(made["top"]) == ["LEAF", "LEAF", "CONE"]
    made["geom"].SetVisOption(0)
    assert names(made["top"]) == ["BOX", "LEAF", "LEAF", "CONE"]
    made["geom"].SetVisLevel(1)
    assert names(made["top"]) == ["BOX"]


def test_the_top_is_painted_when_it_is_shown_or_is_all_there_is() -> None:
    made = world()
    made["geom"].SetTopVisible()
    assert names(made["top"])[0] == "TOP"
    made["geom"].SetTopVisible(False)
    made["top"].SetVisOnly()
    assert names(made["top"]) == ["TOP"]  # only the volume drawn is asked for
    assert names(made["leaf"]) == ["LEAF"]


def test_hidden_volumes_and_hidden_daughters_are_not_painted() -> None:
    made = world()
    made["box"].VisibleDaughters(False)
    made["top"].SetVisLeaves()
    assert names(made["top"]) == ["BOX", "CONE"]
    made["top"].InvisibleAll()
    expect((names(made["top"]), []), (made["leaf"].IsVisible(), False))
    made["top"].SetVisContainers()
    assert made["geom"].GetVisOption() == 0


def test_a_volume_is_as_transparent_as_it_says_or_as_its_material() -> None:
    made = world()
    made["leaf"].GetMaterial().SetTransparency(30)
    expect((made["leaf"].GetTransparency(), 30), (made["group"].GetMaterial(), None))
    made["leaf"].SetTransparency(70)
    assert made["leaf"].GetTransparency() == 70


def test_the_placements_above_a_leaf_put_it_in_the_world() -> None:
    made = world()
    cone = solids(made["top"])[2]
    assert cone.mesh.extent()[0].tolist() == pytest.approx([-2.0, 78.0, -5.0])


def test_an_iterator_walks_every_node_and_skips_what_it_is_told_to() -> None:
    made = world()
    walker = ROOT.TGeoIterator(made["top"])
    seen = []
    for node in walker:
        seen.append((node.GetName(), walker.GetLevel()))
        if node.GetName() == "BOX_1":
            walker.Skip()
    path = ROOT.TString()
    walker.GetPath(path)
    walker.Reset()
    expect((seen, [("BOX_1", 1), ("GROUP_1", 1), ("CONE_1", 2)]), (str(path),
           "/TOP_1/GROUP_1/CONE_1"),
           (walker().GetName(), "BOX_1"), (walker.GetNode(1).GetName(), "BOX_1"),
           (walker.GetNode(0), None), (walker.GetTopVolume(), made["top"]),
           (walker.GetCurrentMatrix().is_identity(), True))  # fmt: skip


class Recolour(ROOT.TGeoIteratorPlugin):
    """A plugin colouring the volumes under one path, as ``runplugin.C``'s does."""

    def ProcessNode(self) -> None:
        cell = type("Cell", (), {"value": ""})()
        self.fIterator.GetPath(cell)
        if "BOX_1" in cell.value:
            level = self.fIterator.GetLevel()
            self.fIterator.GetNode(level).GetVolume().SetLineColor(ROOT.kGreen)


def test_a_plugin_recolours_what_it_is_shown_for_that_painting_only() -> None:
    made = world()
    made["geom"].GetGeomPainter().SetIteratorPlugin(Recolour())
    made["top"].Draw()
    painted = solids(made["top"])
    expect(([s.line_color for s in painted], [ROOT.kGreen, ROOT.kGreen, 1]),
           (made["leaf"].GetLineColor(), 1))
    base = ROOT.TGeoIteratorPlugin()
    base.ProcessNode()
    base.SetIterator(None)
    made["geom"].GetGeomPainter().ModifiedPad()
    made["geom"].GetGeomPainter().SetRaytracing(True)


def test_the_manager_finds_what_it_lists_by_name_or_number() -> None:
    made = world()
    geom = made["geom"]
    expect((geom.GetVolume("BOX"), made["box"]), (geom.GetVolume(99), None),
           (geom.GetShape("LEAF"), made["leaf"].GetShape()), (geom.GetMasterVolume(), made["top"]),
           (geom.GetListOfVolumes().GetEntries(), 5), (geom.GetListOfShapes().GetEntries(), 4),
           (geom.GetListOfMedia().GetEntries(), 1), (geom.GetListOfMatrices().GetEntries(), 0),
           (ROOT.gROOT.FindObject("world"), geom))  # fmt: skip
    geom.AddTransformation(ROOT.TGeoTranslation(1, 0, 0))
    geom.SetNsegments(2)
    geom.SetMaxVisNodes(5)
    expect((geom.GetListOfMatrices().GetEntries(), 1), (geom.GetNsegments(), 3),
           (geom.IsTopVisible(), False))  # fmt: skip


def test_default_colours_follow_each_volume_s_material() -> None:
    made = world()
    made["geom"].DefaultColors()
    assert made["leaf"].GetLineColor() == 2 + 1 % 6


def test_closing_a_geometry_with_no_top_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    geom = ROOT.TGeoManager("empty", "nothing")
    geom.CloseGeometry()
    assert "Error in <TGeoManager::CloseGeometry>: you have to define the top volume first" in (
        capsys.readouterr().err)
    assert not geom.IsClosed()


def test_the_current_geometry_is_there_only_once_one_is_made() -> None:
    assert not ROOT.gGeoManager
    with pytest.raises(AttributeError, match="gGeoManager is null"):
        ROOT.gGeoManager.GetTopVolume()
    material = ROOT.TGeoMaterial("m", 1, 1, 1)  # makes the default geometry, as ROOT's does
    expect((bool(ROOT.gGeoManager), True), (ROOT.gGeoManager == ROOT.gGeoManager.GetTopVolume(),
           False),
           (ROOT.gGeoManager.GetName(), "Geometry"), (material.GetIndex(), 0),
           (hash(ROOT.gGeoManager) == id(ROOT.gGeoManager), True))  # fmt: skip


@pytest.mark.parametrize(
    ("call", "said"),
    [(lambda g: g.CreateParallelWorld("p"), "CreateParallelWorld needs ROOT's geometry navigator"),
     (lambda g: g.MakePhysicalNode("/TOP_1"), "MakePhysicalNode"),
     (lambda g: g.CheckOverlaps(0.1), "CheckOverlaps"), (lambda g: g.FindNode(0, 0, 0), "FindNode"),
     (lambda g: g.Export("g.gdml"), "'g.gdml' was not written"),
     (lambda g: ROOT.TGeoManager.Import("g.gdml"), "'g.gdml' was not read"),
     (lambda g: g.GetElementTable(), "RadioNuclides.txt"),
     (lambda g: g.GetTopVolume().RandomRays(10), "RandomRays tracks rays"),
     (lambda g: g.GetTopVolume().CheckOverlaps(), "CheckOverlaps needs"),
     (lambda g: g.GetTopVolume().Divide("s", 1, 4), "place the slices with AddNode")],
)  # fmt: skip
def test_what_needs_the_navigator_or_other_formats_is_refused_by_name(call: Any, said: str) -> None:
    made = world()
    with pytest.raises(UnsupportedFeatureError, match=said):
        call(made["geom"])


def test_export_precision_is_kept_to_be_asked_back() -> None:
    ROOT.TGeoManager.SetExportPrecision(8)
    assert ROOT.TGeoManager.GetExportPrecision() == 8


def test_every_maker_makes_a_volume_of_its_shape_named_as_the_volume() -> None:
    geom = ROOT.TGeoManager("g", "g")
    medium = ROOT.TGeoMedium("m", 1, ROOT.TGeoMaterial("m", 1, 1, 1))
    pgon = geom.MakePgon("PGON", medium, 0, 360, 6, 2)
    xtru = geom.MakeXtru("XTRU", medium, 2)
    expect((type(pgon.GetShape()).__name__, "TGeoPgon"), (pgon.GetShape().GetName(), "PGON"),
           (xtru.GetShape().GetNz(), 2.0), (pgon.GetMedium(), medium),
           (type(geom.MakeSphere("S", medium, 0, 1).GetShape()).__name__,
            "TGeoSphere"))  # fmt: skip
    volume = ROOT.TGeoVolume("V", ROOT.TGeoBBox(1, 1, 1))
    volume.SetShape(ROOT.TGeoTube(1, 2, 3))
    volume.SetMedium(medium)
    expect((type(volume.GetShape()).__name__, "TGeoTube"), (volume.GetMaterial().GetName(), "m"))


def test_drawing_clears_the_pad_unless_told_same_and_makes_a_view(tmp_path: Any) -> None:
    made = world()
    ROOT.TLine(0, 0, 1, 1).Draw()
    made["box"].Draw()
    pad = ROOT.gPad.GetCanvas()
    expect((len(pad.primitives), 1), (pad.GetView().IsPerspective(), True))
    made["leaf"].GetShape().Draw("same")
    made["top"].GetNode(0).Draw("same")
    expect((len(pad.primitives), 3), (made["geom"].GetGeomPainter().top, made["box"]))
    made["leaf"].Raytrace()
    pad.SaveAs(str(tmp_path / "drawn.png"))
    assert (tmp_path / "drawn.png").stat().st_size > 0
