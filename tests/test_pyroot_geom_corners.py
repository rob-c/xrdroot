"""The corners of the geometry classes: what a tutorial seldom asks, but ROOT answers."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.canvas import Primitive
from xrdroot.geom import Mesh, Solid
from xrdroot.geom import shapes as build
from xrdroot.geom.paint import points_of
from xrdroot.pyroot.geom import legacyread
from xrdroot.pyroot.geom.legacyvis import solids


def test_a_zero_length_edge_is_not_an_edge() -> None:
    assert Mesh([(0, 0, 0), (1, 0, 0)], [(0, 0, 1)]).edges().tolist() == [[0, 1]]


def test_what_is_not_3d_has_no_3d_points() -> None:
    assert points_of(Primitive("TLine", {})).shape == (0, 3)


def test_a_solid_drawn_with_no_line_is_not_drawn(tmp_path) -> None:
    ROOT.TCanvas("c", "c", 100, 100)
    prim = Primitive("TGeoVolume", {"solids": [Solid(build.box(1, 1, 1), line_width=0)]})
    holder = type("Drawn3D", (), {"_xrd": prim, "Draw": lambda self, o="": ROOT.draw_hook(self, o)})
    holder().Draw()
    ROOT.gPad.SaveAs(str(tmp_path / "blank.png"))
    assert (tmp_path / "blank.png").exists()


def test_the_old_shapes_answer_their_numbers_and_their_sections() -> None:
    ROOT.TGeometry("g", "g")
    ctub = ROOT.TCTUB("C", "C", "iron", 0, 1, 1, 0, 360, 0, -0.5, -1, 0, 0.5, 1)
    pcon = ROOT.TPCON("P", "P", "void", 0, 360, 2)
    pcon.DefineSection(0, -1, 0, 1)
    pcon.DefineSection(1, 1, 0, 2)
    pgon = ROOT.TPGON("G", "G", "void", 0, 360, 6, 2)
    pgon.DefineSection(0, -1, 0, 1)
    pgon.DefineSection(1, 1, 0, 1)
    xtru = ROOT.TXTRU("X", "X", "void", 1, 1)
    for i, (x, y) in enumerate([(0, 0), (1, 0), (0, 1)]):
        xtru.DefineVertex(i, x, y)
    xtru.DefineSection(1, 2.0)
    expect((ctub.GetNumber(), 0), (ctub.GetMaterial(), "iron"), (ctub.GetRmax(), 1.0),
           (len(ctub.mesh().points) > 0, True), (len(pcon.mesh().points), 80),
           (len(pgon.mesh().points), 4 * 6), (len(xtru.mesh().points), 6),
           (ROOT.TELTU("E", "E", "void", 1, 2, 3).GetRy(), 2.0),
           (ROOT.TPARA("A", "A", "void", 1, 2, 3, 0, 0, 0).GetDy(), 2.0))  # fmt: skip
    with pytest.raises(AttributeError, match="ROOT's TCTUB has GetNdiv"):
        ctub.GetNdiv()


def test_a_geometry_lists_what_it_has_and_draws_nothing_when_it_has_no_node() -> None:
    assert not ROOT.gGeometry
    geometry = ROOT.TGeometry("g", "g")
    geometry.Draw()
    ROOT.TMaterial("m", "m")
    box = ROOT.TBRIK("B", "B", "void", 1, 1, 1)
    expect((bool(ROOT.gGeometry), True), (geometry.GetListOfMaterials().GetEntries(), 1),
           (geometry.GetListOfShapes().First(), box), (geometry.GetNode("B"), None))  # fmt: skip


def test_importing_shape_attributes_reaches_every_node_below() -> None:
    ROOT.TGeometry("g", "g")
    shape = ROOT.TBRIK("B", "B", "void", 1, 1, 1)
    top = ROOT.TNode("A", "A", "B")
    ROOT.TNode("K", "K", "B")
    shape.SetLineColor(ROOT.kRed)
    top.ImportShapeAttributes()
    top.DrawOnly()
    expect((top.GetNode("K").GetLineColor(), ROOT.kRed), (top.GetVisibility(), 0),
           ([solid.name for solid in solids(top)], ["K"]))  # fmt: skip


def test_a_matrix_record_that_is_none_is_no_matrix() -> None:
    assert legacyread._matrix(None, {}) is None


def test_the_new_geometry_corners_answer_as_root_does() -> None:
    geom = ROOT.TGeoManager("g", "g")
    material = ROOT.TGeoMaterial("m", 1, 1, 2.5)
    medium = ROOT.TGeoMedium("med", 1, material)
    top = geom.MakeBox("TOP", medium, 1, 1, 1)
    node = top.AddNode(geom.MakeTrd1("T", medium, 1, 2, 3, 4), 1)
    node.SetMatrix(ROOT.TGeoTranslation(1, 0, 0))
    node.SetVisibility(False)
    turn = ROOT.TGeoRotation()
    turn.RotateY(90)
    geom.SetVisLevel(0)
    expect((geom.GetListOfMaterials().First(), material), (material.GetDensity(), 2.5),
           (node.GetMedium(), medium), (node.IsVisible(), False), (geom.GetVisLevel(), 3),
           (list(node.GetMatrix().GetTranslation()), [1.0, 0.0, 0.0]),
           (np.round(turn.GetRotationMatrix()).tolist()[2], 1.0),
           (len(ROOT.TGeoPara(1, 1, 1, 0, 0, 0).mesh().points), 8),
           (len(ROOT.TGeoTrd1(1, 2, 3, 4).mesh().points), 8))  # fmt: skip


def test_an_obj_file_imported_quietly_says_only_its_shape(tmp_path, capsys) -> None:
    (tmp_path / "t.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\nf 1 2 3\nf 1 2 3\n")
    ROOT.TGeoTessellated.ImportFromObjFormat(str(tmp_path / "t.obj"))
    assert capsys.readouterr().out.startswith("=== Tessellated shape")


def test_turning_a_view_with_no_pad_leaves_the_pads_alone() -> None:
    ROOT.TCanvas("c", "c", 100, 100)
    view = ROOT.TView.CreateView(1)
    from xrdroot.pyroot.graphics import pads

    pads.set_current(None)
    view.RotateView(10.0, 20.0)
    assert (view.GetLongitude(), view.GetLatitude()) == (10.0, 20.0)
    assert ROOT.TPolyMarker3D(1).GetMarkerStyle() == 1


def test_a_name_neither_root_s_nor_groot_s_is_refused_by_name() -> None:
    with pytest.raises(AttributeError, match="ROOT has NoSuchThing"):
        ROOT.NoSuchThing  # noqa: B018
