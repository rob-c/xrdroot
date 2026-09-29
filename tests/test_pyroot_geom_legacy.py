"""ROOT's first geometry package: ``TGeometry``, ``TNode`` and the ``TBRIK`` family.

``mkgeom.C`` builds a tree of one of each old shape; ROOT
6.40 drew it into ``tgeometry-small-6.40.png`` and wrote it into
``tgeometry-small-6.40.root``. The macro is run here, and the file read
back, and both drawn pixel for pixel as ROOT drew them.
"""

from __future__ import annotations

import shutil
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import DATA, differing, geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.cint.errors import MacroError
from xrdroot.cint.execute import run
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.geom import legacyread
from xrdroot.pyroot.geom.legacyvis import solids


def test_the_small_geometry_is_drawn_as_root_drew_it_and_not_written(tmp_path: Any) -> None:
    with pytest.raises(MacroError, match="writing a TGeometry is not supported"):
        run(DATA / "mkgeom.C", use_cache=False)
    assert differing(tmp_path / "tgeometry-small-6.40.png", "tgeometry-small-6.40.png") == 0.0


def test_the_small_geometry_read_from_roots_file_is_drawn_as_root_drew_it(tmp_path: Any) -> None:
    shutil.copy(DATA / "tgeometry-small-6.40.root", tmp_path / "small.root")
    canvas = ROOT.TCanvas("c", "c", 300, 300)
    ROOT.TFile("small.root")
    small = ROOT.gROOT.FindObject("small")
    small.Draw()
    canvas.SaveAs(str(tmp_path / "read.png"))
    assert differing(tmp_path / "read.png", "tgeometry-small-6.40.png") == 0.0
    expect((small.GetTitle(), "every old shape"), (small.GetNode("I").GetShape().GetName(), "SPHE"),
           (small.GetShape("SPHE").GetLineColor(), ROOT.kRed), (ROOT.gGeometry.GetName(), "small"),
           (small.GetRotMatrix("tilt").GetNumber(), 1))  # fmt: skip


def test_a_shape_the_reader_does_not_know_is_refused_saying_which_it_does() -> None:
    with pytest.raises(UnsupportedFeatureError, match="it knows TBRIK"):
        legacyread._shape_class({"fOdd": 1.0})


def test_nodes_are_made_inside_the_node_cd_to_last() -> None:
    ROOT.TGeometry("g", "g")
    ROOT.TBRIK("BOX", "BOX", "void", 10, 10, 10)
    ROOT.TTUBE("TUBE", "TUBE", "void", 1, 2, 3, 0.5)
    top = ROOT.TNode("TOP", "TOP", "BOX")
    kid = ROOT.TNode("KID", "KID", "TUBE", 1, 2, 3)
    kid.cd()
    grandchild = ROOT.TNode("GRAND", "GRAND", ROOT.gGeometry.GetShape("TUBE"))
    grandchild.SetPosition(0, 0, 5)
    expect((kid.GetParent(), top), (top.GetListOfNodes(), [kid]), (grandchild.GetParent(), kid),
           (top.GetNode("GRAND"), grandchild), (top.GetNode("NONE"), None),
           ((kid.GetX(), kid.GetY(), kid.GetZ()), (1.0, 2.0, 3.0)), (grandchild.GetZ(), 5.0),
           (ROOT.gGeometry.GetListOfNodes().GetEntries(), 1), (kid.GetMatrix(), None),
           (ROOT.gGeometry.FindObject("GRAND"), grandchild), (ROOT.gGeometry.FindObject("x"), None),
           (ROOT.TUBE.GetAspectRatio(), 0.5))  # fmt: skip


def test_a_node_of_a_shape_that_is_not_there_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    ROOT.TNode("N", "N", "NOSHAPE")
    assert capsys.readouterr().out == "Error Referenced shape does not exist: NOSHAPE\n"


def tree() -> Any:
    """A node with a daughter and a granddaughter, all boxes."""
    ROOT.TGeometry("g", "g")
    ROOT.TBRIK("B", "B", "void", 1, 1, 1)
    top = ROOT.TNode("A", "A", "B")
    ROOT.TNode("B1", "B1", "B", 5, 0, 0).cd()
    ROOT.TNode("C1", "C1", "B", 0, 5, 0)
    return top


@pytest.mark.parametrize(
    ("code", "drawn"),
    [(1, ["A", "B1", "C1"]), (0, ["B1", "C1"]), (-1, []), (-2, ["A"]), (-3, ["C1"]),
     (-4, ["B1"]), (2, ["B1", "C1"]), (3, ["A", "B1", "C1"]), (7, ["A", "B1", "C1"])],
)  # fmt: skip
def test_each_visibility_code_shows_what_root_s_says(code: int, drawn: list[str]) -> None:
    top = tree()
    top.SetVisibility(code)
    assert [solid.name for solid in solids(top)] == drawn


def test_a_hidden_shape_hides_its_nodes_and_an_exploded_view_spreads_them() -> None:
    top = tree()
    ROOT.gGeometry.SetBomb(2.0)
    far = solids(top)[1].mesh.extent()[0][0]
    ROOT.gGeometry.GetShape("B").SetVisibility(0)
    expect((far, 9.0), (ROOT.gGeometry.GetBomb(), 2.0), (solids(top), []))


def test_rotation_matrices_are_made_from_angles_or_nine_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ROOT.TGeometry("g", "g")
    angles = ROOT.TRotMatrix("a", "a", 90, 0, 90, 90, 0, 0)
    mirror = ROOT.TRotMatrix("m", "m", np.array([1.0, 0, 0, 0, -1, 0, 0, 0, 1]))
    ROOT.TRotMatrix("e", "e", 10.0, 20.0, 30.0)
    expect((list(angles.GetMatrix()), [1, 0, 0, 0, 1, 0, 0, 0, 1]), (mirror.IsReflection(), True),
           (capsys.readouterr().out,
            "ERROR: This form of TRotMatrix constructor not implemented yet\n"),
           (ROOT.gGeometry.GetListOfMatrices().GetEntries(), 3))  # fmt: skip


def test_materials_and_mixtures_are_listed_with_the_geometry() -> None:
    ROOT.TGeometry("g", "g")
    iron = ROOT.TMaterial("mat1", "IRON", 55.85, 26, 7.87)
    mylar = ROOT.TMixture("mix", "MYLAR", 2)
    mylar.DefineElement(0, 12.01, 6, 0.5)
    expect(((iron.GetA(), iron.GetZ(), iron.GetDensity(), iron.GetNumber()),
           (55.85, 26.0, 7.87, 0)),
           (mylar.GetNmixt(), 2), (ROOT.gGeometry.GetMaterial("mix"), mylar))  # fmt: skip
