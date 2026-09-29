"""``xrdroot.pyroot.geom``: ``TGeoManager`` building a geometry, and drawing it as ROOT does.

ROOT's own ``rootgeom.C`` is run against this namespace and its picture held
to ROOT 6.40's pixel for pixel; what the manager says as it closes, and
what matrices print, are what ROOT printed for the same geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import differing, geometry_session  # noqa: F401
from pyrootsupport import expect
from xrdroot.cint.execute import run

#: ROOT's ``rootgeom.C``, copied unchanged.
ROOTGEOM = Path(__file__).parent / "data" / "cint" / "visualisation" / "geom" / "rootgeom.C"


def test_rootgeom_is_drawn_pixel_for_pixel_as_root_draws_it(tmp_path: Any) -> None:
    canvas = ROOT.TCanvas("c", "c", 700, 500)
    run(ROOTGEOM, use_cache=False)
    canvas.SaveAs(str(tmp_path / "rootgeom.png"))
    assert differing(tmp_path / "rootgeom.png", "tgeo-rootgeom-6.40.png") == 0.0


def test_closing_rootgeom_counts_its_nodes_and_levels_as_root_does(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run(ROOTGEOM, (False,), use_cache=False)
    said = capsys.readouterr().err.splitlines()
    assert said[0] == ("Info in <TGeoManager::TGeoManager>: Geometry simple1, Simple geometry "
                       "created")
    assert "Info in <TGeoManager::CountLevels>: max level = 4, max placements = 6" in said
    assert ("Info in <TGeoManager::CloseGeometry>: 485 nodes/ 13 volume UID's in Simple "
            "geometry") in said  # fmt: skip
    assert ROOT.gGeoManager.IsClosed() and ROOT.gGeoManager.GetTopVolume().GetName() == "TOP"


def test_a_quiet_geometry_says_nothing(capsys: pytest.CaptureFixture[str]) -> None:
    ROOT.TGeoManager.SetVerboseLevel(0)
    run(ROOTGEOM, (False,), use_cache=False)
    expect((capsys.readouterr().err, ""), (ROOT.TGeoManager.GetVerboseLevel(), 0))


def test_matrices_print_as_root_prints_them(capsys: pytest.CaptureFixture[str]) -> None:
    turn = ROOT.TGeoRotation("rot1", 90.0, 0.0, 90.0, 270.0, 0.0, 0.0)
    placed = ROOT.TGeoCombiTrans("c", 7.5, -7.5, 0.0, turn)
    placed.RegisterYourself()
    turn.Print()
    placed.Print()
    assert capsys.readouterr().out.splitlines() == [
        "matrix rot1 - tr=0  rot=1  refl=1  scl=0 shr=0 reg=0 own=0",
        "  1.000000    0.000000    0.000000    Tx =   0.000000",
        "  0.000000   -1.000000    0.000000    Ty =   0.000000",
        "  0.000000    0.000000    1.000000    Tz =   0.000000",
        "matrix c - tr=1  rot=1  refl=1  scl=0 shr=0 reg=1 own=0",
        "  1.000000    0.000000    0.000000    Tx =   7.500000",
        "  0.000000   -1.000000    0.000000    Ty =  -7.500000",
        "  0.000000    0.000000    1.000000    Tz =   0.000000",
    ]


def test_a_rotation_hands_its_angles_back_through_what_it_is_given() -> None:
    euler = ROOT.TGeoRotation("e", 30.0, 40.0, 50.0)
    cells = [np.zeros(1) for _ in range(6)]
    euler.GetAngles(*cells)
    assert [float(c[0]) for c in cells] == pytest.approx(
        [60.50129577, 72.39408604, 65.59550266, 177.2675928, 40.0, 300.0], abs=5e-8)
    assert euler.GetAngles(*cells[:3]) == pytest.approx((30.0, 40.0, 50.0))


def test_rotations_turn_about_the_mothers_axes_and_multiply() -> None:
    turn = ROOT.TGeoRotation()
    turn.RotateX(90)
    copy = ROOT.TGeoRotation(turn)
    copy.RotateZ(90)
    copy.MultiplyBy(ROOT.TGeoRotation("z", 0.0, 0.0, 0.0))
    turn.SetAngles(0.0, 0.0, 0.0)
    expect((turn.IsRotation(), False), (copy.IsRotation(), True), (copy.IsTranslation(), False))
    turn.SetMatrix([0, -1, 0, 1, 0, 0, 0, 0, 1])
    assert list(turn.GetRotationMatrix()) == [0, -1, 0, 1, 0, 0, 0, 0, 1]
