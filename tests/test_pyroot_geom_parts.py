"""The parts of a ``TGeo`` geometry: matrices, materials, shapes, as ROOT's classes answer.

Boxes are what ROOT 6.40's ``GetDX``, ``GetDY``, ``GetDZ`` and
``GetOrigin`` printed for the same shapes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from geomsupport import geometry_session  # noqa: F401
from pyrootsupport import expect


def test_translations_are_made_named_or_not_and_moved_axis_by_axis() -> None:
    shift = ROOT.TGeoTranslation("t", 1.0, 2.0, 3.0)
    shift.SetDx(4.0)
    shift.SetDy(5.0)
    shift.SetDz(6.0)
    copied = ROOT.TGeoTranslation(shift)
    copied.SetTranslation(0.0, 0.0, 1.0)
    expect((list(shift.GetTranslation()), [4.0, 5.0, 6.0]), (shift.IsTranslation(), True),
           (list(copied.GetTranslation()), [0.0, 0.0, 1.0]), (shift.IsCombi(), False))  # fmt: skip


def test_a_point_goes_to_the_mother_and_back() -> None:
    placed = ROOT.TGeoCombiTrans(1.0, 0.0, 0.0, ROOT.TGeoRotation("r", 90.0, 0.0, 0.0))
    master, local = np.zeros(3), np.zeros(3)
    placed.LocalToMaster([1.0, 0.0, 0.0], master)
    placed.MasterToLocal(master, local)
    assert list(master) == pytest.approx([1.0, 1.0, 0.0])
    assert list(local) == pytest.approx([1.0, 0.0, 0.0])
    assert list(placed.Inverse().GetTranslation()) == pytest.approx([0.0, 1.0, 0.0])


def test_combined_placements_are_made_every_way_root_makes_them() -> None:
    turn = ROOT.TGeoRotation("r", 0.0, 90.0, 0.0)
    both = ROOT.TGeoCombiTrans(ROOT.TGeoTranslation(1.0, 2.0, 3.0), turn)
    copied = ROOT.TGeoCombiTrans(both)
    bare = ROOT.TGeoCombiTrans("bare")
    bare.SetTranslation(1.0, 0.0, 0.0)
    bare.SetRotation(turn)
    expect((both.GetRotation() is turn, True), (copied.GetRotation(), None),
           (list(copied.GetTranslation()), [1.0, 2.0, 3.0]), (bare.IsCombi(), True))  # fmt: skip


def test_general_matrices_multiply_on_either_side_and_scales_stretch() -> None:
    general = ROOT.TGeoHMatrix()
    general.Multiply(ROOT.TGeoTranslation(1.0, 0.0, 0.0))
    general.MultiplyLeft(ROOT.TGeoRotation("r", 90.0, 0.0, 0.0))
    scale = ROOT.TGeoScale("s", 2.0, 3.0, 4.0)
    expect((list(general.GetTranslation()), pytest.approx([0.0, 1.0, 0.0])),
           (list(scale.GetRotationMatrix()), [2, 0, 0, 0, 3, 0, 0, 0, 4]),
           (ROOT.TGeoScale().IsIdentity(), True), (ROOT.gGeoIdentity.GetName(),
           "Identity"))  # fmt: skip


def test_reflections_mirror_through_each_plane() -> None:
    mirrored = ROOT.TGeoRotation()
    mirrored.ReflectX()
    mirrored.ReflectY()
    assert not mirrored.IsReflection()
    mirrored.ReflectZ()
    assert mirrored.IsReflection()


def test_materials_are_listed_with_the_geometry_and_vacuums_have_huge_lengths() -> None:
    geom = ROOT.TGeoManager("g", "g")
    vacuum = ROOT.TGeoMaterial("Vacuum", 0, 0, 0)
    aluminium = ROOT.TGeoMaterial(" Al ", 26.98, 13, 2.7, 8.9, 39.0)
    carbon = ROOT.TGeoMaterial("C", ROOT.TGeoElement("C", "carbon", 6, 12.011), 2.2)
    aluminium.SetTransparency(40)
    aluminium.SetDensity(2.7)
    expect((vacuum.GetRadLen(), 1e30), (aluminium.GetIntLen(), 39.0), (aluminium.GetName(), "Al"),
           (carbon.GetZ(), 6.0), (carbon.GetIndex(), 2), (geom.GetMaterial("Al"), aluminium),
           (geom.GetMaterial(0), vacuum), (geom.GetMaterial(9), None),
           (aluminium.GetTransparency(), 40), (vacuum.IsMixture(), False))  # fmt: skip


def test_a_mixture_is_its_elements_by_weight() -> None:
    mylar = ROOT.TGeoMixture("mylar", 3, 1.39)
    mylar.AddElement(12.01, 6, 0.625)
    mylar.DefineElement(1, 1.008, 1, 0.042)
    mylar.AddElement(ROOT.TGeoElement("O", "oxygen", 8, 16.0), 0.333)
    expect((mylar.GetNelements(), 3), (mylar.IsMixture(), True),
           (mylar.GetA(),
           pytest.approx((12.01 * .625 + 1.008 * .042 + 16 * .333) / 1.0)))  # fmt: skip


def test_an_element_prints_as_root_prints_it(capsys: pytest.CaptureFixture[str]) -> None:
    ROOT.TGeoElement("C", "carbon", 6, 12.011).Print()
    assert capsys.readouterr().out == "Element: C      Z=6   N=12.000000   A=12.011000 [g/mole]\n"


def test_a_medium_keeps_its_number_material_and_parameters() -> None:
    material = ROOT.TGeoMaterial("Fe", 55.845, 26, 7.87)
    medium = ROOT.TGeoMedium("Iron", 3, material, np.arange(12.0))
    plain = ROOT.TGeoMedium("Plain", 4, material)
    expect((medium.GetId(), 3), (medium.GetMaterial(), material), (medium.GetParam(11), 0.0),
           (medium.GetParam(4), 4.0), (plain.GetParam(0), 0.0),
           (ROOT.gGeoManager.GetMedium("Plain"), plain))  # fmt: skip


def box_of(shape: Any) -> list[float]:
    return [shape.GetDX(), shape.GetDY(), shape.GetDZ(), *shape.GetOrigin()]


def test_shapes_have_the_boxes_root_finds_for_them() -> None:
    segment = ROOT.TGeoTubeSeg("ts", 5, 15, 5, 90, 270)
    sphere = ROOT.TGeoSphere("sp", 1, 2, 10, 80, 0, 90)
    trap = ROOT.TGeoTrap("tr", 190, 0, 0, 60, 40, 90, 15, 120, 80, 180, 15)
    assert box_of(segment) == pytest.approx([7.5, 15, 5, -7.5, 0, 0], abs=1e-9)
    assert box_of(sphere) == pytest.approx([0.984807753, 0.984807753, 0.8979836642, 0.984807753,
                                            0.984807753, 1.071631842], abs=2e-4)  # fmt: skip
    # (the box is the extent of the surface drawn in 360 steps, 2e-4 from the arcs it cuts)
    assert box_of(trap)[:3] == pytest.approx([180.0, 120.0, 190.0])


def test_each_shape_answers_its_own_getters_and_defaults_what_it_was_not_given() -> None:
    expect((ROOT.TGeoTube(1, 2, 3).GetRmax(), 2.0), (ROOT.TGeoCone(1, 2, 3, 4, 5).GetRmin2(), 4.0),
           (ROOT.TGeoSphere(1, 2).GetTheta2(), 180.0), (ROOT.TGeoTorus(10, 1, 2).GetDphi(), 360.0),
           (ROOT.TGeoTrd1(1, 2, 3, 4).GetDx2(), 2.0), (ROOT.TGeoPara(1, 2, 3, 4, 5, 6).GetPhi(),
           6.0),
           (ROOT.TGeoEltu(1, 2, 3).GetB(), 2.0), (ROOT.TGeoHype(1, 2, 3, 4, 5).GetStOut(), 4.0),
           (ROOT.TGeoParaboloid(1, 2, 3).GetRhi(), 2.0), (ROOT.TGeoTube().IsComposite(), False),
           (ROOT.TGeoBBox("b", 1, 2, 3).GetNsegments(), 20))  # fmt: skip
    with pytest.raises(AttributeError, match="ROOT's TGeoTube has GetVolume"):
        ROOT.TGeoTube(1, 2, 3).GetVolume()


def test_shapes_are_reshaped_after_they_are_made() -> None:
    box = ROOT.TGeoBBox("b", 1, 1, 1)
    box.SetBoxDimensions(2, 3, 4, np.array([1.0, 0.0, 0.0]))
    tube = ROOT.TGeoTube(1, 2, 3)
    tube.SetTubeDimensions(4, 5)
    arb = ROOT.TGeoArb8("a", 5.0, np.arange(16.0))
    arb.SetVertex(0, -1, -1)
    box.ComputeBBox()
    expect((box_of(box), [2.0, 3.0, 4.0, 1.0, 0.0, 0.0]), (tube.GetRmin(), 4.0),
           (list(arb.GetVertices()[:4]), [-1.0, -1.0, 2.0, 3.0]))  # fmt: skip


def test_every_solid_makes_a_surface_to_draw() -> None:
    made = [ROOT.TGeoTrd2(1, 2, 3, 4, 5), ROOT.TGeoGtra(10, 0, 0, 30, 5, 3, 3, 0, 5, 3, 3, 0),
            ROOT.TGeoCtub(0, 1, 1, 0, 360, 0, -0.5, -1, 0, 0.5, 1),
                   ROOT.TGeoConeSeg(1, 0, 1, 0, 2, 0, 90),
            ROOT.TGeoTorus(10, 1, 2), ROOT.TGeoEltu(1, 2, 3), ROOT.TGeoParaboloid(1, 2, 3),
            ROOT.TGeoHype(1, 0, 2, 45, 1)]  # fmt: skip
    assert all(len(shape.mesh().edges()) > 0 for shape in made)
