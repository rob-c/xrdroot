"""``xrdroot.geom.matrix``: placements as ``TGeoMatrix`` makes, composes and reports them.

The angles are ROOT's own tutorials' (``rootgeom.C``'s ``rot1``) and ones
ROOT 6.40 printed back through ``TGeoRotation::GetAngles``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.geom import IDENTITY, Matrix

#: How far a turn's elements, worked out through sines and cosines, are from the exact.
ROUNDING = 1e-12


def test_geant_angles_put_each_local_axis_where_its_theta_and_phi_say() -> None:
    rot1 = Matrix.geant(90.0, 0.0, 90.0, 270.0, 0.0, 0.0)
    assert rot1.rotation.tolist() == [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
    assert rot1.is_reflection() and rot1.is_rotation() and not rot1.is_identity()


def test_euler_angles_turn_about_z_then_the_new_x_then_the_new_z() -> None:
    made = Matrix.euler(30.0, 40.0, 50.0)
    wanted = [[0.263258, -0.909616, 0.321394], [0.829598, 0.043412, -0.556670],
              [0.492404, 0.413176, 0.766044]]  # fmt: skip
    assert np.allclose(made.rotation, wanted, atol=5e-7)  # ROOT printed six decimals
    assert made.euler_angles() == pytest.approx((30.0, 40.0, 50.0), abs=ROUNDING)


def test_geant_angles_come_back_as_root_prints_them_with_phi_from_0_to_360() -> None:
    found = Matrix.euler(30.0, 40.0, 50.0).geant_angles()
    wanted = (60.50129577, 72.39408604, 65.59550266, 177.2675928, 40.0, 300.0)
    assert found == pytest.approx(wanted, abs=5e-8)  # ROOT printed ten figures


def test_an_axis_along_z_has_no_phi() -> None:
    assert Matrix().geant_angles() == (90.0, 0.0, 90.0, 90.0, 0.0, 0.0)


def test_euler_angles_at_a_pole_give_the_whole_turn_to_phi() -> None:
    assert Matrix.euler(20.0, 0.0, 15.0).euler_angles() == pytest.approx((35.0, 0.0, 0.0))
    assert Matrix.euler(0.0, 180.0, 0.0).euler_angles() == pytest.approx((0.0, 180.0, 0.0))


def test_euler_angles_with_phi_a_whole_turn_take_theta_from_the_other_element() -> None:
    assert Matrix.euler(0.0, 40.0, 50.0).euler_angles() == pytest.approx((0.0, 40.0, 50.0))


def test_turns_about_the_mothers_axes_are_made_after_what_was_there() -> None:
    made = Matrix().rotated(0, 30.0).rotated(1, 20.0).rotated(2, 10.0)
    wanted = [[0.925417, 0.018028, 0.378522], [0.163176, 0.882564, -0.440970],
              [-0.342020, 0.469846, 0.813798]]  # fmt: skip
    assert np.allclose(made.rotation, wanted, atol=5e-7)


def test_a_turn_takes_the_shift_round_with_it() -> None:
    made = Matrix(translation=[1.0, 0.0, 0.0]).rotated(2, 90.0)
    assert made.translation == pytest.approx([0.0, 1.0, 0.0], abs=ROUNDING)


def test_a_reflection_mirrors_the_turn_and_the_shift() -> None:
    made = Matrix(translation=[1.0, 2.0, 3.0]).reflected(1)
    assert made.is_reflection()
    assert made.translation.tolist() == [1.0, -2.0, 3.0]


def test_placements_compose_the_daughters_on_the_right() -> None:
    mother = Matrix(Matrix.euler(90.0, 0.0, 0.0).rotation, [10.0, 0.0, 0.0])
    daughter = Matrix(translation=[1.0, 0.0, 0.0])
    world = (mother @ daughter).to_master([[0.0, 0.0, 0.0]])
    assert world[0] == pytest.approx([10.0, 1.0, 0.0], abs=ROUNDING)


def test_an_inverse_takes_a_point_back_where_it_came_from() -> None:
    placed = Matrix(Matrix.euler(10.0, 20.0, 30.0).rotation, [1.0, 2.0, 3.0])
    point = np.array([[4.0, 5.0, 6.0]])
    assert placed.inverse().to_master(placed.to_master(point)) == pytest.approx(point)


def test_the_identity_is_nothing_at_all() -> None:
    assert IDENTITY.is_identity() and not IDENTITY.is_rotation() and not IDENTITY.is_reflection()
    assert math.isclose(float(np.linalg.det(IDENTITY.rotation)), 1.0)
