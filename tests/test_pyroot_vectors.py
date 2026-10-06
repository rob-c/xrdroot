"""``TVector2``, ``TVector3``, ``TRotation`` and ``TLorentzVector``, operation for operation."""

from __future__ import annotations

import array
import math

import numpy as np

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_plane_vector_measures_and_turns(capsys):
    v = ROOT.TVector2(3, 4)
    expect(
        ((v.X(), v.Y(), v.Px(), v.Py()), (3, 4, 3, 4)),
        (v.Mod(), 5),
        (v.Mod2(), 25),
        (v.Phi(), pytest.approx(math.atan2(4, 3))),
        (ROOT.TVector2(-1, 0).Phi(), pytest.approx(math.pi)),
        (v.Unit().Mod(), pytest.approx(1)),
        (ROOT.TVector2().Unit().Mod(), 0),
        (v.Ort().X(), pytest.approx(0.6)),
    )
    turned = v.Rotate(math.pi / 2)
    expect(
        ((turned.X(), turned.Y()), pytest.approx((-4, 3))),
        (v.DeltaPhi(turned), pytest.approx(math.pi / 2)),
    )
    along = ROOT.TVector2(1, 0)
    expect(
        (v.Proj(along).X(), 3),
        (v.Norm(along).Y(), 4),
    )
    v.SetMagPhi(-2, 0)
    assert (v.X(), v.Y()) == pytest.approx((2, 0))
    v.Set(1, 2)
    v.SetX(5)
    v.SetY(6)
    v.Set(ROOT.TVector2(7, 8))
    assert (v[0], v(1)) == (7, 8)
    ROOT.TVector2([1, 1]).Print()
    assert capsys.readouterr().out == (
        "TVector2 A 2D physics vector (x,y)=(1.000000,1.000000) (rho,phi)=(1.414214,45.000000)\n"
    )


def test_plane_vectors_add_scale_and_multiply():
    a, b = ROOT.TVector2(1, 2), ROOT.TVector2(3, 5)
    expect(
        (a + b, ROOT.TVector2(4, 7)),
        (b - a, ROOT.TVector2(2, 3)),
        (-a, ROOT.TVector2(-1, -2)),
        (a * 2, ROOT.TVector2(2, 4)),
        (2 * a, a * 2),
        (a * b, 13),
        (a ^ b, -1),
        ((b / 2).X(), 1.5),
        (bool(a != "a"), True),
        (len({ROOT.TVector2(1, 2), ROOT.TVector2(1, 2)}), 1),
    )


def test_angles_are_brought_into_roots_ranges(capsys):
    expect(
        (ROOT.TVector2.Phi_0_2pi(-0.5), pytest.approx(2 * math.pi - 0.5)),
        (ROOT.TVector2.Phi_0_2pi(7.0), pytest.approx(7.0 - 2 * math.pi)),
        (ROOT.TVector2.Phi_mpi_pi(4.0), pytest.approx(4.0 - 2 * math.pi)),
        (ROOT.TVector2.Phi_mpi_pi(-4.0), pytest.approx(2 * math.pi - 4.0)),
        (bool(math.isnan(ROOT.TVector2.Phi_mpi_pi(math.nan))), True),
        (bool("function called with NaN" in capsys.readouterr().err), True),
    )


def test_a_space_vector_measures_itself_as_root_does(capsys):
    v = ROOT.TVector3(1, 2, 3)
    expect(
        ((v.x(), v.y(), v.z(), v.Px(), v.Py(), v.Pz()), (1, 2, 3, 1, 2, 3)),
        (v.Mag2(), 14),
        (v.Perp2(), 5),
        (v.Pt(), pytest.approx(math.sqrt(5))),
        (v.Theta(), pytest.approx(math.atan2(math.sqrt(5), 3))),
        (v.CosTheta(), pytest.approx(3 / math.sqrt(14))),
        (ROOT.TVector3().CosTheta(), 1),
        (ROOT.TVector3().Phi(), 0),
        (ROOT.TVector3().Theta(), 0),
        (v.Eta(), pytest.approx(math.asinh(3 / math.sqrt(5)))),
        (ROOT.TVector3(0, 0, 2).Eta(), 10e10),
        (ROOT.TVector3(0, 0, -2).Eta(), -10e10),
        (ROOT.TVector3().Eta(), 0),
        (v.Perp2(ROOT.TVector3(0, 0, 1)), pytest.approx(5)),
        (v.Perp(ROOT.TVector3()), pytest.approx(math.sqrt(14))),
        (ROOT.TVector3(1, 0, 0).Perp2(ROOT.TVector3(1, 0, 0)), 0),
        (v.Angle(ROOT.TVector3(1, 2, 3)), pytest.approx(0, abs=1e-7)),
        (v.Angle(ROOT.TVector3()), 0),
        (v.XYvector(), ROOT.TVector2(1, 2)),
        (v.EtaPhiVector().Y(), v.Phi()),
    )
    v.Print()
    assert capsys.readouterr().out.startswith("TVector3 A 3D physics vector (x,y,z)=(1.000000,")


def test_space_vectors_combine():
    a, b = ROOT.TVector3(1, 0, 0), ROOT.TVector3(0, 1, 0)
    expect(
        (a.Cross(b), ROOT.TVector3(0, 0, 1)),
        (a.Dot(b), 0),
        (a * b, 0),
        ((a + b) - b, a),
        (-a, ROOT.TVector3(-1, 0, 0)),
        ((2 * a)[0], 2),
        ((a / 2).X(), 0.5),
        (a.Unit(), a),
        (ROOT.TVector3().Unit(), ROOT.TVector3()),
    )
    c = ROOT.TVector3(a)
    c += b
    c -= a
    c *= 3
    expect(
        (c, ROOT.TVector3(0, 3, 0)),
        (list(c), [0, 3, 0]),
        (bool(c != "c"), True),
    )
    c[2] = 4
    expect(
        (c(2), 4),
        (len({c, ROOT.TVector3(0, 3, 4)}), 1),
        (a.DeltaR(b), pytest.approx(math.pi / 2)),
        (a.DrEtaPhi(b), a.DeltaR(b)),
    )
    for x, y, z in ((1, 2, 3), (3, 2, 1), (2, 1, 3), (1, 3, 2)):
        v = ROOT.TVector3(x, y, z)
        assert v.Dot(v.Orthogonal()) == 0
    made = ROOT.TVector3(array.array("d", [1, 2, 3]))
    out = array.array("d", [0, 0, 0])
    made.GetXYZ(out)
    assert list(out) == [1, 2, 3]


def test_a_space_vector_bad_index_is_roots_error(capsys):
    expect(
        (ROOT.TVector3(1, 2, 3)[5], 0.0),
        (bool("bad index (5) returning 0" in capsys.readouterr().err), True),
    )


def test_space_vectors_are_set_in_their_other_coordinates():
    v = ROOT.TVector3(1, 1, 1)
    v.SetMag(2)
    assert v.Mag() == pytest.approx(2)
    v.SetPerp(1)
    assert v.Perp() == pytest.approx(1)
    v.SetTheta(0.5)
    assert v.Theta() == pytest.approx(0.5)
    v.SetPhi(1.0)
    assert v.Phi() == pytest.approx(1.0)
    v.SetMagThetaPhi(-3, 0.2, 0.3)
    assert (v.Mag(), v.Theta(), v.Phi()) == pytest.approx((3, 0.2, 0.3))
    v.SetPtEtaPhi(2, 0.7, 0.1)
    assert (v.Pt(), v.Eta(), v.Phi()) == pytest.approx((2, 0.7, 0.1))


def test_space_vectors_are_set_by_transverse_momentum_and_by_component(capsys):
    v = ROOT.TVector3(1, 1, 1)
    v.SetPtThetaPhi(2, 0.4, 0.1)
    assert v.Theta() == pytest.approx(0.4)
    v.SetPtThetaPhi(2, 0.0, 0.1)
    assert v.Z() == 0
    ROOT.TVector3().SetMag(1)
    ROOT.TVector3().SetPerp(1)
    assert "zero vector can't be stretched" in capsys.readouterr().err
    v.SetX(1)
    v.SetY(2)
    v.SetZ(3)
    assert v == ROOT.TVector3(1, 2, 3)


def test_space_vectors_rotate_about_every_axis():
    v = ROOT.TVector3(1, 0, 0)
    v.RotateZ(math.pi / 2)
    assert (v.X(), v.Y()) == pytest.approx((0, 1))
    v.RotateX(math.pi / 2)
    assert v.Z() == pytest.approx(1)
    v.RotateY(math.pi / 2)
    assert v.X() == pytest.approx(1)
    w = ROOT.TVector3(1, 0, 0)
    w.Rotate(math.pi / 2, ROOT.TVector3(0, 0, 5))
    assert (w.X(), w.Y()) == pytest.approx((0, 1))
    u = ROOT.TVector3(0, 0, 1)
    u.RotateUz(ROOT.TVector3(1, 0, 0))
    assert u.X() == pytest.approx(1)
    t = ROOT.TVector3(1, 2, 3)
    t.RotateUz(ROOT.TVector3(0, 0, -1))
    assert t == ROOT.TVector3(-1, 2, -3)
    t.RotateUz(ROOT.TVector3(0, 0, 1))
    assert t == ROOT.TVector3(-1, 2, -3)


def test_a_rotation_is_a_matrix_that_composes(capsys):
    r = ROOT.TRotation().RotateZ(math.pi / 2).RotateX(0.0)
    moved = r * ROOT.TVector3(1, 0, 0)
    expect(
        ((moved.X(), moved.Y()), pytest.approx((0, 1))),
        (r(0, 1), pytest.approx(-1)),
    )
    back = r.Inverse() * moved
    assert back.X() == pytest.approx(1)
    both = r * ROOT.TRotation().RotateY(math.pi / 2)
    assert both(0, 0) == pytest.approx(0, abs=1e-12)
    v = ROOT.TVector3(1, 0, 0)
    v *= r
    expect(
        (v.Y(), pytest.approx(1)),
        (v.Transform(r.Inverse()).X(), pytest.approx(1)),
    )
    ROOT.TRotation().Rotate(1.0, ROOT.TVector3())
    assert "zero axis" in capsys.readouterr().err


def test_an_array_numpy_copies_from_a_container_is_its_own() -> None:
    for made in (ROOT.std.vector["int"]([1, 2]), ROOT.TVectorD(2), ROOT.TArrayD(2),
                 ROOT.TMatrixD(1, 2)):  # fmt: skip
        copied = np.array(made)
        copied[...] = 7
        assert 7 not in np.asarray(made)
        assert np.shares_memory(np.asarray(made), np.asarray(made))
