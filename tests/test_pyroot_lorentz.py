"""``TLorentzVector`` and ``ROOT.Math``'s four-vectors, three-vectors and ``VectorUtil``."""

from __future__ import annotations

import array
import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh

M = ROOT.Math


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_four_vector_is_made_every_way_root_makes_one():
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    expect(
        (ROOT.TLorentzVector(v), v),
        (ROOT.TLorentzVector(ROOT.TVector3(1, 2, 3), 10), v),
        (ROOT.TLorentzVector(array.array("d", [1, 2, 3, 10])), v),
        ((v.X(), v.Y(), v.Z(), v.T(), v.Px(), v.E(), v.Energy()), (1, 2, 3, 10, 1, 10, 10)),
        (v.Vect(), ROOT.TVector3(1, 2, 3)),
        (list(v), [1, 2, 3, 10]),
        (v(3), 10),
    )


def test_a_four_vector_measures_itself(capsys):
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    expect(
        (bool(v.P() == v.Rho() == pytest.approx(math.sqrt(14))), True),
        (v.Pt(), pytest.approx(math.sqrt(5))),
        (v.M2(), 86),
        (v.M(), pytest.approx(math.sqrt(86))),
        (v.Mag(), v.M()),
        (ROOT.TLorentzVector(3, 0, 0, 1).M(), pytest.approx(-math.sqrt(8))),
        (v.Mt2(), 91),
        (v.Mt(), pytest.approx(math.sqrt(91))),
        (ROOT.TLorentzVector(0, 0, 5, 1).Mt(), pytest.approx(-math.sqrt(24))),
        (v.Et(), pytest.approx(10 * math.sqrt(5 / 14))),
        (ROOT.TLorentzVector(0, 0, 1, 2).Et(), 0),
        (ROOT.TLorentzVector(1, 0, 0, -2).Et(), pytest.approx(-2)),
        (v.Et2(ROOT.TVector3(0, 0, 1)), pytest.approx(v.Et2())),
        (v.Et(ROOT.TVector3(0, 0, 1)), pytest.approx(v.Et())),
        (v.Beta(), pytest.approx(math.sqrt(14) / 10)),
        (v.Gamma(), pytest.approx(10 / math.sqrt(86))),
        ((v.Plus(), v.Minus()), (13, 7)),
        (v.Rapidity(), pytest.approx(0.5 * math.log(13 / 7))),
        (bool(v.Eta() == v.PseudoRapidity() == v.Vect().Eta()), True),
        (v.Theta(), v.Vect().Theta()),
        (v.CosTheta(), v.Vect().CosTheta()),
        (v.Phi(), v.Vect().Phi()),
        (v.Perp2(ROOT.TVector3(0, 0, 1)), pytest.approx(5)),
        (v.Perp(ROOT.TVector3(0, 0, 1)), pytest.approx(math.sqrt(5))),
        (v.Angle(ROOT.TVector3(1, 2, 3)), pytest.approx(0, abs=1e-7)),
        (v.EtaPhiVector().X(), v.Eta()),
    )
    v.Print()
    assert capsys.readouterr().out == (
        "(x,y,z,t)=(1.000000,2.000000,3.000000,10.000000) "
        "(P,eta,phi,E)=(3.741657,1.103587,1.107149,10.000000)\n"
    )


def test_a_four_vector_is_set_in_roots_coordinates():
    v = ROOT.TLorentzVector()
    v.SetPtEtaPhiM(10, 1.2, 0.4, 0.105)
    assert (v.Pt(), v.Eta(), v.Phi(), v.M()) == pytest.approx((10, 1.2, 0.4, 0.105))
    v.SetPtEtaPhiE(-5, 0.5, 0.1, 20)
    assert (v.Pt(), v.E()) == pytest.approx((5, 20))
    v.SetXYZM(1, 2, 2, -4)
    assert v.E() == 0.0
    v.SetXYZM(1, 2, 2, -1)
    assert v.E() == pytest.approx(math.sqrt(8))
    v.SetVectM(ROOT.TVector3(0, 0, 3), 4)
    assert v.E() == pytest.approx(5)
    v.SetVectMag(ROOT.TVector3(0, 0, 0), 2)
    v.SetPxPyPzE(1, 1, 1, 5)
    v.SetX(2)
    v.SetY(3)
    v.SetZ(4)
    v.SetT(9)
    v.SetE(10)
    v.SetVect(ROOT.TVector3(1, 0, 0))
    assert v == ROOT.TLorentzVector(1, 0, 0, 10)
    v.SetRho(2)
    v.SetTheta(math.pi / 2)
    v.SetPhi(0.5)
    v.SetPerp(1)
    assert (v.P(), v.Phi(), v.Pt()) == pytest.approx((1, 0.5, 1))
    out = array.array("d", [0] * 4)
    v.GetXYZT(out)
    assert out[3] == 10
    v[3] = 12
    v[0] = 3
    expect(
        (v.T(), 12),
        (v.X(), 3),
    )


def test_a_four_vector_bad_index_is_roots_error(capsys):
    expect(
        (ROOT.TLorentzVector()[7], 0.0),
        (bool("bad index (7)" in capsys.readouterr().err), True),
    )


def test_four_vectors_add_and_multiply_in_minkowski_space():
    a, b = ROOT.TLorentzVector(1, 0, 0, 2), ROOT.TLorentzVector(0, 1, 0, 3)
    expect(
        (a + b, ROOT.TLorentzVector(1, 1, 0, 5)),
        (b - a, ROOT.TLorentzVector(-1, 1, 0, 1)),
        (-a, ROOT.TLorentzVector(-1, 0, 0, -2)),
        (a * b, 6),
        (a.Dot(b), 6),
        ((a * 2).T(), 4),
        ((2 * a).T(), 4),
        (bool(a != "a"), True),
        (len({a, ROOT.TLorentzVector(a)}), 1),
    )
    c = ROOT.TLorentzVector(a)
    c += b
    c -= a
    c *= 2
    assert c == ROOT.TLorentzVector(0, 2, 0, 6)
    c *= ROOT.TRotation().RotateZ(-math.pi / 2)
    expect(
        (c.X(), pytest.approx(2)),
        (a.DeltaPhi(b), pytest.approx(-math.pi / 2)),
    )
    p, q = ROOT.TLorentzVector(1, 0, 1, 3), ROOT.TLorentzVector(0, 1, -1, 3)
    expect(
        (p.DeltaR(q), pytest.approx(math.hypot(p.Eta() - q.Eta(), p.DeltaPhi(q)))),
        (p.DrEtaPhi(q), p.DeltaR(q)),
        (p.DrRapidityPhi(q), p.DeltaR(q, True)),
    )


def test_a_four_vector_boosts_and_rotates_as_root_does():
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    b = v.BoostVector()
    v.Boost(-b)
    expect(
        ((v.X(), v.Y(), v.Z()), pytest.approx((0, 0, 0), abs=1e-12)),
        (v.T(), pytest.approx(math.sqrt(86))),
    )
    v.Boost(0, 0, 0)
    w = ROOT.TLorentzVector(1, 0, 0, 2)
    w.RotateZ(math.pi / 2)
    w.RotateX(0.1)
    w.RotateY(0.1)
    w.Rotate(0.2, ROOT.TVector3(0, 0, 1))
    w.RotateUz(ROOT.TVector3(0, 0, 1))
    expect(
        (w.P(), pytest.approx(1)),
        (w.Transform(ROOT.TRotation()).E(), 2),
    )


def test_a_genvector_keeps_the_coordinates_it_was_made_in():
    v = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105)
    expect(
        (v.M(), 0.105),
        (v.Pt(), 10),
        (v.Rho(), 10),
        (str(v), "(10,1.2,0.4,0.105)"),
    )
    other = ROOT.TLorentzVector()
    other.SetPtEtaPhiM(10, 1.2, 0.4, 0.105)
    expect(
        (
            (v.Px(), v.X(), v.E(), v.P()),
            pytest.approx((other.Px(), other.X(), other.E(), other.P())),
        ),
        (v.Theta(), pytest.approx(other.Theta())),
        (v.Rapidity(), pytest.approx(other.Rapidity())),
        (v.Mt(), pytest.approx(other.Mt())),
        (v.Et(), pytest.approx(other.Et())),
        (v.Beta(), pytest.approx(other.Beta())),
        (v.Gamma(), pytest.approx(other.Gamma())),
        (v.M2(), pytest.approx(other.M2())),
        (v.Perp2(), pytest.approx(100)),
        (bool(repr(v).startswith("<ROOT.Math.PtEtaPhiMVector (10,")), True),
        (M.PxPyPzMVector(3, 0, 4, 0).E(), 5),
        (M.PxPyPzEVector(0, 0, 0, 1).Et(), 0),
        (M.PtEtaPhiEVector(1, 0, 0, 2).M(), pytest.approx(math.sqrt(3))),
        (M.PxPyPzEVector(0, 0, 3, 1).M(), pytest.approx(-math.sqrt(8))),
        (M.PxPyPzMVector(0, 0, 3, -1).E(), pytest.approx(math.sqrt(8))),
        (M.PtEtaPhiMVector(0, 0, 0, 0).Theta(), 0),
        (M.PxPyPzEVector(0, 0, 0, 1).Eta(), 0),
        (M.PxPyPzEVector(0, 0, 2, 3).Eta(), 2 + 22756),
        (M.PxPyPzEVector(0, 0, -2, 3).Eta(), -22758),
    )


def test_genvectors_add_in_the_left_ones_coordinates():
    a, b = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105), M.PxPyPzEVector(1, 2, 3, 10)
    s = a + b
    expect(
        (type(s).__name__, "PtEtaPhiMVector"),
        (s.Px(), pytest.approx(a.Px() + 1)),
        ((b - b).E(), 0),
        ((-b).E(), -10),
        ((b * 2).E(), 20),
        ((2 * b).E(), 20),
        ((b / 2).E(), 5),
        (b * b, pytest.approx(86)),
        (b.Dot(b), pytest.approx(86)),
    )
    c = M.PxPyPzEVector(b)
    c += b
    c -= b
    c *= 3
    expect(
        (c, M.PxPyPzEVector(3, 6, 9, 30)),
        (hash(c), hash(M.PxPyPzEVector(3, 6, 9, 30))),
        (list(b), [1, 2, 3, 10]),
        (b[3], 10),
        (bool(c != "c"), True),
        (bool(M.XYZTVector is M.PxPyPzEVector), True),
        (bool(M.LorentzVector["ROOT::Math::PtEtaPhiM4D<double>"] is M.PtEtaPhiMVector), True),
    )
    with pytest.raises(TypeError, match="no four-vector"):
        M.LorentzVector["Nothing"]


def test_genvectors_set_their_own_and_others_coordinates():
    v = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105)
    v.SetPt(20)
    v.SetM(1)
    expect(
        (v.Pt(), 20),
        (v.M(), 1),
    )
    v.SetPx(5)
    assert v.Px() == pytest.approx(5)
    w = M.PxPyPzEVector()
    w.SetPxPyPzE(1, 2, 3, 10)
    w.SetXYZT(1, 2, 3, 11)
    w.SetCoordinates(1, 2, 3, 12)
    got = array.array("d", [0] * 4)
    expect(
        (w.GetCoordinates(got), (1, 2, 3, 12)),
        (got[3], 12),
        (bool(w.Coordinates() is w), True),
        (w.Vect().Z(), 3),
        (w.BoostToCM().Z(), pytest.approx(-0.25)),
        (bool(M.PxPyPzEVector(1, 0, 0, 1).isLightlike()), True),
        (bool(w.isTimelike()), True),
        (bool(not w.isSpacelike()), True),
        (
            w.ColinearRapidity(),
            pytest.approx(0.5 * math.log((12 + math.sqrt(14)) / (12 - math.sqrt(14)))),
        ),
        (M.PxPyPzEVector(v).M(), pytest.approx(v.M())),
    )


def test_three_and_two_dimensional_genvectors():
    v = M.XYZVector(1, 2, 3)
    expect(
        (str(v), "(1,2,3)"),
        (v.R(), pytest.approx(math.sqrt(14))),
        (v.Mag2(), pytest.approx(14)),
        (v.Rho(), pytest.approx(math.sqrt(5))),
        (v.Perp2(), 5),
        (bool(v.Theta() > 0), True),
        (v.Unit().R(), pytest.approx(1)),
        (M.XYZVector().Unit().R(), 0),
        (v.Cross(M.XYZVector(0, 0, 1)), M.XYZVector(2, -1, 0)),
        (v.Dot(v), 14),
        (v.SetXYZ(0, 0, 1), M.XYZVector(0, 0, 1)),
    )
    p = M.Polar3DVector(2, math.pi / 2, 0)
    expect(
        (p.X(), pytest.approx(2)),
        (M.Polar3DVector(M.XYZVector()).Theta(), 0),
    )
    r = M.RhoEtaPhiVector(1, 0, 0)
    expect(
        (r.X(), pytest.approx(1)),
        (M.RhoZPhiVector(1, 2, 0).Z(), 2),
        (M.RhoZPhiVector(M.XYZVector(0, 1, 2)).Phi(), pytest.approx(math.pi / 2)),
        (bool(M.DisplacementVector3D["ROOT::Math::Polar3D<double>"] is M.Polar3DVector), True),
        (bool(M.DisplacementVector3D["CylindricalEta3D"] is M.RhoEtaPhiVector), True),
        (bool(M.DisplacementVector3D["Cylindrical3D<double>"] is M.RhoZPhiVector), True),
        (bool(M.DisplacementVector3D["Cartesian3D"] is M.XYZVector), True),
        (M.XYZPoint(1, 1, 1).X(), 1),
    )
    with pytest.raises(TypeError, match="no three-vector"):
        M.DisplacementVector3D["Nothing"]
    q = M.XYVector(3, 4)
    expect(
        (q.R(), 5),
        (q.Mag2(), 25),
        (q.Unit().R(), pytest.approx(1)),
        (M.XYVector().Unit().R(), 0),
        (M.Polar2DVector(2, 0).X(), 2),
    )
    q.SetR(10)
    assert q.X() == pytest.approx(6)


def test_vector_util_asks_of_two_vectors_at_once():
    a, b = M.PtEtaPhiMVector(10, 1.0, 0.1, 0), M.PtEtaPhiMVector(10, -1.0, 3.0, 0)
    expect(
        (M.VectorUtil.DeltaPhi(a, b), pytest.approx(2.9)),
        (bool(M.VectorUtil.DeltaPhi(M.XYZVector(1, 0, 0), M.XYZVector(-1, -0.1, 0)) < 0), True),
        (M.VectorUtil.DeltaR(a, b), pytest.approx(math.hypot(2.9, 2.0))),
        (M.VectorUtil.DeltaR2(a, b), pytest.approx(2.9**2 + 4)),
        (bool(M.VectorUtil.DeltaRapidityPhi(a, b) > 0), True),
        (M.VectorUtil.InvariantMass(a, b), pytest.approx((a + b).M())),
        (M.VectorUtil.InvariantMass2(a, b), pytest.approx((a + b).M2())),
        (M.VectorUtil.CosTheta(M.XYZVector(1, 0, 0), M.XYZVector(1, 0, 0)), pytest.approx(1)),
        (M.VectorUtil.CosTheta(M.XYZVector(), M.XYZVector(1, 0, 0)), 1),
        (
            M.VectorUtil.Angle(M.XYZVector(1, 0, 0), M.XYZVector(0, 1, 0)),
            pytest.approx(math.pi / 2),
        ),
        (M.VectorUtil.Perp(M.XYZVector(1, 1, 0), M.XYZVector(1, 0, 0)), pytest.approx(1)),
    )
    rest = M.VectorUtil.boost(
        M.PxPyPzEVector(1, 2, 3, 10), M.PxPyPzEVector(1, 2, 3, 10).BoostToCM()
    )
    expect(
        (rest.Px(), pytest.approx(0, abs=1e-12)),
        (rest.E(), pytest.approx(math.sqrt(86))),
    )
