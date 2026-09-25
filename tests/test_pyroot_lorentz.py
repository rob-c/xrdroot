"""``TLorentzVector`` and ``ROOT.Math``'s four-vectors, three-vectors and ``VectorUtil``."""

from __future__ import annotations

import array
import math

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh

M = ROOT.Math


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_four_vector_is_made_every_way_root_makes_one():
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    assert ROOT.TLorentzVector(v) == v
    assert ROOT.TLorentzVector(ROOT.TVector3(1, 2, 3), 10) == v
    assert ROOT.TLorentzVector(array.array("d", [1, 2, 3, 10])) == v
    assert (v.X(), v.Y(), v.Z(), v.T(), v.Px(), v.E(), v.Energy()) == (1, 2, 3, 10, 1, 10, 10)
    assert v.Vect() == ROOT.TVector3(1, 2, 3)
    assert list(v) == [1, 2, 3, 10]
    assert v(3) == 10


def test_a_four_vector_measures_itself(capsys):
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    assert v.P() == v.Rho() == pytest.approx(math.sqrt(14))
    assert v.Pt() == pytest.approx(math.sqrt(5))
    assert v.M2() == 86
    assert v.M() == pytest.approx(math.sqrt(86))
    assert v.Mag() == v.M()
    assert ROOT.TLorentzVector(3, 0, 0, 1).M() == pytest.approx(-math.sqrt(8))
    assert v.Mt2() == 91
    assert v.Mt() == pytest.approx(math.sqrt(91))
    assert ROOT.TLorentzVector(0, 0, 5, 1).Mt() == pytest.approx(-math.sqrt(24))
    assert v.Et() == pytest.approx(10 * math.sqrt(5 / 14))
    assert ROOT.TLorentzVector(0, 0, 1, 2).Et() == 0
    assert ROOT.TLorentzVector(1, 0, 0, -2).Et() == pytest.approx(-2)
    assert v.Et2(ROOT.TVector3(0, 0, 1)) == pytest.approx(v.Et2())
    assert v.Et(ROOT.TVector3(0, 0, 1)) == pytest.approx(v.Et())
    assert v.Beta() == pytest.approx(math.sqrt(14) / 10)
    assert v.Gamma() == pytest.approx(10 / math.sqrt(86))
    assert (v.Plus(), v.Minus()) == (13, 7)
    assert v.Rapidity() == pytest.approx(0.5 * math.log(13 / 7))
    assert v.Eta() == v.PseudoRapidity() == v.Vect().Eta()
    assert v.Theta() == v.Vect().Theta()
    assert v.CosTheta() == v.Vect().CosTheta()
    assert v.Phi() == v.Vect().Phi()
    assert v.Perp2(ROOT.TVector3(0, 0, 1)) == pytest.approx(5)
    assert v.Perp(ROOT.TVector3(0, 0, 1)) == pytest.approx(math.sqrt(5))
    assert v.Angle(ROOT.TVector3(1, 2, 3)) == pytest.approx(0, abs=1e-7)
    assert v.EtaPhiVector().X() == v.Eta()
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
    assert v.T() == 12
    assert v.X() == 3


def test_a_four_vector_bad_index_is_roots_error(capsys):
    assert ROOT.TLorentzVector()[7] == 0.0
    assert "bad index (7)" in capsys.readouterr().err


def test_four_vectors_add_and_multiply_in_minkowski_space():
    a, b = ROOT.TLorentzVector(1, 0, 0, 2), ROOT.TLorentzVector(0, 1, 0, 3)
    assert a + b == ROOT.TLorentzVector(1, 1, 0, 5)
    assert b - a == ROOT.TLorentzVector(-1, 1, 0, 1)
    assert -a == ROOT.TLorentzVector(-1, 0, 0, -2)
    assert a * b == 6
    assert a.Dot(b) == 6
    assert (a * 2).T() == 4
    assert (2 * a).T() == 4
    assert a != "a"
    assert len({a, ROOT.TLorentzVector(a)}) == 1
    c = ROOT.TLorentzVector(a)
    c += b
    c -= a
    c *= 2
    assert c == ROOT.TLorentzVector(0, 2, 0, 6)
    c *= ROOT.TRotation().RotateZ(-math.pi / 2)
    assert c.X() == pytest.approx(2)
    assert a.DeltaPhi(b) == pytest.approx(-math.pi / 2)
    p, q = ROOT.TLorentzVector(1, 0, 1, 3), ROOT.TLorentzVector(0, 1, -1, 3)
    assert p.DeltaR(q) == pytest.approx(math.hypot(p.Eta() - q.Eta(), p.DeltaPhi(q)))
    assert p.DrEtaPhi(q) == p.DeltaR(q)
    assert p.DrRapidityPhi(q) == p.DeltaR(q, True)


def test_a_four_vector_boosts_and_rotates_as_root_does():
    v = ROOT.TLorentzVector(1, 2, 3, 10)
    b = v.BoostVector()
    v.Boost(-b)
    assert (v.X(), v.Y(), v.Z()) == pytest.approx((0, 0, 0), abs=1e-12)
    assert v.T() == pytest.approx(math.sqrt(86))
    v.Boost(0, 0, 0)
    w = ROOT.TLorentzVector(1, 0, 0, 2)
    w.RotateZ(math.pi / 2)
    w.RotateX(0.1)
    w.RotateY(0.1)
    w.Rotate(0.2, ROOT.TVector3(0, 0, 1))
    w.RotateUz(ROOT.TVector3(0, 0, 1))
    assert w.P() == pytest.approx(1)
    assert w.Transform(ROOT.TRotation()).E() == 2


def test_a_genvector_keeps_the_coordinates_it_was_made_in():
    v = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105)
    assert v.M() == 0.105
    assert v.Pt() == 10
    assert v.Rho() == 10
    assert str(v) == "(10,1.2,0.4,0.105)"
    other = ROOT.TLorentzVector()
    other.SetPtEtaPhiM(10, 1.2, 0.4, 0.105)
    assert (v.Px(), v.X(), v.E(), v.P()) == pytest.approx(
        (other.Px(), other.X(), other.E(), other.P())
    )
    assert v.Theta() == pytest.approx(other.Theta())
    assert v.Rapidity() == pytest.approx(other.Rapidity())
    assert v.Mt() == pytest.approx(other.Mt())
    assert v.Et() == pytest.approx(other.Et())
    assert v.Beta() == pytest.approx(other.Beta())
    assert v.Gamma() == pytest.approx(other.Gamma())
    assert v.M2() == pytest.approx(other.M2())
    assert v.Perp2() == pytest.approx(100)
    assert repr(v).startswith("<ROOT.Math.PtEtaPhiMVector (10,")
    assert M.PxPyPzMVector(3, 0, 4, 0).E() == 5
    assert M.PxPyPzEVector(0, 0, 0, 1).Et() == 0
    assert M.PtEtaPhiEVector(1, 0, 0, 2).M() == pytest.approx(math.sqrt(3))
    assert M.PxPyPzEVector(0, 0, 3, 1).M() == pytest.approx(-math.sqrt(8))
    assert M.PxPyPzMVector(0, 0, 3, -1).E() == pytest.approx(math.sqrt(8))
    assert M.PtEtaPhiMVector(0, 0, 0, 0).Theta() == 0
    assert M.PxPyPzEVector(0, 0, 0, 1).Eta() == 0
    assert M.PxPyPzEVector(0, 0, 2, 3).Eta() == 2 + 22756
    assert M.PxPyPzEVector(0, 0, -2, 3).Eta() == -22758


def test_genvectors_add_in_the_left_ones_coordinates():
    a, b = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105), M.PxPyPzEVector(1, 2, 3, 10)
    s = a + b
    assert type(s).__name__ == "PtEtaPhiMVector"
    assert s.Px() == pytest.approx(a.Px() + 1)
    assert (b - b).E() == 0
    assert (-b).E() == -10
    assert (b * 2).E() == 20
    assert (2 * b).E() == 20
    assert (b / 2).E() == 5
    assert b * b == pytest.approx(86)
    assert b.Dot(b) == pytest.approx(86)
    c = M.PxPyPzEVector(b)
    c += b
    c -= b
    c *= 3
    assert c == M.PxPyPzEVector(3, 6, 9, 30)
    assert hash(c) == hash(M.PxPyPzEVector(3, 6, 9, 30))
    assert list(b) == [1, 2, 3, 10]
    assert b[3] == 10
    assert c != "c"
    assert M.XYZTVector is M.PxPyPzEVector
    assert M.LorentzVector["ROOT::Math::PtEtaPhiM4D<double>"] is M.PtEtaPhiMVector
    with pytest.raises(TypeError, match="no four-vector"):
        M.LorentzVector["Nothing"]


def test_genvectors_set_their_own_and_others_coordinates():
    v = M.PtEtaPhiMVector(10, 1.2, 0.4, 0.105)
    v.SetPt(20)
    v.SetM(1)
    assert v.Pt() == 20
    assert v.M() == 1
    v.SetPx(5)
    assert v.Px() == pytest.approx(5)
    w = M.PxPyPzEVector()
    w.SetPxPyPzE(1, 2, 3, 10)
    w.SetXYZT(1, 2, 3, 11)
    w.SetCoordinates(1, 2, 3, 12)
    got = array.array("d", [0] * 4)
    assert w.GetCoordinates(got) == (1, 2, 3, 12)
    assert got[3] == 12
    assert w.Coordinates() is w
    assert w.Vect().Z() == 3
    assert w.BoostToCM().Z() == pytest.approx(-0.25)
    assert M.PxPyPzEVector(1, 0, 0, 1).isLightlike()
    assert w.isTimelike()
    assert not w.isSpacelike()
    assert w.ColinearRapidity() == pytest.approx(
        0.5 * math.log((12 + math.sqrt(14)) / (12 - math.sqrt(14)))
    )
    assert M.PxPyPzEVector(v).M() == pytest.approx(v.M())


def test_three_and_two_dimensional_genvectors():
    v = M.XYZVector(1, 2, 3)
    assert str(v) == "(1,2,3)"
    assert v.R() == pytest.approx(math.sqrt(14))
    assert v.Mag2() == pytest.approx(14)
    assert v.Rho() == pytest.approx(math.sqrt(5))
    assert v.Perp2() == 5
    assert v.Theta() > 0
    assert v.Unit().R() == pytest.approx(1)
    assert M.XYZVector().Unit().R() == 0
    assert v.Cross(M.XYZVector(0, 0, 1)) == M.XYZVector(2, -1, 0)
    assert v.Dot(v) == 14
    assert v.SetXYZ(0, 0, 1) == M.XYZVector(0, 0, 1)
    p = M.Polar3DVector(2, math.pi / 2, 0)
    assert p.X() == pytest.approx(2)
    assert M.Polar3DVector(M.XYZVector()).Theta() == 0
    r = M.RhoEtaPhiVector(1, 0, 0)
    assert r.X() == pytest.approx(1)
    assert M.RhoZPhiVector(1, 2, 0).Z() == 2
    assert M.RhoZPhiVector(M.XYZVector(0, 1, 2)).Phi() == pytest.approx(math.pi / 2)
    assert M.DisplacementVector3D["ROOT::Math::Polar3D<double>"] is M.Polar3DVector
    assert M.DisplacementVector3D["CylindricalEta3D"] is M.RhoEtaPhiVector
    assert M.DisplacementVector3D["Cylindrical3D<double>"] is M.RhoZPhiVector
    assert M.DisplacementVector3D["Cartesian3D"] is M.XYZVector
    assert M.XYZPoint(1, 1, 1).X() == 1
    with pytest.raises(TypeError, match="no three-vector"):
        M.DisplacementVector3D["Nothing"]
    q = M.XYVector(3, 4)
    assert q.R() == 5
    assert q.Mag2() == 25
    assert q.Unit().R() == pytest.approx(1)
    assert M.XYVector().Unit().R() == 0
    assert M.Polar2DVector(2, 0).X() == 2
    q.SetR(10)
    assert q.X() == pytest.approx(6)


def test_vector_util_asks_of_two_vectors_at_once():
    a, b = M.PtEtaPhiMVector(10, 1.0, 0.1, 0), M.PtEtaPhiMVector(10, -1.0, 3.0, 0)
    assert M.VectorUtil.DeltaPhi(a, b) == pytest.approx(2.9)
    assert M.VectorUtil.DeltaPhi(M.XYZVector(1, 0, 0), M.XYZVector(-1, -0.1, 0)) < 0
    assert M.VectorUtil.DeltaR(a, b) == pytest.approx(math.hypot(2.9, 2.0))
    assert M.VectorUtil.DeltaR2(a, b) == pytest.approx(2.9**2 + 4)
    assert M.VectorUtil.DeltaRapidityPhi(a, b) > 0
    assert M.VectorUtil.InvariantMass(a, b) == pytest.approx((a + b).M())
    assert M.VectorUtil.InvariantMass2(a, b) == pytest.approx((a + b).M2())
    assert M.VectorUtil.CosTheta(M.XYZVector(1, 0, 0), M.XYZVector(1, 0, 0)) == pytest.approx(1)
    assert M.VectorUtil.CosTheta(M.XYZVector(), M.XYZVector(1, 0, 0)) == 1
    assert M.VectorUtil.Angle(M.XYZVector(1, 0, 0), M.XYZVector(0, 1, 0)) == pytest.approx(
        math.pi / 2
    )
    assert M.VectorUtil.Perp(M.XYZVector(1, 1, 0), M.XYZVector(1, 0, 0)) == pytest.approx(1)
    rest = M.VectorUtil.boost(
        M.PxPyPzEVector(1, 2, 3, 10), M.PxPyPzEVector(1, 2, 3, 10).BoostToCM()
    )
    assert rest.Px() == pytest.approx(0, abs=1e-12)
    assert rest.E() == pytest.approx(math.sqrt(86))
