"""An array object is written as the class ROOT knows, not as anonymous numbers."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import FormatError, UnsupportedFeatureError


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("TArrayD", [1.0, 2.5], [1.0, 2.5]),
        ("TArrayF", [1, 2, 3], [1.0, 2.0, 3.0]),
        ("TArrayI", [4, 5], [4, 5]),
        ("TArrayC", [1, 2], [1, 2]),
    ],
)
def test_a_tarray_is_a_key_that_reads_back_as_itself(tmp_path, name, values, expected):
    """The key carries the ``TArray`` record, so the class survives the file."""
    path = tmp_path / f"{name}.root"
    with xrdroot.create(str(path)) as handle:
        handle["a"] = getattr(ROOT, name)(values)
    with xrdroot.open_root(str(path)) as handle:
        read = np.asarray(handle["a"])
        assert [float(value) for value in read] == [float(v) for v in expected]


def test_every_array_class_names_itself_as_root_does():
    """ROOT asks ``ClassName()`` of anything it writes, so every array answers it."""
    for name in ("TArrayC", "TArrayS", "TArrayI", "TArrayF", "TArrayD", "TArrayL"):
        assert getattr(ROOT, name)().ClassName() == name


def test_a_plain_object_is_not_mistaken_for_an_array(tmp_path):
    """The writer's array path needs a class it knows *and* values it can take."""
    with pytest.raises((FormatError, TypeError, UnsupportedFeatureError)):
        with xrdroot.create(str(tmp_path / "b.root")) as handle:
            handle["a"] = ROOT.TNamed("x", "a named thing")


def test_a_three_vector_is_written_as_its_coordinates(tmp_path):
    """ROOT's ``TVector3`` record is ``fX``, ``fY`` and ``fZ``, and nothing else."""
    import xrdroot

    path = tmp_path / "v.root"
    with xrdroot.create(str(path)) as handle:
        handle["v"] = ROOT.TVector3(1.5, -2.5, 3.0)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["v"]
    assert (read["fX"], read["fY"], read["fZ"]) == (1.5, -2.5, 3.0)


def test_a_lorentz_vector_is_written_as_a_three_vector_and_an_energy(tmp_path):
    """``TLorentzVector`` streams ``fP`` (a ``TVector3``) and ``fE``."""
    import xrdroot

    path = tmp_path / "lv.root"
    with xrdroot.create(str(path)) as handle:
        handle["p"] = ROOT.TLorentzVector(1.0, 2.0, 3.0, 4.0)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["p"]
    assert (read["fP"]["fX"], read["fP"]["fY"], read["fP"]["fZ"]) == (1.0, 2.0, 3.0)
    assert read["fE"] == 4.0


def test_a_matrix_is_written_as_its_shape_and_values(tmp_path):
    """ROOT's ``TMatrixT`` record is the base's shape, then a pointer counted by ``fNelems``."""
    import xrdroot

    path = tmp_path / "m.root"
    matrix = ROOT.TMatrixD(2, 2)
    matrix[0][0] = 1.5
    matrix[1][1] = -2.5
    with xrdroot.create(str(path)) as handle:
        handle["m"] = matrix
    with xrdroot.open_root(str(path)) as handle:
        read = handle["m"]
    base = read["TMatrixTBase<double>"]
    assert (base["fNrows"], base["fNcols"], base["fNelems"]) == (2, 2, 4)
    assert [float(v) for v in read["fElements"]] == [1.5, 0.0, 0.0, -2.5]


def test_a_random_engine_is_written_with_its_mersenne_state(tmp_path):
    """``TRandom3`` streams its name, seed and the 624-word state ROOT continues from."""
    import xrdroot

    path = tmp_path / "r.root"
    with xrdroot.create(str(path)) as handle:
        handle["r"] = ROOT.TRandom3(7)
    with xrdroot.open_root(str(path)) as handle:
        read = handle["r"]
    engine = read["TRandom"]
    assert engine["TNamed"]["fName"] == "Random3"
    assert len(read["fMt"]) == 624
    assert 0 <= read["fCount624"] <= 624


def test_a_single_precision_matrix_is_written_as_its_float_twin(tmp_path):
    """``TMatrixF`` streams as ``TMatrixT<float>``: the same shape, four-byte members."""
    import xrdroot

    path = tmp_path / "mf.root"
    matrix = ROOT.TMatrixF(2, 2)
    matrix[0][1] = 2.5
    with xrdroot.create(str(path)) as handle:
        handle["m"] = matrix
    with xrdroot.open_root(str(path)) as handle:
        read = handle["m"]
    base = read["TMatrixTBase<float>"]
    assert (base["fNrows"], base["fNcols"], base["fNelems"]) == (2, 2, 4)
    assert [float(v) for v in read["fElements"]] == [0.0, 2.5, 0.0, 0.0]


def test_a_tstring_names_itself_as_root_does():
    """``TObject``'s lookup reaches for these, so a ``TString`` spells them."""
    text = ROOT.TString("x")
    assert (text.ClassName(), text.GetName(), text.GetTitle()) == ("TString", "", "")


def test_a_tstring_key_is_written_as_its_characters(tmp_path):
    """ROOT writes a ``TString`` key as the length word and the bytes, nothing round them."""
    import xrdroot

    path = tmp_path / "s.root"
    with xrdroot.create(str(path)) as handle:
        handle["s"] = ROOT.TString("hello")
    assert path.stat().st_size > 0
    assert b"hello" in path.read_bytes()
