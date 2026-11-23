"""Native workspace records preserve model identity, data, configuration and file cycles."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from refmachine import roots
from xrdroot import open_root
from xrdroot.cli.diff import Tolerance, compare
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit import rng
from xrdroot.roofit.io.processids import SESSION
from xrdroot.roofit.messages import service

DATA = Path(__file__).parent / "data/roofit-write-workspace-6.40.root"


@pytest.fixture(autouse=True)
def _session(tmp_path, monkeypatch):
    monkeypatch.setattr(rng, "_GENERATOR", [])
    with contextmanager(fresh)(tmp_path):
        service().reset()
        yield
        service().reset()


def native_workspace():
    with open_root(DATA) as file:
        return file["w"]


def small_workspace(name="w"):
    x = ROOT.RooRealVar("x", "observable", 0., 10.)
    mean = ROOT.RooRealVar("mean", "location", 5., 0., 10.)
    sigma = ROOT.RooRealVar("sigma", "width", 1., .1, 3.)
    model = ROOT.RooGaussian("g", "signal", x, mean, sigma)
    made = ROOT.RooWorkspace(name, "workspace")
    made.Import(model)
    return made


def test_a_native_workspace_round_trip_preserves_every_model_and_dataset_field(tmp_path):
    workspace = native_workspace()
    filename = tmp_path / "roundtrip.root"
    assert workspace.writeToFile(str(filename)) is True
    with open_root(DATA) as native, open_root(filename) as written:
        assert list(compare(native, written, Tolerance())) == []
        made = written["w"]
    _check_models(made)
    _check_binnings(made)
    _check_data_and_snapshots(made)


def _check_models(made):
    assert made.pdf("g").mean is made.var("mean")
    assert made.pdf("g").x is made.var("x")
    # Gaussian normalization calls libm erf, whose final bit can differ off ROOT's machine.
    assert made.pdf("recursive").getVal({made.var("x")}) == roots(
        .38100595856944175, rel=1e-12)
    assert made.pdf("polynomial").getVal({made.var("extra")}) == .09999999999999999
    assert made.pdf("g").getIntegratorConfig().epsAbs() == 2e-8


def _check_binnings(made):
    np.testing.assert_array_equal(made.var("extra").getBinning("variable").array(),
                                  [0., 1., 3., 7., 10.])
    assert (made.var("extra").getMin("side"), made.var("extra").getMax("side")) == (1., 3.)
    assert made.var("extra").getBinning("variable")._streamed_bounds == (False, True)
    assert not made.var("sigma").hasError() and not made.var("sigma").hasAsymError()


def _check_data_and_snapshots(made):
    assert made.set("observables").find("x") is made.var("x")
    assert made.getSnapshot("initial").find("mean") is not made.var("mean")
    assert made.data("modelData").column("x").tolist() == [
        3.7304989267140627, 4.873037251329387, 4.681709477098905,
        4.3479951846397356, 5.817292653024197]


def test_writing_again_in_update_mode_retains_both_cycles_and_process_identifiers(tmp_path):
    made = small_workspace()
    filename = tmp_path / "cycles.root"
    assert made.writeToFile(str(filename))
    made.var("mean").setVal(7.)
    assert made.writeToFile(str(filename), False)
    with open_root(filename) as file:
        assert file["w;1"].var("mean").getVal() == 5.
        assert file["w;2"].var("mean").getVal() == 7.
        assert file.classnames() == {"w": "RooWorkspace", "ProcessID0": "TProcessID",
                                    "ProcessID1": "TProcessID"}


def test_update_mode_creates_a_missing_file_like_root(tmp_path):
    filename = tmp_path / "new.root"
    assert small_workspace().writeToFile(str(filename), False)
    with open_root(filename) as file:
        assert file["w"].pdf("g").GetName() == "g"


def test_an_empty_workspace_has_no_process_id_key(tmp_path):
    filename = tmp_path / "empty.root"
    assert ROOT.RooWorkspace("w", "empty").writeToFile(str(filename))
    with open_root(filename) as file:
        assert file.classnames() == {"w": "RooWorkspace"}
        assert file["w"].GetTitle() == "empty"


def check_written_outer_workspace_key(file, outer_bytes):
    assert outer_bytes == file.GetKey("outer").GetNbytes()
    assert file.GetKey("outer").GetClassName() == "RooWorkspace"
    assert file.GetKey("ProcessID0").GetClassName() == "TProcessID"
    assert file.GetKey("ProcessID0").GetTitle() == SESSION.title


def check_written_inner_workspace_key(file, nested, inner_bytes):
    assert inner_bytes == nested.GetKey("inner").GetNbytes()
    assert nested.GetKey("inner").GetClassName() == "RooWorkspace"
    assert file.GetKey("ProcessID1") is None
    assert file.GetKey("ProcessID0").GetClassName() == "TProcessID"
    assert file.GetKey("ProcessID0").GetTitle() == SESSION.title


def check_reopened_nested_workspace_keys(file, outer_bytes, inner_bytes):
    assert outer_bytes == file.key("outer").nbytes
    assert inner_bytes == file["nested"].key("inner").nbytes
    assert file.key("ProcessID0").title == SESSION.title
    assert "ProcessID1" not in file.classnames()


def check_reopened_nested_workspace_graph(outer, inner):
    assert outer.var("mean").getVal() == 5.
    assert inner.var("mean").getVal() == 5.
    assert outer is not inner
    assert outer.pdf("g").mean is outer.var("mean")
    assert inner.pdf("g").mean is inner.var("mean")


def test_workspace_write_tracks_its_keys_in_a_file_and_in_a_subdirectory(tmp_path):
    with ROOT.TFile(str(tmp_path / "write.root"), "RECREATE") as file:
        made = small_workspace()
        outer_bytes = made.Write("outer")
        check_written_outer_workspace_key(file, outer_bytes)
        nested = file.mkdir("nested")
        nested.cd()
        inner_bytes = made.Write("inner")
        check_written_inner_workspace_key(file, nested, inner_bytes)
    with open_root(tmp_path / "write.root") as file:
        check_reopened_nested_workspace_keys(file, outer_bytes, inner_bytes)
        outer, inner = file["outer"], file["nested/inner"]
        check_reopened_nested_workspace_graph(outer, inner)


def test_an_empty_workspace_can_be_written_through_the_current_file(tmp_path):
    with ROOT.TFile(str(tmp_path / "empty-write.root"), "RECREATE") as file:
        written = ROOT.RooWorkspace("w", "").Write()
        assert written == file.GetKey("w").GetNbytes()
        assert file.GetKey("ProcessID0") is None
    with open_root(tmp_path / "empty-write.root") as file:
        assert written == file.key("w").nbytes


def test_writing_without_a_file_returns_zero():
    assert small_workspace().Write() == 0


def test_files_that_cannot_be_opened_return_false(tmp_path):
    made = small_workspace()
    assert made.writeToFile(str(tmp_path / "missing/target.root")) is False
    invalid = tmp_path / "invalid.root"
    invalid.write_text("This is not a ROOT file.")
    assert made.writeToFile(str(invalid), False) is False


def test_unsupported_models_are_refused_before_an_existing_file_is_truncated(tmp_path):
    x = ROOT.RooRealVar("x", "x", 0., 10.)
    slope = ROOT.RooRealVar("slope", "slope", -.2)
    model = ROOT.RooExponential("e", "e", x, slope)
    made = ROOT.RooWorkspace("w", "w")
    made.Import(model)
    filename = tmp_path / "existing.root"
    filename.write_bytes(b"Existing content must survive an unsupported export.")
    before = filename.read_bytes()
    with pytest.raises(UnsupportedFeatureError, match="Writing the workspace class RooExponential"):
        made.writeToFile(str(filename))
    assert filename.read_bytes() == before


@pytest.mark.parametrize("kind", ["weighted", "categorical", "binned"])
def test_dataset_forms_not_yet_written_are_refused_without_silently_losing_fields(kind, tmp_path):
    x = ROOT.RooRealVar("x", "x", 0., 10.)
    if kind == "weighted":
        weight = ROOT.RooRealVar("weight", "weight", 1., 0., 10.)
        data = ROOT.RooDataSet("d", "d", {x, weight}, WeightVar="weight")
    elif kind == "categorical":
        category = ROOT.RooCategory("c", "c", {"A": 2, "B": -3})
        data = ROOT.RooDataSet("d", "d", {x, category})
    else:
        data = ROOT.RooDataHist("d", "d", {x})
    made = ROOT.RooWorkspace("w", "w")
    made.Import(data)
    with pytest.raises(UnsupportedFeatureError, match="workspace dataset"):
        made.writeToFile(str(tmp_path / "unsupported.root"))


def test_plot_labels_and_reference_ranges_are_persisted(tmp_path):
    made = native_workspace()
    made.var("x").setRange("side", 1., 3.)
    made.var("x").setUnit("GeV")
    made.var("x").setPlotLabel("energy")
    made.pdf("b").selectNormalizationRange("side")
    made.pdf("model").setNormRange("side")
    filename = tmp_path / "labels.root"
    assert made.writeToFile(str(filename))
    with open_root(filename) as file:
        found = file["w"]
    assert found.var("x").getUnit() == "GeV"
    assert found.var("x").getPlotLabel() == "energy"
    assert found.pdf("b")._reference == "side"
    assert found.pdf("model")._norm_range == "side"


@pytest.mark.parametrize("kind", ["generic", "embedded"])
def test_unimplemented_workspace_payloads_are_refused_before_writing(kind, tmp_path):
    made = small_workspace()
    if kind == "generic":
        made.Import(ROOT.RooStats.ModelConfig("model_config"))
    else:
        data = ROOT.RooDataHist("bins", "bins", {made.var("x")})
        made.Import(ROOT.RooHistFunc("histogram", "histogram", {made.var("x")}, data))
    with pytest.raises(UnsupportedFeatureError, match="generic or embedded"):
        made.writeToFile(str(tmp_path / "unsupported.root"))


def test_parameterized_ranges_are_refused_instead_of_becoming_fixed_ranges(tmp_path):
    made = small_workspace()
    low = ROOT.RooRealVar("low", "low", 0.)
    high = ROOT.RooRealVar("high", "high", 10.)
    made.var("x").setRange("moving", low, high)
    with pytest.raises(UnsupportedFeatureError, match="parameterized workspace binnings"):
        made.writeToFile(str(tmp_path / "moving.root"))


def test_variables_cloned_together_keep_their_shared_binning_records(tmp_path):
    x = ROOT.RooRealVar("x", "x", 0., 10.)
    x.setRange("side", 1., 3.)
    mean = x.Clone("mean")
    sigma = ROOT.RooRealVar("sigma", "sigma", 1., .1, 3.)
    made = ROOT.RooWorkspace("w", "w")
    made.Import(ROOT.RooGaussian("g", "g", x, mean, sigma))
    assert made.var("x")._shared is made.var("mean")._shared
    assert made.writeToFile(str(tmp_path / "shared.root"))
    with open_root(tmp_path / "shared.root") as file:
        found = file["w"]
    assert found.var("x").getMin("side") == found.var("mean").getMin("side") == 1.


def test_workspace_imports_own_dataset_rows_and_mark_only_their_cloned_observables():
    x = ROOT.RooRealVar("x", "x", 0., 10.)
    data = ROOT.RooDataSet("d", "d", {x})
    data.add({x})
    made = ROOT.RooWorkspace("w", "w")
    made.Import(data)
    assert made.data("d") is not data
    assert made.data("d").get().find("x") is not made.var("x")
    assert not made.data("d").get().find("x").getAttribute("SnapShot_ExtRefClone")
    assert made.var("x").getAttribute("SnapShot_ExtRefClone")
    assert not data.get().find("x").getAttribute("SnapShot_ExtRefClone")
    data.add({x})
    assert (data.numEntries(), made.data("d").numEntries()) == (2, 1)
    assert made.obj("d") is made.data("d")
    assert made.data("d").get(0).find("x").getVal() == 5.


def test_native_shared_binning_records_reopen_as_shared_mutable_properties(tmp_path):
    made = small_workspace()
    made.var("x").setRange("side", 1., 3.)
    made.var("mean")._shared = made.var("x")._shared
    path = tmp_path / "shared-native-properties.root"
    assert made.writeToFile(str(path))
    with open_root(path) as file:
        found = file["w"]
    assert found.var("x")._shared is found.var("mean")._shared
    found.var("x").setRange("side", 2., 4.)
    assert found.var("mean").getMin("side") == 2.
    assert found.var("mean").getMax("side") == 4.


def test_dataset_global_observables_remain_an_owned_snapshot_after_file_round_trip(tmp_path):
    x = ROOT.RooRealVar("x", "x", 2., 0., 10.)
    aux = ROOT.RooRealVar("aux", "auxiliary measurement", 3.)
    aux.setError(.4)
    data = ROOT.RooDataSet("d", "d", ROOT.RooArgSet(x))
    data.add(ROOT.RooArgSet(x))
    data.setGlobalObservables(ROOT.RooArgSet(aux))
    aux.setVal(7.)
    made = ROOT.RooWorkspace("w", "w")
    made.Import(data)
    path = tmp_path / "global-observables.root"
    assert made.writeToFile(str(path))
    with open_root(path) as file:
        found = file["w"].data("d").getGlobalObservables().find("aux")
    assert found.getVal() == 3.
    assert found.getError() == .4
    assert found.isConstant()
    assert found.GetTitle() == "auxiliary measurement"


def test_category_ranges_are_refused_before_an_existing_file_can_be_truncated(tmp_path):
    category = ROOT.RooCategory("category", "category", {"signal": 2, "control": -3})
    category.setRange("selected", "signal")
    made = ROOT.RooWorkspace("w", "w")
    made.Import(category)
    path = tmp_path / "keep.root"
    path.write_bytes(b"Existing bytes must survive a refused category-range export.")
    original = path.read_bytes()
    with pytest.raises(UnsupportedFeatureError, match="category named ranges"):
        made.writeToFile(str(path))
    assert path.read_bytes() == original



def coefficient_workspace():
    made = small_workspace()
    x = made.var("x")
    background = ROOT.RooPolynomial("background", "background", x, ROOT.RooArgList())
    fraction = ROOT.RooRealVar("fraction", "fraction", .4, 0., 1.)
    model = ROOT.RooAddPdf("model", "model", ROOT.RooArgList(made.pdf("g"), background),
                           ROOT.RooArgList(fraction))
    made.Import(model)
    return made


def test_reserved_coefficient_normalization_is_refused_before_truncating_a_file(tmp_path):
    made = coefficient_workspace()
    made.pdf("model").setStringAttribute("ref_coef_norm", "x")
    path = tmp_path / "keep.root"
    path.write_bytes(b"Existing bytes must survive a refused coefficient-normalization export.")
    original = path.read_bytes()
    with pytest.raises(
        UnsupportedFeatureError, match="Writing RooAddPdf fixed coefficient normalization"
    ):
        made.writeToFile(str(path))
    assert path.read_bytes() == original


@pytest.mark.parametrize("name,value", [("ref_coef_norm", ""), ("analysis_note", "x")])
def test_unrelated_or_empty_attributes_do_not_become_fixed_coefficient_definitions(
    name, value, tmp_path
):
    made = coefficient_workspace()
    made.pdf("model").setStringAttribute(name, value)
    path = tmp_path / "attributes.root"
    assert made.writeToFile(str(path))
    with open_root(path) as file:
        found = file["w"].pdf("model")
    assert found.getStringAttribute(name) == value


@pytest.mark.parametrize("name", ["w", "ProcessID0"])
def test_workspace_write_returns_its_own_encoded_key_bytes_even_when_metadata_shares_the_name(
    name, tmp_path
):
    path = tmp_path / "byte-count.root"
    with ROOT.TFile(str(path), "RECREATE") as file:
        written = small_workspace().Write(name)
        key = file.GetKey(name, 1)
        assert key.GetClassName() == "RooWorkspace"
        assert written == key.GetNbytes()
        metadata = (key.GetNbytes(), key.GetObjlen(), key.GetKeylen(), key.GetSeekKey())
    with open_root(path) as file:
        key = file.key(name + ";1")
        assert written == key.nbytes
        assert metadata == (key.nbytes, key.objlen, key.keylen, key.seek_key)
        made = file[name + ";1"]
        assert made.pdf("g").mean is made.var("mean")


def test_repeated_workspace_writes_return_each_actual_cycle_size_and_keep_its_metadata(tmp_path):
    path = tmp_path / "byte-cycles.root"
    with ROOT.TFile(str(path), "RECREATE") as file:
        made = small_workspace()
        first_bytes = made.Write()
        made.var("mean").setVal(7.)
        second_bytes = made.Write()
        assert first_bytes == file.GetKey("w", 1).GetNbytes()
        assert second_bytes == file.GetKey("w", 2).GetNbytes()
        assert file.GetKey("ProcessID1") is None
    with open_root(path) as file:
        assert first_bytes == file.key("w;1").nbytes
        assert second_bytes == file.key("w;2").nbytes
        assert file["w;1"].var("mean").getVal() == 5.
        assert file["w;2"].var("mean").getVal() == 7.


def test_replacing_a_snapshot_retains_only_the_new_values_and_constness_with_exact_diagnostics(
    capsys
):
    made = small_workspace()
    assert made.saveSnapshot("fit", "mean") is True
    assert made.getSnapshot("fit").find("mean").getVal() == 5.
    made.var("mean").setVal(7.)
    made.var("mean").setConstant(True)
    capsys.readouterr()
    assert made.saveSnapshot("fit", "mean") is True
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::saveSnapshot(w) "
        "replacing previous snapshot with name fit\n"
    )
    saved = made.getSnapshot("fit").find("mean")
    assert (saved.getVal(), saved.isConstant()) == (7., True)
    made.var("mean").setVal(4.)
    made.var("mean").setConstant(False)
    assert made.loadSnapshot("fit") is True
    assert (made.var("mean").getVal(), made.var("mean").isConstant()) == (7., True)
