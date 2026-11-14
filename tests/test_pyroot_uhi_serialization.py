"""UHI's serialization of ROOT's histograms: to UHI's dictionary and JSON, and back."""

from __future__ import annotations

import json

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.errors import UnsupportedFeatureError


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_histogram_comes_back_from_json_as_it_went():
    io_json = pytest.importorskip("uhi.io.json")
    h = ROOT.TH2D("j2", "j2", 3, 0, 3, 2, 0, 2)
    h.Fill(-1, 1)
    h[...] = np.arange(6).reshape(3, 2)
    text = json.dumps(h, default=io_json.default)
    back = ROOT.TH2D(json.loads(text, object_hook=io_json.object_hook))
    expect(
        (back.GetName(), "j2"),
        (back.values(flow=True).tolist(), h.values(flow=True).tolist()),
        (ROOT.gDirectory.FindObject("j2") is h, True),
    )


def test_the_dictionary_says_the_class_the_axes_and_the_storage():
    h = ROOT.TH1I("d1", "", 4, 0, 2)
    ir = h._to_uhi_()
    expect(
        (ir["uhi_schema"], 1),
        (ir["writer_info"]["ROOT"]["class"], "TH1I"),
        (ir["writer_info"]["ROOT"]["version"], ROOT.__version__),
        (ir["axes"][0]["type"], "regular"),
        ((ir["axes"][0]["lower"], ir["axes"][0]["upper"], ir["axes"][0]["bins"]), (0.0, 2.0, 4)),
        (ir["storage"]["type"], "int"),
        (ROOT.TH1D("d2", "", 1, 0, 1)._to_uhi_()["storage"]["type"], "double"),
    )


def test_uneven_bins_and_weights_survive_the_round_trip():
    h = ROOT.TH1D("u", "", 3, np.array([0.0, 1.0, 3.0, 7.0]))
    h.Fill(2.0, 3.0)
    h.Fill(9.0, 2.0)
    ir = h._to_uhi_()
    back = ROOT.TH1D._from_uhi_(ir)
    expect(
        (ir["axes"][0]["type"], "variable"),
        (ir["storage"]["type"], "weighted"),
        (back.GetXaxis().GetBinUpEdge(3), 7.0),
        (back.values(flow=True).tolist(), [0.0, 0.0, 3.0, 0.0, 2.0]),
        (back.variances(flow=True).tolist(), [0.0, 0.0, 9.0, 0.0, 4.0]),
    )
    ir["storage"]["variances"] = ir["storage"]["variances"][1:-1]
    assert ROOT.TH1D(ir).variances().tolist() == [0.0, 9.0, 0.0]


def test_a_dictionary_without_a_name_makes_one_up():
    ir = ROOT.TH1D("n", "", 2, 0, 1)._to_uhi_()
    del ir["writer_info"]
    made = ROOT.TH1F(ir)
    assert made.GetName().startswith("h_uhi_") and made.ClassName() == "TH1F"


def test_a_profile_is_written_as_a_weighted_mean_and_not_read_back():
    p = ROOT.TProfile("pm", "", 2, 0, 2)
    p.Fill(0.5, 2.0)
    p.Fill(0.5, 4.0, 3.0)
    storage = p._to_uhi_()["storage"]
    expect(
        (storage["type"], "weighted_mean"),
        (storage["values"].tolist(), [0.0, 3.5, 0.0, 0.0]),
        (storage["sum_of_weights"].tolist(), [0.0, 4.0, 0.0, 0.0]),
        (storage["sum_of_weights_squared"].tolist(), [0.0, 10.0, 0.0, 0.0]),
    )
    plain = ROOT.TProfile("pu", "", 1, 0, 1)
    plain.Fill(0.5, 2.0)
    plain._xrd.members["fBinSumw2"] = np.zeros(0)  # one never filled with weights keeps none
    assert plain._to_uhi_()["storage"]["sum_of_weights_squared"].tolist() == [0.0, 1.0, 0.0]
    with pytest.raises(UnsupportedFeatureError, match="mean storage"):
        ROOT.TProfile(p._to_uhi_())


def test_a_dictionary_ROOT_cannot_read_is_refused():
    ir = ROOT.TH1D("bad", "", 2, 0, 1)._to_uhi_()
    ir["uhi_schema"] = 999
    with pytest.raises(ValueError, match="Unsupported UHI schema version: 999"):
        ROOT.TH1D(ir)
    with pytest.raises(ValueError, match="Unsupported UHI schema version: 999"):
        ROOT.TH1D._from_uhi_(ir)
    with pytest.raises(ValueError, match="missing required keys 'axes' and 'storage'"):
        ROOT.TH1D({"uhi_schema": 1})
    with pytest.raises(ValueError, match="not a valid UHI dictionary"):
        ROOT.TH1D._from_uhi_({"axes": []})
    ir["uhi_schema"] = 1
    ir["axes"][0] = {"type": "category_str", "categories": ["a"]}
    with pytest.raises(ValueError, match="Unsupported axis type for conversion to ROOT"):
        ROOT.TH1D(ir)
