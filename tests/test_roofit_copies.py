"""C++'s ways of making and filling RooFit objects that Python spells otherwise.

``RooAddPdf model1(model)`` is a copy constructor; ``RooAbsData::None`` an
enum value Python's syntax will not name; ``RooDataHist::set`` a bin filled
by hand, its squared weight kept as ROOT keeps it; ``{{"label", &pdf}}`` a map.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.store import RooAbsData
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooGaussian, ref
from xrdroot.roofit.variables import RooRealVar


def _model() -> tuple[RooAddPdf, RooRealVar, RooRealVar]:
    x = RooRealVar("x", "x", 0.5, -5, 5)
    frac = RooRealVar("frac", "frac", 0.3, 0, 1)
    narrow = RooGaussian("narrow", "narrow", x, ref(0.0), ref(1.0))
    wide = RooGaussian("wide", "wide", x, ref(0.0), ref(3.0))
    return RooAddPdf("model", "the model", [narrow, wide], [frac]), x, frac


def test_a_node_made_from_a_node_of_its_class_is_a_copy_of_it_sharing_its_inputs() -> None:
    model, x, frac = _model()
    copy, renamed = RooAddPdf(model), RooAddPdf(model, "model1")
    assert copy is not model and copy.GetName() == "model" and copy.GetTitle() == "the model"
    assert renamed.GetName() == "model1" and model.GetName() == "model"
    assert copy._serial > model._serial
    assert copy.getVal(RooArgSet(x)) == model.getVal(RooArgSet(x))
    frac.setVal(0.8)
    assert renamed.getVal(RooArgSet(x)) == pytest.approx(model.getVal(RooArgSet(x)))


def test_a_node_of_another_class_is_no_copy_but_the_constructors_to_read() -> None:
    model, _, _ = _model()
    with pytest.raises(TypeError):
        RooGaussian(model)


def test_none_is_the_error_type_of_no_error_bars_by_its_cpp_name() -> None:
    assert getattr(RooAbsData, "None") == RooAbsData.NONE == 2


def test_a_bin_set_by_hand_keeps_its_weight_as_its_square_until_an_error_is_given() -> None:
    x = RooRealVar("x", "x", 0, 4)
    x.setBins(4)
    hist = RooDataHist("h", "h", RooArgSet(x))
    hist.set(1, 3.0, -1)
    assert list(hist.weights()) == [0.0, 3.0, 0.0, 0.0]
    assert list(hist.weights_squared()) == [0.0, 3.0, 0.0, 0.0]
    x.setVal(2.5)
    hist.set(RooArgSet(x), 5.0, 2.0)
    assert list(hist.weights_squared()) == [0.0, 3.0, 4.0, 0.0]
    # From the first error on, ROOT keeps the squares: a weight given none squares to one.
    hist.set(0, 7.0)
    assert np.array_equal(hist.weights(), [7.0, 3.0, 5.0, 0.0])
    assert list(hist.weights_squared()) == [1.0, 3.0, 4.0, 0.0]


def test_a_simultaneous_density_takes_a_cpp_map_initializer_of_label_and_density_pairs() -> None:
    from xrdroot.roofit.categories import RooCategory
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    model, x, _ = _model()
    other = RooGaussian("other", "other", x, ref(1.0), ref(1.0))
    sample = RooCategory("sample", "sample")
    sample.defineType("physics")
    sample.defineType("control")
    sim = RooSimultaneous("sim", "sim", [["control", other], ["physics", model]], sample)
    assert sim.getPdf("physics") is model and sim.getPdf("control") is other
    listed = RooSimultaneous("listed", "listed", [model, other], sample)
    assert listed.getPdf("physics") is model and listed.getPdf("control") is other


def test_a_datasets_store_is_its_columns_whatever_type_is_asked_for() -> None:
    """StandardHypoTestInvDemo: ``setDefaultStorageType(RooAbsData::Vector)`` and the
    conversions change nothing - the events are columns, always."""
    x = RooRealVar("x", "x", 0, 1)
    data = RooDataHist("h", "h", RooArgSet(x))
    RooAbsData.setDefaultStorageType(RooAbsData.Vector)
    data.convertToVectorStore()
    data.convertToTreeStore()
    assert (RooAbsData.Tree, RooAbsData.Vector, RooAbsData.Composite) == (0, 1, 2)
    assert data.numEntries() == 100
