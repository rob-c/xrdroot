"""``RooMultiPdf``: the density a category's index selects."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT


@pytest.fixture
def indexed():
    x = ROOT.RooRealVar("x", "x", -10.0, 10.0)
    index = ROOT.RooCategory("pdfIndex", "pdf index")
    gauss = ROOT.RooGaussian("g", "g", x, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(1.0))
    expo = ROOT.RooExponential("e", "e", x, ROOT.RooFit.RooConst(-0.3))
    multi = ROOT.RooMultiPdf("multi", "multi", index, ROOT.RooArgList(gauss, expo))
    return x, index, gauss, expo, multi


def test_the_index_gets_one_state_per_density_in_the_order_given(indexed):
    _, index, _, _, multi = indexed
    assert index.numTypes() == 2
    assert [pdf.GetName() for pdf in multi.pdfList()] == ["g", "e"]
    assert multi.getCurrentIndex() == 0
    assert multi.getCurrentPdf().GetName() == "g"


def test_it_is_the_selected_density_which_is_what_its_value_says(indexed):
    x, index, gauss, expo, multi = indexed
    x.setVal(1.5)
    assert multi.getVal() == gauss.getVal()
    assert index.setIndex(1) is False         # ROOT's setters answer true on error
    assert multi.getCurrentIndex() == 1
    assert multi.getVal() == expo.getVal()
    assert multi.setCurrentIndex(0) is False  # back to the first
    assert multi.getVal() == gauss.getVal()


def test_anything_else_asked_of_it_is_asked_of_the_selected_density(indexed):
    _, _, gauss, _, multi = indexed
    assert multi.GetTitle() == "multi"
    assert multi.mean.getVal() == gauss.mean.getVal()
    assert "multi" in repr(multi)


def test_an_attribute_it_has_not_got_is_not_looked_for_on_the_selected_density(indexed):
    _, _, _, _, multi = indexed
    missing = "_no_such_private_name"
    with pytest.raises(AttributeError):
        assert getattr(multi, missing)


def test_a_multi_pdf_with_nothing_to_choose_between_says_so():
    index = ROOT.RooCategory("empty", "empty")
    with pytest.raises(ValueError, match="at least one density"):
        ROOT.RooMultiPdf("bad", "bad", index, ROOT.RooArgList())
