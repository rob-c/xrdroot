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


def test_its_leaves_are_the_selected_densitys_plus_the_index(indexed):
    _, index, _, _, multi = indexed
    names = [one.GetName() for one in multi.leaves()]
    assert "x" in names and index.GetName() in names
    assert multi.GetTitle() == "multi"
    assert "multi" in repr(multi)


def test_the_index_is_not_listed_twice_when_the_density_already_carries_it(indexed):
    x, index, _, _, _ = indexed
    carrying = ROOT.RooPolynomial("p", "p", x, ROOT.RooArgList(index))
    multi = ROOT.RooMultiPdf("carrying", "carrying", index, ROOT.RooArgList(carrying))
    listed = [one.GetName() for one in multi.leaves()]
    assert listed.count(index.GetName()) == 1


def test_a_multi_pdf_with_nothing_to_choose_between_says_so():
    index = ROOT.RooCategory("empty", "empty")
    with pytest.raises(ValueError, match="at least one density"):
        ROOT.RooMultiPdf("bad", "bad", index, ROOT.RooArgList())


def test_a_component_with_no_observables_draws_the_empty_event():
    from xrdroot.roofit.generation.samplerconfig import NoObservables, sampler

    drawn = NoObservables().generate(4)
    assert drawn.shape == (4, 0)
    assert NoObservables().resample_ratio == 1.0
    assert "no observables" in repr(NoObservables())
    # the sampler the generation context asks for is that one, never a Foam of no dimension
    x = ROOT.RooRealVar("x", "x", -1.0, 1.0)
    gauss = ROOT.RooGaussian("g", "g", x, ROOT.RooFit.RooConst(0.0), ROOT.RooFit.RooConst(1.0))
    assert isinstance(sampler(gauss, frozenset(), lambda points: points, [], None), NoObservables)
