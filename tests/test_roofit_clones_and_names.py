"""Clones, printed names, histograms by variable names, and the other small pieces RooFit's
tutorials ask for, each as ROOT 6.40 answers it."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.fitting.result import _global_cc
from xrdroot.roofit.functions import RooAddition
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous, _Projection
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar


def gaussian() -> tuple[Any, Any, Any, Any]:
    x = RooRealVar("x", "x", 0, -10, 10)
    m = RooRealVar("m", "m", 1, -5, 5)
    s = RooRealVar("s", "s", 2, 0.1, 10)
    return x, m, s, RooGaussian("g", "g", x, m, s)


def test_a_cloned_set_holds_the_same_members_under_a_new_name() -> None:
    """rf508's ``s3.Clone("sclone")``: the members themselves, not copies, as ROOT's is."""
    x, m, _s, _g = gaussian()
    held = RooArgSet(x, m)
    made = held.Clone("sclone")
    assert (made.GetName(), list(made), held.Clone().GetName()) == ("sclone", [x, m], "")


def test_a_collection_prints_as_cppyy_prints_it() -> None:
    """``print(s)`` under PyROOT is ``{ @0x..., @0x... }``: pointers to the members."""
    x, m, _s, _g = gaussian()
    assert re.fullmatch(r"\{ @0x[0-9a-f]+, @0x[0-9a-f]+ \}", str(RooArgList(x, m)))


def test_a_cloned_frame_has_copies_of_what_was_drawn_on_it() -> None:
    """rf205's ``xframe.Clone("xframe2")``: plotting on the copy leaves the original alone."""
    x, _m, _s, g = gaussian()
    frame = x.frame()
    g.plotOn(frame)
    made = frame.Clone("xframe2")
    g.plotOn(made)
    copied = made.getObject(0)
    assert (made.GetName(), made.numItems(), frame.numItems()) == ("xframe2", 2, 1)
    assert copied is not frame.getObject(0) and frame.Clone().GetName() == frame.GetName()


def test_an_event_past_the_end_of_a_dataset_is_none() -> None:
    """ROOT's ``get(i)`` past the end is a null set, and the current event stays as it was."""
    x, _m, _s, g = gaussian()
    generator().SetSeed(1)
    data = g.generate([x], 3)
    assert (data.get(3), data.get(-1), data.get(2) is not None) == (None, None, True)


def test_a_sum_of_products_of_two_lists_prints_both_lists() -> None:
    """``RooAddition(a, b)`` of two lists prints its proxies; of one, its terms joined by +."""
    x, m, s, _g = gaussian()
    assert RooAddition("ad", "ad", [x, m], [s, m]).printArgs() == "[ set=(x,m) set2=(s,m) ]"


def test_a_function_is_histogrammed_by_the_names_of_its_variables_and_their_bins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``createHistogram("x,y", 5, 4)``: ROOT's ``Binning(5)`` and ``YVar(y, Binning(4))``."""
    from xrdroot.roofit import histograms

    monkeypatch.setattr(histograms, "WRAP", [lambda made: made])  # the engine's own histogram
    _x, m, s, g = gaussian()
    y, z = RooRealVar("y", "y", 0, -5, 5), RooRealVar("z", "z", 0, -5, 5)
    gz = RooGaussian("gz", "gz", z, RooRealVar("mz", "mz", 0), RooRealVar("sz", "sz", 1))
    prod = RooProdPdf("p", "p", [g, RooGaussian("gy", "gy", y, m, s), gz])
    made = prod.createHistogram("x,y", 5, 4)
    unbinned = g.createHistogram("x")
    three = prod.createHistogram("x,y,z", 2, 2, 2)
    assert (made.name, made.shape, unbinned.shape, three.shape) == (
        "p__x_y",
        (5, 4),
        (100,),
        (2, 2, 2),
    )
    assert made.integral() == pytest.approx(1.0074047519919986, rel=1e-6)


def test_raising_the_minimum_above_the_maximum_warns_and_closes_the_range(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ROOT's ``setMin(20)`` on ``[3, 10]`` warns and makes the range ``[10, 10]``."""
    x = RooRealVar("x", "x", 3, 3, 10)
    x.setMin(20)
    assert (x.getMin(), x.getMax(), x.getVal()) == (10.0, 10.0, 10.0)
    assert capsys.readouterr().out == (
        "[#0] WARNING:InputArguments -- RooRealVar::setMin(x): Proposed new fit min. "
        "larger than max., setting min. to max.\n"
    )


def test_a_fit_without_a_covariance_has_no_global_correlations() -> None:
    """A matrix of zeros - Minuit gave none - has none to invert."""
    assert _global_cc(np.zeros((2, 2))).tolist() == [0.0, 0.0]


def test_a_simultaneous_projection_averages_its_channels_as_weighed() -> None:
    """The projection a ``ProjWData`` plot draws: normalised in itself, integrated exactly."""
    x, _m, s, g = gaussian()
    cat = RooCategory("c", "c", {"a": 0, "b": 1})
    e = RooGaussian("e", "e", x, RooRealVar("m2", "m2", -1), s)
    sim = RooSimultaneous("sim", "sim", {"a": g, "b": e}, cat)
    made = _Projection(sim, {"a": 0.25, "b": 0.75})
    names = frozenset(["x"])
    assert (made.selfNormalized(), made.analytic_names(names, None)) == (True, names)
    assert made.compute({"x": 0.0}) == pytest.approx(0.25 * g.getVal() + 0.75 * e.getVal())
    assert made.analytic(names, {}, None) == pytest.approx(1.0)


def test_a_sum_described_in_a_fit_names_each_term_with_its_own_coefficient() -> None:
    """With a coefficient for every term there is no ``[%]`` remainder to describe."""
    x, _m, s, g = gaussian()
    e = RooGaussian("e", "e", x, RooRealVar("m2", "m2", -1), s)
    a, b = RooRealVar("a", "a", 0.3, 0, 1), RooRealVar("b", "b", 0.7, 0, 1)
    add = RooAddPdf("add", "add", [g, e], [a, b])
    assert add.compiled_origin(frozenset(["x"])) == (
        "RooAddPdf::add[ a * g_over_g_Int[x] + b * e_over_e_Int[x] ]"
    )


def test_a_factor_of_parameters_the_others_do_not_share_is_no_constraint() -> None:
    """``getConstraints`` keeps a parameters-only factor only if another factor shares one."""
    _x, m, s, g = gaussian()
    k = RooRealVar("k", "k", 1, 0, 2)
    lone = RooGaussian("lone", "lone", k, 1.0, 0.1)
    prod = RooProdPdf("p", "p", [g, lone])
    assert prod.constraint_terms(frozenset(["x"]), [m, s, k], True) == []
    assert prod.constraint_terms(frozenset(["x"]), [k], False) == [lone]
