"""The RooFit engine's core nodes as RooStats and HistFactory lean on them.

Categories iterated as C++ iterates them, collections that clone and assign,
the deprecated ``CloneData`` command, sums of products, the raw error the
formatter reads, the progress line only a node's message ends, variables that
randomise themselves and densities that announce their normalisation once:
each held to what ROOT 6.40 printed or returned for the same calls.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterator
from typing import Any

import pytest

from xrdroot.roofit import pdf as pdf_module
from xrdroot.roofit.categories import RooCategory, State
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.formatting import format_var
from xrdroot.roofit.functions import RooAddition, RooProduct
from xrdroot.roofit.messages import INFO, PROGRESS, RooMsgService, log_plain, service
from xrdroot.roofit.pdfs.generic import RooGenericPdf
from xrdroot.roofit.real import RooAbsReal
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar, is_infinite


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


class Plain:
    """A value with a name and nothing else: neither clonable nor constant."""

    def __init__(self, name: str, value: float) -> None:
        self._name, self._value = name, value

    def GetName(self) -> str:
        return self._name

    def getVal(self) -> float:
        return self._value


def test_a_category_iterates_as_label_and_index_pairs_with_first_and_second() -> None:
    """``for s in cat`` gives ``std::pair``s - ``first`` the label, ``second`` the index."""
    cat = RooCategory("c", "c")
    cat.defineType("A", 3)
    cat.defineType("B", 7)
    states = list(cat)
    assert [(s.first, s.second) for s in states] == [("A", 3), ("B", 7)]
    assert states[0] == ("A", 3) and isinstance(states[0], State)
    assert cat.begin().first == "A"


def test_add_clone_adds_copies_refusing_a_name_already_held() -> None:
    """``addClone`` copies each member of what it is given; a set keeps its first of a name."""
    x, y = RooRealVar("x", "x", 1.0, 0, 5), RooRealVar("y", "y", 2.0, 0, 5)
    held = RooArgSet()
    assert held.addClone(RooArgList(x, y)) is True
    assert held.find("x") is not x and held.find("x").getVal() == 1.0
    assert held.addClone(x) is False
    plain = Plain("p", 3.0)
    assert held.addClone(plain) is True and held.find("p") is plain
    assert held.clone("named").GetName() == "named"


def test_assign_takes_values_and_constness_but_assign_value_only_takes_values() -> None:
    """``assign`` copies the constant flag too; ``assignValueOnly`` only the value."""
    mine = RooArgSet(RooRealVar("a", "a", 0.0, -5, 5), RooRealVar("b", "b", 0.0, -5, 5))
    other_a = RooRealVar("a", "a", 1.5, -5, 5)
    other_a.setConstant(True)
    mine.assign(RooArgSet(other_a, Plain("b", 2.5), Plain("zz", 9.0)))
    assert (mine.find("a").getVal(), mine.find("a").isConstant()) == (1.5, True)
    assert mine.find("b").getVal() == 2.5 and not mine.find("b").isConstant()
    other_a.setConstant(False)
    other_a.setVal(-1.0)
    mine.assignValueOnly(RooArgSet(other_a, Plain("zz", 1.0)))
    assert (mine.find("a").getVal(), mine.find("a").isConstant()) == (-1.0, True)


def test_assign_value_only_copies_a_category_by_its_index() -> None:
    """A member without a real value - a category - takes the other's index instead."""
    cat, other = RooCategory("c", "c"), RooCategory("c", "c")
    for one in (cat, other):
        one.defineType("A", 0)
        one.defineType("B", 1)
    other.setIndex(1)
    RooArgSet(cat).assignValueOnly(RooArgSet(other))
    assert cat.getIndex() == 1


def test_clone_data_is_ignored_with_roofits_notice_as_the_command_is_made(capsys: Any) -> None:
    """``CloneData(b)`` says, when made, that ``createNLL`` ignores it - its flag a digit."""
    RooCmdArg("CloneData", False)
    RooCmdArg("CloneData", True)
    assert capsys.readouterr().out == (
        "[#1] INFO:InputArguments -- The deprecated RooFit::CloneData(0) option passed to "
        "createNLL() is ignored.\n"
        "[#1] INFO:InputArguments -- The deprecated RooFit::CloneData(1) option passed to "
        "createNLL() is ignored.\n"
    )


def test_a_sum_of_two_lists_is_a_sum_of_products_named_as_roofit_names_them() -> None:
    """``RooAddition(name, title, a, b)`` sums ``a[i]*b[i]``, each a ``RooProduct`` of its own."""
    x, y = RooRealVar("x", "x", 2.0, 0, 5), RooRealVar("y", "y", 3.0, 0, 5)
    total = RooAddition("p", "p", RooArgList(x, y), RooArgList(y, y))
    assert [t.GetName() for t in total.terms] == ["p_[x_x_y]", "p_[y_x_y]"]
    assert all(isinstance(t, RooProduct) for t in total.terms)
    assert total.getVal() == 2.0 * 3.0 + 3.0 * 3.0
    assert total.printArgs() == "[ p_[x_x_y] + p_[y_x_y] ]"
    assert RooAddition("e", "e", RooArgList()).getVal() == 0.0


def test_a_product_takes_its_bins_from_the_first_binned_factor() -> None:
    """A product of a histogram function and a number is binned as the histogram is."""
    from xrdroot.roofit.data.datahist import RooDataHist
    from xrdroot.roofit.pdfs.histpdf import RooHistFunc

    x, c = RooRealVar("x", "x", 0, 10), RooRealVar("c", "c", 2.0)
    x.setBins(2)
    hist = RooHistFunc("hf", "hf", RooArgSet(x), RooDataHist("dh", "dh", RooArgSet(x)))
    product = RooProduct("p", "p", RooArgList(c, hist))
    assert list(product.bin_boundaries("x")) == [0.0, 5.0, 10.0]
    assert product.isBinnedDistribution(RooArgSet(x)) is True
    shaped = RooProduct("q", "q", RooArgList(hist, RooGenericPdf("f", "f", "x", RooArgList(x))))
    assert shaped.isBinnedDistribution(RooArgSet(x)) is False
    assert RooProduct("r", "r", RooArgList(c)).bin_boundaries("x") is None


def test_the_formatter_reads_the_raw_error_so_none_prints_no_error() -> None:
    """``format`` reads ``_error`` - ``-1`` for none - as RooFit does: ``x =  1.2``."""
    x = RooRealVar("x", "x", 1.23456, 0, 10)
    assert (format_var(x, 2, "NEU"), format_var(x, 3, "NE")) == ("x =  1.2", "x =  1.23")
    x.setError(0.0123)
    assert format_var(x, 2, "NEU") == "x =  1.235 +/- 0.012"


def test_a_plain_message_about_a_node_ends_a_line_of_progress_dots(capsys: Any) -> None:
    """``log_plain`` about a node after a progress message about one starts a new line."""
    node = RooRealVar("x", "x", 0.0)
    service().log(node, PROGRESS, "Fitting", ".")
    log_plain(node, INFO, "Fitting", "after")
    log_plain(None, INFO, "Fitting", " more")
    assert capsys.readouterr().out == "[#0] PROGRESS:Fitting -- .\n\nafter more"
    assert isinstance(service(), RooMsgService)


def test_randomize_draws_uniformly_from_roofits_generator(monkeypatch: Any) -> None:
    """``randomize`` at a fresh generator gives ROOT's first draw over ``[0, 10]``."""
    from xrdroot.roofit import rng

    monkeypatch.setattr(rng, "_GENERATOR", [])
    x = RooRealVar("x", "x", 1.23456, 0, 10)
    x.randomize()
    assert x.getVal() == 9.99741748906672
    assert generator() is rng.generator()


def test_randomize_refuses_an_unbounded_range_with_roofits_error(capsys: Any) -> None:
    """A variable without both ends keeps its value, and RooFit says why."""
    y = RooRealVar("y", "y", 1.0, -math.inf, math.inf)
    y.randomize()
    z = RooRealVar("z", "z", 1.0, 0.0, math.inf)
    z.randomize()
    assert (y.getVal(), z.getVal(), is_infinite(z.getMax())) == (1.0, 1.0, True)
    assert capsys.readouterr().out == (
        "[#0] ERROR:Generation -- y::RooRealVar:randomize: fails with unbounded fit range\n"
        "[#0] ERROR:Generation -- z::RooRealVar:randomize: fails with unbounded fit range\n"
    )


def test_a_bin_width_and_the_scale_types_are_roofits() -> None:
    """``getBinWidth`` of a uniform binning, and ``ScaleType``'s numbers ``0`` to ``3``."""
    x = RooRealVar("x", "x", 1, 0, 2)
    x.setBins(4)
    assert x.getBinWidth(1) == 0.5
    assert (RooAbsReal.Raw, RooAbsReal.Relative, RooAbsReal.NumEvent) == (0, 1, 2)
    assert RooAbsReal.RelativeExpected == 3


def test_forcing_numeric_integration_is_remembered() -> None:
    """``setForceNumInt`` sets what ``getForceNumInt`` reads, off until it is set."""
    x = RooRealVar("x", "x", 1, 0, 2)
    g = RooGenericPdf("g", "g", "x*x+1", RooArgList(x))
    assert g.getForceNumInt() is False
    g.setForceNumInt()
    assert g.getForceNumInt() is True


def test_a_density_announces_its_normalisation_once_per_set_and_per_copy(capsys: Any) -> None:
    """The numeric integral is said once for ``x``; a set it does not depend on is a unit
    normalisation, and a copy's cache starts empty, as a clone's does."""
    x, y = RooRealVar("x", "x", 1, 0, 2), RooRealVar("y", "y", 1, 0, 2)
    g = RooGenericPdf("g", "g", "x*x+1", RooArgList(x))
    first = g.getVal(RooArgSet(x))
    assert g.getVal(RooArgSet(x)) == first == pytest.approx(0.42857142857142855, rel=1e-12)
    assert g.getVal(RooArgSet(y)) == 2.0
    twin = copy.copy(g)
    twin.getVal(RooArgSet(x))
    line = (
        "[#1] INFO:NumericIntegration -- RooRealIntegral::init(g_Int[x]) using numeric "
        "integrator RooIntegrator1D to calculate Int(x)\n"
    )
    assert capsys.readouterr().out == line + line


def test_global_observables_of_a_single_density_are_generated_as_events() -> None:
    """``generateSimGlobal`` of a density that is not simultaneous is its ``generate``."""
    from xrdroot.roofit.pdfs.basic import RooGaussian

    x = RooRealVar("x", "x", 0, -5, 5)
    gauss = RooGaussian("g", "g", x, RooRealVar("m", "m", 0.0), RooRealVar("s", "s", 1.0))
    made = gauss.generateSimGlobal(RooArgSet(x), 3)
    assert made.numEntries() == 3


def test_range_checks_are_skipped_for_a_density_read_from_a_file(capsys: Any) -> None:
    """With ``UNCHECKED`` set, an unsafe parameter range warns of nothing."""
    s = RooRealVar("s", "s", 1.0, -1.0, 2.0)
    pdf_module.UNCHECKED[0] = True
    try:
        pdf_module.check_range(object(), [s], 0.0)
    finally:
        pdf_module.UNCHECKED[0] = False
    assert capsys.readouterr().out == ""


def test_create_profile_names_the_profile_after_the_function_and_its_parameters() -> None:
    """``createProfile(poi)`` is ``<nll>_Profile[poi]``, titled ``Profile of <title>``."""
    x, m = RooRealVar("x", "x", 1, 0, 2), RooRealVar("m", "m", 1, 0, 2)
    f = RooGenericPdf("f", "the f", "(x-m)*(x-m)", RooArgList(x, m))
    profile = f.createProfile(RooArgSet(m))
    assert (profile.GetName(), profile.GetTitle()) == ("f_Profile[m]", "Profile of the f")
