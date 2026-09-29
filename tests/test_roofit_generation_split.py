"""The binned and the split generator contexts, against ROOT 6.40's own draws.

A histogram density is drawn bin by bin, as a weighted dataset of its bin
centres (``wu``); a simultaneous density asked for its category too draws
each state from its own context - every context set up before any is drawn
from, as ``RooSimSplitGenContext`` sets them up - and joins them in
``hmaster``, weighted only if a state's events are.
"""

from __future__ import annotations

from typing import Any

import xrdroot.pyroot as ROOT


def _model() -> tuple[Any, ...]:
    x = ROOT.RooRealVar("x", "x", 0, 10)
    x.setBins(5)
    h = ROOT.TH1D("h", "h", 5, 0, 10)
    for i, v in enumerate([1, 2, 3, 2, 1]):
        h.SetBinContent(i + 1, v)
    hp = ROOT.RooHistPdf("hp", "hp", {x}, ROOT.RooDataHist("dh", "dh", [x], h))
    ext = ROOT.RooExtendPdf("ext", "ext", hp, ROOT.RooRealVar("n", "n", 100, 0, 1000))
    cat = ROOT.RooCategory("c", "c")
    cat.defineType("a")
    cat.defineType("b")
    g = ROOT.RooGaussian("g", "g", x, ROOT.RooFit.RooConst(5), ROOT.RooFit.RooConst(1))
    ext2 = ROOT.RooExtendPdf("ext2", "ext2", g, ROOT.RooRealVar("m", "m", 30, 0, 100))
    return x, hp, ext, cat, ext2


def _seed(seed: int) -> None:
    ROOT.RooRandom.randomGenerator().SetSeed(seed)


def test_a_histogram_density_is_drawn_bin_by_bin_as_root_draws_it() -> None:
    x, hp, ext, _, _ = _model()
    _seed(5)
    assert ext.generate({x}, Extended=True).numEntries() == 86  # unbinned: the extension hides it
    data = hp.generate({x}, 50)
    assert (data.GetName(), data.numEntries(), data.sumEntries()) == ("wu", 5, 50.0)
    rows = [(data.get(i).getRealValue("x"), data.weight()) for i in range(5)]
    assert rows == [(1.0, 5.0), (3.0, 10.0), (5.0, 16.0), (7.0, 13.0), (9.0, 6.0)]


def _rows(data: Any) -> list[tuple[float, float]]:
    return [(data.get(i).getRealValue("x"), data.weight()) for i in range(data.numEntries())]


def test_a_binned_sum_is_drawn_poisson_varied_expected_or_to_its_rounded_yield() -> None:
    x, hp, _, _, _ = _model()
    ext = ROOT.RooRealSumPdf("ext", "ext", [hp], [ROOT.RooRealVar("n", "n", 100.4, 0, 1000)], True)
    _seed(3)
    assert [w for _, w in _rows(ext.generate({x}, Extended=True))] == [118, 201, 304, 220, 104]
    assert [w for _, w in _rows(ext.generate({x}))] == [80, 213, 327, 195, 89]
    assert [w for _, w in _rows(hp.generate({x}, 50, ExpectedData=True))] == [
        5.555555555555555, 11.11111111111111, 16.666666666666664, 11.11111111111111,
        5.555555555555555]  # fmt: skip
    assert [w for _, w in _rows(ext.generate({x}, ExpectedData=True))] == [
        100.4, 200.8, 301.20000000000005, 200.8, 100.4]  # fmt: skip
    assert ROOT.RooRandom.randomGenerator().Rndm() == 0.7871385004837066


def test_no_count_and_no_yield_is_an_empty_dataset() -> None:
    x, hp, _, _, _ = _model()
    assert hp.generate({x}).GetName() == "emptyData"


def test_a_simultaneous_density_draws_each_state_as_its_own_context() -> None:
    """The extended states' numbers each Poisson-varied - after every context is set up."""
    x, _, ext, cat, ext2 = _model()
    sim = ROOT.RooSimultaneous("sim", "sim", {"a": ext, "b": ext2}, cat)
    _seed(7)
    data = sim.generate({x, cat}, Extended=True)
    assert (data.GetName(), data.numEntries(), data.isWeighted()) == ("hmaster", 146, False)
    _seed(7)
    data = sim.generate({x, cat}, Extended=True)
    assert (data.sumEntries("c==c::a"), data.sumEntries("c==c::b")) == (109.0, 37.0)


def test_a_count_is_shared_between_the_states_as_they_expect_events() -> None:
    x, _, ext, cat, ext2 = _model()
    sim = ROOT.RooSimultaneous("sim", "sim", {"a": ext, "b": ext2}, cat)
    _seed(7)
    sim.generate({x, cat}, Extended=True)
    data = sim.generate({x, cat}, 60)
    assert (data.numEntries(), data.sumEntries("c==c::a")) == (60, 51.0)


def test_binned_states_make_the_joined_data_weighted() -> None:
    x, hp, _, cat, ext2 = _model()
    ext = ROOT.RooRealSumPdf("ext", "ext", [hp], [ROOT.RooRealVar("n", "n", 100, 0, 1000)], True)
    sim = ROOT.RooSimultaneous("sim", "sim", {"a": ext, "b": ext2}, cat)
    _seed(7)
    data = sim.generate({x, cat}, Extended=True)
    assert data.isWeighted()
    assert data.numEntries() == 5 + int(data.sumEntries("c==c::b"))


def test_a_simultaneous_density_of_states_without_yields_is_refused(capsys: Any) -> None:
    x, hp, _, cat, ext2 = _model()
    sim = ROOT.RooSimultaneous("sim2", "sim2", {"a": hp, "b": ext2}, cat)
    capsys.readouterr()
    assert sim.generate({x, cat}, 10) is None
    assert "All components of the simultaneous PDF must be extended PDFs." in (
        capsys.readouterr().out)  # fmt: skip


def test_a_draw_past_every_state_is_drawn_again(monkeypatch: Any) -> None:
    """``RooSimSplitGenContext``'s lookup of a state: a draw that finds none is drawn again."""
    from xrdroot.roofit.generation import split

    draws = iter([1.0, 0.75])

    class _Fake:
        def Rndm(self) -> float:
            return next(draws)

    monkeypatch.setattr(split, "generator", lambda: _Fake())
    assert split._shared([1.0, 3.0], 1) == [0.0, 1.0]


def test_a_sum_of_functions_not_binned_is_drawn_event_by_event() -> None:
    """Binned in ``x``, the histogram is drawn as bins even asked for ``y`` too, as in ROOT; a
    sum of a formula is not binned, and is drawn event by event."""
    x, hp, _, _, _ = _model()
    y = ROOT.RooRealVar("y", "y", 0, 1)
    assert hp.generate({x, y}, 3).GetName() == "wu"
    f = ROOT.RooFormulaVar("f", "f", "1+x", [x])
    s = ROOT.RooRealSumPdf("s", "s", [f], [ROOT.RooRealVar("n", "n", 100.4, 0, 1000)], True)
    data = s.generate({x}, 3)
    assert (data.GetName(), data.numEntries(), data.isWeighted()) == ("sData", 3, False)


def test_all_binned_draws_an_unbinned_density_bin_by_bin_as_root() -> None:
    """``AllBinned()``: RooFit's binned tag ``*`` - every density binned, a Gaussian too."""
    _seed(4357)
    y = ROOT.RooRealVar("y", "", -5, 5)
    y.setBins(5)
    gauss = ROOT.RooGaussian("gs", "", y, ROOT.RooRealVar("m", "", 0.5, -1, 2),
                             ROOT.RooRealVar("s", "", 1, 0.5, 2))  # fmt: skip
    ext = ROOT.RooExtendPdf("ext", "", gauss, ROOT.RooRealVar("n", "", 10, 0, 100))
    data = ext.generate({y}, ROOT.RooFit.Extended(), ROOT.RooFit.AllBinned())
    assert (data.GetName(), data.numEntries(), data.sumEntries()) == ("wu", 5, 15.0)
