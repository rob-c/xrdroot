"""``RooEfficiency``, generating its category event by event, and plotting ``Efficiency(cat)``.

The intervals are ``RooHistError::getBinomialIntervalEff``'s as ROOT 6.40
gives them, which its root search finds to a few ulps of the beta quantiles
here: 1e-12 is a thousand times that and a millionth of the bar's width.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdfs.basic import RooPolynomial, ref
from xrdroot.roofit.pdfs.efficiency import RooEfficiency
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.plot.efficiency import binomial_interval, from_efficiency
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

#: ROOT 6.40's intervals of ``n`` of ``n + m``.
ROOTS = {
    (0, 0): (-1.0, 1.0),
    (0, 5): (0.0, 0.0920381495684605),
    (5, 0): (0.9079618504315395, 1.0),
    (1, 9): (0.03622232785787077, 0.22370017023796304),
    (7, 3): (0.5692908152548807, 0.8045381027156078),
    (60, 70): (0.43605917582375053, 0.48733918477682014),
}


def _efficiency() -> tuple[RooEfficiency, RooRealVar, RooCategory, RooRealVar]:
    x = RooRealVar("x", "x", -10, 10)
    a = RooRealVar("a", "a", 0.4, 0, 1)
    eff = RooFormulaVar("effFunc", "(1-a)+a*cos((x+1)/5)", [a, x])
    cut = RooCategory("cut", "cut")
    cut.defineType("accept", 1)
    cut.defineType("reject", 0)
    return RooEfficiency("effPdf", "effPdf", eff, cut, "accept"), x, cut, a


@pytest.mark.parametrize(("counts", "interval"), sorted(ROOTS.items()))
def test_the_binomial_interval_is_roots(counts: tuple[int, int], interval: tuple) -> None:
    assert binomial_interval(*counts) == pytest.approx(interval, abs=1e-12)


def test_an_efficiency_is_the_function_held_to_one_for_accepted_and_its_complement() -> None:
    pdf, x, cut, a = _efficiency()
    x.setVal(-1.0)
    cut.setLabel("accept")
    assert pdf.getVal() == pytest.approx(1.0) and pdf.getVal(RooArgSet(cut)) == pytest.approx(1.0)
    a.setVal(0.9)
    x.setVal(9.0)  # 0.1 + 0.9 cos(2) < 0: held to nothing accepted, all rejected
    assert pdf.getVal() == 0.0
    cut.setLabel("reject")
    assert pdf.getVal() == 1.0
    assert pdf.analytic(frozenset({"cut"}), {"x": np.array([1.0, 2.0])}, None).tolist() == [1, 1]
    assert pdf.analytic(frozenset({"cut"}), {}, None) == 1.0
    assert pdf.analytic_names(frozenset({"cut", "x"}), None) == frozenset()
    assert pdf.printMetaArgs() == ""


def test_a_conditional_category_is_drawn_at_each_events_value_and_plotted_as_a_fraction() -> None:
    pdf, x, cut, _ = _efficiency()
    generator().SetSeed(4357)
    model = RooProdPdf("model", "model", [RooPolynomial("shape", "shape", x, [ref(-0.095)])],
                       Conditional=(pdf, cut))  # fmt: skip
    data = model.generate([x, cut], 2000)
    states = data.column("cut")
    assert data.numEntries() == 2000 and 0.75 < float(np.mean(states)) < 0.87
    edges = np.linspace(-10, 10, 5)
    made = from_efficiency("h", edges, [3, 0, 60, 0], [1, 0, 70, 5], 0.5)
    assert made.y.tolist() == [0.75, 0.0, 60 / 130, 0.0] and made.entries == 139.0
    assert made.members["fEXlow"][0] == 1.25
    assert isinstance(data, RooDataSet)


def test_a_category_conditional_on_two_reals_says_how_many_trials_and_raises_its_maximum(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """rf702's warnings: two real dimensions, and trials past RooAcceptReject's threshold - here
    with fewer trials than ROOT's hundred thousand, that the test be quick."""
    from xrdroot.roofit.generation import acceptreject

    monkeypatch.setattr(acceptreject, "TRIALS", (10, 10, 20, 10))
    monkeypatch.setattr(acceptreject, "MANY_TRIALS", 30)
    _, x, cut, _ = _efficiency()
    y = RooRealVar("y", "y", 0, 1)
    shape = RooFormulaVar("e2", "0.5+0.4*y*cos(x)", [x, y])
    both = RooEfficiency("both", "both", shape, cut, "accept")
    capsys.readouterr()
    generator().SetSeed(4357)
    label = "both_Int[]_Norm[cut]"
    made = acceptreject.AcceptRejectContext(both, frozenset({"cut"}), frozenset({"x", "y"}), label)
    shown = capsys.readouterr().out
    assert f"ctor({label}) WARNING: performing accept/reject sampling on a p.d.f in 2" in shown
    assert "40 trial samples requested by p.d.f for 2-dimensional" in shown
    made.maximum = 1e-9  # a maximum too low is raised by the first value above it
    assert made.event(1)["cut"] in (0, 1)


def test_data_plotted_as_an_efficiency_are_each_bins_fraction_accepted() -> None:
    from xrdroot.roofit.plot.frame import make_frame

    _, x, cut, _ = _efficiency()
    data = RooDataSet("d", "d", [x, cut])
    for i in range(10):
        x.setVal(-9 + 2 * i)
        cut.setLabel("accept" if i % 2 else "reject")
        data.add(RooArgSet(x, cut))
    frame = make_frame(x, (), {"Bins": 5})
    data.plotOn(frame, Efficiency=cut)
    shown = frame.items[-1][0]
    assert shown.y.tolist() == [0.5] * 5 and shown.GetName() == "h_d_Eff[cut]"
    assert shown.y_label == "Efficiency of cut=accept"


def test_a_density_times_an_efficiency_is_normalised_and_drawn_from_the_densitys_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """rf703: ``RooEffProd`` of an exponential and a turn-on - its events the exponential's
    kept with the efficiency's chance, and no integral of the product made to draw them."""
    from xrdroot.roofit.pdfs.basic import RooExponential
    from xrdroot.roofit.pdfs.effprod import RooEffProd

    t = RooRealVar("t", "t", 0, 5)
    tau = RooRealVar("tau", "tau", -1.54, -4, -0.1)
    eff = RooFormulaVar("eff", "0.5*(TMath::Erf((t-1)/0.5)+1)", [t])
    model = RooEffProd("modelEff", "model with efficiency", RooExponential("m", "m", t, tau), eff)
    t.setVal(0.2)
    low = model.getVal(RooArgSet(t))
    t.setVal(2.0)
    assert low < model.getVal(RooArgSet(t))  # the turn-on outweighs the fall below one
    capsys.readouterr()
    generator().SetSeed(4357)
    drawn = model.generate([t], 500).column("t")
    assert len(drawn) == 500 and float(np.mean(drawn < 0.5)) < 0.1  # the bare density: 0.54
    assert "RooRealIntegral" not in capsys.readouterr().out
