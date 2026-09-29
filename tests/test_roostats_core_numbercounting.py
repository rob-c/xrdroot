"""RooStats' number counting: the factory's model and data, and the Z_Bi significances.

``rs_numberCountingCombination``'s model and data, the tau it sets for each
bin, the ranges it gives the observables and backgrounds, and the p-value and
interval a ProfileLikelihoodCalculator finds on it, as ROOT 6.40 gave them;
and ``rs_numbercountingutils``' p-values and significances.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats.numbercounting import NumberCountingPdfFactory

U = ROOT.RooStats.NumberCountingUtils


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def said(name: str, value: str, data: str) -> str:
    """The factory's warning for a bin's ``tau``, as ROOT words it."""
    return (
        f"[#0] WARNING:ObjectHandling -- NumberCountingPdfFactory: changed value of {name} to "
        f"{value} to be consistent with background and its uncertainty.  Also stored these "
        f"values of tau into workspace with name . {name}{data} if you test with a different "
        "dataset, you should adjust tau appropriately.\n\n"
    )


def ranges(w: Any, *names: str) -> list[tuple[float, float, float]]:
    return [(w.var(n).getVal(), w.var(n).getMin(), w.var(n).getMax()) for n in names]


def test_the_expected_data_sets_tau_and_the_ranges_as_root_does(capsys: Any) -> None:
    """ROOT 6.40: tau 100.01 for 1% on 100, ``b`` up to 208, ``x`` to 1200, ``y`` to 100010."""
    f = ROOT.RooStats.NumberCountingPdfFactory()
    w = ROOT.RooWorkspace()
    f.AddModel([20.0, 10.0], 2, w, "TopLevelPdf", "masterSignal")
    f.AddExpData([20.0, 10.0], [100.0, 100.0], [0.01, 0.01], 2, w, "ExpectedNumberCountingData")
    assert capsys.readouterr().out == (
        said("tau_0", "100.01", "ExpectedNumberCountingData")
        + said("tau_1", "100.01", "ExpectedNumberCountingData")
    )
    assert ranges(w, "tau_0", "tau_1", "b_0", "x_0", "y_0") == [
        (100.00999900019995, 0.0, 1000.0999900019996),
        (100.00999900019995, 0.0, 1000.0999900019996),
        (100.0, 0.0, 208.0),
        (120.0, 0.0, 1200.0),
        (10000.999900019995, 0.0, 100009.99900019995),
    ]
    assert w.var("tau_0ExpectedNumberCountingData").getVal() == 100.00999900019995
    data = w.data("ExpectedNumberCountingData")
    assert data.numEntries() == 1
    assert [v.GetName() for v in data.get()] == ["x_0", "y_0", "x_1", "y_1"]
    assert [p.GetName() for p in w.pdf("TopLevelPdf").pdfList()] == [
        "sigRegion_0", "sideband_0", "sigRegion_1", "sideband_1",
    ]


def test_the_combination_excludes_no_signal_as_roots_tutorial_does(capsys: Any) -> None:
    """ROOT 6.40: p 0.01529395569707792 (Z 2.1623928), interval [0.0890690, 2.0012684]."""
    f = NumberCountingPdfFactory()
    w = ROOT.RooWorkspace()
    f.AddModel([20.0, 10.0], 2, w, "TopLevelPdf", "masterSignal")
    f.AddExpData([20.0, 10.0], [100.0, 100.0], [0.01, 0.01], 2, w, "ExpectedNumberCountingData")
    mu = w.var("masterSignal")
    null = ROOT.RooArgSet("nullParams")
    null.addClone(mu)
    null.setRealValue("masterSignal", 0)
    plc = ROOT.RooStats.ProfileLikelihoodCalculator(
        w.data("ExpectedNumberCountingData"), w.pdf("TopLevelPdf"), ROOT.RooArgSet(mu), 0.05, null
    )
    result = plc.GetHypoTest()
    assert result.NullPValue() == pytest.approx(0.01529395569707792, rel=1e-9)
    assert result.Significance() == pytest.approx(2.1623928089004956, rel=1e-9)
    plc.SetParameters(null)
    interval = plc.GetInterval()
    assert interval.LowerLimit(mu) == pytest.approx(0.08906896842211953, rel=1e-6)
    assert interval.UpperLimit(mu) == pytest.approx(2.0012683535164273, rel=1e-6)
    null.setRealValue("masterSignal", 0.0)
    assert not interval.IsInInterval(null)
    null.setRealValue("masterSignal", 2.0)
    assert interval.IsInInterval(null)


def test_data_with_a_sideband_takes_its_tau_and_fits_the_ranges(capsys: Any) -> None:
    """ROOT 6.40: the given tau of 100; the second dataset leaves the ranges it finds wide
    enough; the expected one is ``s + b`` and ``b tau``."""
    f = NumberCountingPdfFactory()
    w = ROOT.RooWorkspace()
    f.AddModel([20.0, 10.0], 2, w)
    f.AddDataWithSideband([123.0, 117.0], [11123.0, 9876.0], [100.0, 100.0], 2, w, "Side")
    f.AddExpDataWithSideband([20.0, 10.0], [100.0, 100.0], [100.0, 100.0], 2, w)
    assert capsys.readouterr().out == (
        said("tau_0", "100", "Side") + said("tau_1", "100", "Side")
        + said("tau_0", "100", "ExpectedNumberCountingData")
        + said("tau_1", "100", "ExpectedNumberCountingData")
    )
    assert w.pdf("CombinedPdf") is not None
    assert ranges(w, "tau_0", "b_0", "x_0", "y_0", "y_1") == [
        (100.0, 0.0, 1000.0),
        (100.0, 0.0, 208.0),
        (120.0, 0.0, 1230.0),
        (10000.0, 0.0, 111230.0),
        (10000.0, 0.0, 98760.0),
    ]
    assert w.data("Side").get().getRealValue("x_1") == 117.0


def test_an_observable_the_workspace_lacks_is_made_up_to_ten_times_its_value() -> None:
    """``SafeObservableCreation``: ``[0, 10 v]``, or ``[0, maximum]`` - widened if short."""
    w = ROOT.RooWorkspace()
    made = NumberCountingPdfFactory.SafeObservableCreation(w, "fresh", 3.0)
    assert (made.getVal(), made.getMin(), made.getMax()) == (3.0, 0.0, 30.0)
    made = NumberCountingPdfFactory.SafeObservableCreation(w, "fresh", 3.0, 7.0)
    assert (made.getVal(), made.getMin(), made.getMax()) == (3.0, 0.0, 7.0)
    made = NumberCountingPdfFactory.SafeObservableCreation(w, "fresh", 3.0, 2.0)
    assert (made.getVal(), made.getMin(), made.getMax()) == (3.0, 0.0, 30.0)
    assert w.var("fresh") is None


def test_the_z_bi_p_values_and_significances_are_rs_numbercountingutils() -> None:
    """ROOT 6.40: p 0.0009416504675382 and Z 3.1080438957471 for 50 over 100 with 10%, all
    four ways."""
    for p_value, z in (
        (U.BinomialExpP(50, 100, 0.1), U.BinomialExpZ(50, 100, 0.1)),
        (U.BinomialObsP(150, 100, 0.1), U.BinomialObsZ(150, 100, 0.1)),
        (U.BinomialWithTauExpP(50, 100, 1), U.BinomialWithTauExpZ(50, 100, 1)),
        (U.BinomialWithTauObsP(150, 100, 1), U.BinomialWithTauObsZ(150, 100, 1)),
    ):
        assert p_value == pytest.approx(0.0009416504675382001, rel=1e-12)
        assert z == pytest.approx(3.108043895747122, rel=1e-12)
