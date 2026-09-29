"""``RooAbsData::moment``, ``mean``, ``sigma``, ``meanVar``, ``rmsVar`` and ``statOn`` of a
weighted dataset, against the numbers ROOT 6.40 prints for the same five events: with and
without an offset, a cut and a range - whose events RooFit leaves out of the sum but not the
count - and the errors for an unknown variable and an empty selection."""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.variables import RooRealVar

EVENTS = ((1.0, 2.0, 1.0), (2.5, -1.0, 2.0), (-3.0, 0.5, 0.5), (4.0, 3.0, 1.5), (0.25, -2.0, 1.0))


def _data() -> tuple[Any, Any]:
    """Five weighted events in ``x`` and ``y``, and ``x`` with a range ``mid`` of ``[-1, 3]``."""
    x = RooRealVar("x", "x", 0, -10, 10)
    y = RooRealVar("y", "y", 0, -10, 10)
    w = RooRealVar("w", "w", 1, 0, 10)
    data = RooDataSet("d", "d", RooArgSet(x, y, w), RooCmdArg("WeightVar", "w"))
    for vx, vy, vw in EVENTS:
        x.setVal(vx)
        y.setVal(vy)
        data.add(RooArgSet(x, y), vw)
    x.setRange("mid", -1, 3)
    return data, x


def test_the_mean_sigma_and_moments_are_roots() -> None:
    """About the mean for an order above one, about an offset when one is given."""
    data, x = _data()
    assert (data.mean(x), data.sigma(x)) == (1.7916666666666667, 1.9494479275482641)
    assert data.moment(x, 3) == -7.050636574074076
    assert data.moment(x, 2, 0.5) == 5.46875


def test_a_cut_selects_the_events_summed_and_counted() -> None:
    data, x = _data()
    assert (data.mean(x, "y>0"), data.sigma(x, "y>0")) == (1.8333333333333333, 2.5440562537456244)


def test_a_range_leaves_its_events_out_of_the_sum_but_not_the_count() -> None:
    """RooFit's ``allInRange`` test turned round: the events inside ``mid`` are skipped."""
    data, x = _data()
    assert data.mean(x, "x>-100", "mid") == 1.125
    assert data.moment(x, 2, "x>-100", "mid") == 5.2265625
    assert data.moment(x, 2, 0.0, "x>-100", "mid") == 7.125


def test_an_unknown_variable_or_an_empty_selection_is_an_error_and_zero(capsys: Any) -> None:
    data, x = _data()
    capsys.readouterr()
    assert data.mean(RooRealVar("z", "z", 0, 0, 1)) == 0.0
    assert data.mean(x, "y>100") == 0.0
    out = capsys.readouterr()
    assert (out.out + out.err) == (
        "[#0] ERROR:InputArguments -- RooDataSet::moment(d) ERROR: unknown variable: z\n"
        "[#0] ERROR:InputArguments -- RooDataSet::moment(d) WARNING: empty dataset\n"
    )


def test_the_mean_and_rms_variables_carry_their_errors() -> None:
    """``meanVar`` and ``rmsVar``: named after the variable, with ROOT's values and errors."""
    data, x = _data()
    mean, rms = data.meanVar(x), data.rmsVar(x)
    assert (mean.GetName(), mean.getVal(), mean.getError()) == (
        "xMean", 1.7916666666666667, pytest.approx(0.8718196169188007, rel=1e-14))
    assert (rms.GetName(), rms.getVal(), rms.getError()) == (
        "xRMS", pytest.approx(2.135513209199762, rel=1e-14),
        pytest.approx(0.6164695630947421, rel=1e-14))  # fmt: skip
    frame = x.frame()
    assert data.statOn(frame) is frame
