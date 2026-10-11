"""``ROOT::Fit``'s data, chi-square and ``FitFCN``: a combined fit of two histograms, as ROOT's.

combinedFit fits an exponential to one histogram and the same exponential
and a Gaussian to another, sharing the slope: the expected numbers are what
ROOT 6.40 prints for it, at the six figures it prints them - which another
machine's ``exp`` cannot move - and its 225 calls on ROOT's machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from refmachine import ROOTS_MACHINE

#: Which of the six shared parameters each histogram's function takes.
B, SB = [0, 2], [1, 2, 3, 4, 5]


def _histograms() -> tuple[Any, Any, Any, Any]:
    ROOT.gRandom.SetSeed(4357)  # a fresh TRandom3, as the tutorial has it
    hB = ROOT.TH1D("hB", "histo B", 100, 0, 100)
    hSB = ROOT.TH1D("hSB", "histo S+B", 100, 0, 100)
    fB = ROOT.TF1("fB", "expo", 0, 100, ROOT.TF1.EAddToList.kNo)
    fB.SetParameters(1, -0.05)
    hB.FillRandom(fB)
    fS = ROOT.TF1("fS", "gaus", 0, 100, ROOT.TF1.EAddToList.kNo)
    fS.SetParameters(1, 30, 5)
    hSB.FillRandom(fB, 2000)
    hSB.FillRandom(fS, 1000)
    return hB, hSB, fB, ROOT.TF1("fSB", "expo + gaus(2)", 0, 100, ROOT.TF1.EAddToList.kNo)


def _data(histogram: Any, low: float, high: float) -> Any:
    window = ROOT.Fit.DataRange()
    window.SetRange(low, high)
    data = ROOT.Fit.BinData(ROOT.Fit.DataOptions(), window)
    ROOT.Fit.FillData(data, histogram)
    return data, window


def test_a_combined_chi_square_fit_is_roots(capsys: pytest.CaptureFixture[str]) -> None:
    hB, hSB, fB, fSB = _histograms()
    (dataB, rangeB), (dataSB, _) = _data(hB, 10, 90), _data(hSB, 10, 50)
    chi2B = ROOT.Fit.Chi2Function(dataB, ROOT.Math.WrappedMultiTF1(fB, 1))
    chi2SB = ROOT.Fit.Chi2Function(dataSB, ROOT.Math.WrappedMultiTF1(fSB, 1))
    assert (dataB.Size(), dataSB.Size(), chi2B.NDim(), rangeB().first, rangeB().second) == (
        80, 40, 2, 10.0, 90.0)  # fmt: skip
    fitter = ROOT.Fit.Fitter()
    fitter.Config().SetParamsSettings(6, np.array([5, 5, -0.1, 100, 30, 10]))
    fitter.Config().ParSettings(4).Fix()
    fitter.Config().ParSettings(2).SetLimits(-10, -1.0e-4)
    fitter.Config().ParSettings(3).SetLimits(0, 10000)
    fitter.Config().ParSettings(3).SetStepSize(5)
    fitter.Config().MinimizerOptions().SetPrintLevel(0)
    assert fitter.Config().MinimizerOptions().PrintLevel() == 0
    fitter.Config().SetMinimizer("Minuit2", "Migrad")

    def both(p: Any) -> float:
        return float(chi2B([p[i] for i in B]) + chi2SB([p[i] for i in SB]))

    assert fitter.FitFCN(6, both, None, dataB.Size() + dataSB.Size(), True)
    result = fitter.Result()
    assert f"{result.Chi2():g}" == "131.104" and result.Ndf() == 115
    assert [f"{result.Parameter(i):g}" for i in range(6)] == [
        "5.5396", "4.66089", "-0.0514037", "77.2733", "30", "4.864"]  # fmt: skip
    assert result.NCalls() == (225 if ROOTS_MACHINE else result.NCalls())
    result.Print(ROOT.std.cout)
    assert "Chi2                      =      131.104" in capsys.readouterr().out
    _check_set(fSB, result)


def _check_set(fSB: Any, result: Any) -> None:
    """``SetFitResult`` with the indices of the shared parameters ``fSB`` takes."""
    fSB.SetFitResult(result, SB)
    assert (fSB.GetChisquare(), fSB.GetNDF(), fSB.GetNumberFitPoints()) == (
        result.Chi2(), 115, 120)  # fmt: skip
    assert fSB.GetParameter(4) == result.Parameter(5) and fSB.GetParError(3) == 0.0


def test_fitfcn_of_a_function_and_its_dimension_minimises_it_and_says_no_chi_square(
    capsys: pytest.CaptureFixture[str],
) -> None:
    square = ROOT.Math.Functor(lambda p: (p[0] - 1.0) ** 2 + 4.0, 1)
    fitter = ROOT.Fit.Fitter()
    assert fitter.FitFCN(square, np.array([3.0]))
    assert fitter.Result().Parameter(0) == pytest.approx(1.0, abs=1e-6)
    assert fitter.Result().Ndf() == 0  # no data size given: none, not one less
    fitter.Result().Print()
    shown = capsys.readouterr().out
    assert "MinFCN" in shown and "Chi2" not in shown
    line = ROOT.TF1("line", "[0]", 0, 1)
    line.SetFitResult(fitter.Result(), [-1])
    assert line.GetChisquare() == pytest.approx(4.0) and line.GetParameter(0) == 0.0
    with pytest.raises(AttributeError, match="MinimizerOptions has Tolerance"):
        fitter.Config().MinimizerOptions().Tolerance()
    wrapped = ROOT.Math.WrappedMultiTF1(ROOT.TF1("w", "[0]*x", 0, 1))
    assert (wrapped.NDim(), wrapped.NPar(), wrapped([2.0], [3.0])) == (1, 1, 6.0)
    assert ROOT.Fit.DataRange(1, 2).Size() == 1 and ROOT.Fit.BinData().Size() == 0
    assert ROOT.Fit.DataRange()() == (-np.inf, np.inf) and ROOT.Fit.BinData().NDim() == 0


def test_points_added_one_by_one_are_fitted_by_the_model_as_root_fits_them() -> None:
    """exampleFit3D: a thousand points of three coordinates, ROOT 6.40's parameters."""
    generator = ROOT.TRandom2()
    data = ROOT.Fit.BinData(1000, 3)
    for _ in range(1000):
        x, y, z = (generator.Uniform(0, 10) for _ in range(3))
        data.Add([x, y, z], np.sin(x) + np.cos(y) + z + generator.Gaus(0, 0.1), 0.1)
    f3 = ROOT.TF3("f3", "[0] * sin(x) + [1] * cos(y) + [2] * z", 0, 10, 0, 10, 0, 10)
    f3.SetParameters(2, 2, 2)
    fitter = ROOT.Fit.Fitter()
    assert fitter.SetFunction(ROOT.Math.WrappedMultiTF1(f3, 3))
    assert fitter.Fit(data) and (data.Size(), data.NDim()) == (1000, 3)
    result = fitter.Result()
    shown = (f"{result.Chi2():g}", result.Ndf(), f"{result.Prob():g}")
    assert shown == ("993.065", 997, "0.529221")
    assert [result.ParName(i) for i in range(3)] == ["p0", "p1", "p2"]
    assert [f"{result.Parameter(i):g}" for i in range(3)] == ["0.993546", "0.99397", "0.997895"]
    line = ROOT.Fit.BinData(2, 1)
    line.Add(1.0, 2.0)
    assert (line.Size(), line.NDim()) == (1, 1)
