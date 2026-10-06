"""``RooBinIntegrator``: a binned distribution integrated numerically over a closed range is
summed bin by bin, whatever integrator is configured - against ROOT 6.40.

The histograms' densities, forced to numeric integration, over their whole
range, a range that cuts bins, a range inside one bin, two dimensions, and a
``RooRealSumPdf`` of a ``RooHistFunc`` - the values and the integrator ROOT
names - with what makes a distribution binned and where its bins are.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot.roofit import integration
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.histpdf import RooHistFunc, RooHistPdf
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.variables import RooRealVar

BIN = "using numeric integrator RooBinIntegrator to calculate Int"


def _hist(name: str, variables: list[Any], rows: list[tuple[float, ...]]) -> RooDataHist:
    """A histogram of weighted events: each row's values, then its weight."""
    w = RooRealVar("w", "w", 1)
    data = RooDataSet(f"s{name}", "s", RooArgSet(*variables, w), RooCmdArg("WeightVar", "w"))
    for *values, weight in rows:
        for var, value in zip(variables, values, strict=False):
            var.setVal(value)
        data.add(RooArgSet(*variables), weight)
    return RooDataHist(name, name, RooArgSet(*variables), data)


def _axes() -> tuple[Any, Any]:
    x = RooRealVar("x", "x", 0, 4)
    x.setBins(4)
    y = RooRealVar("y", "y", 0, 2)
    y.setBins(2)
    return x, y


def _density(x: Any) -> tuple[Any, Any]:
    dh = _hist("dh", [x], [(0.5, 1.0), (1.5, 4.0), (2.5, 2.0), (3.5, 3.0)])
    return RooHistPdf("hp", "hp", RooArgSet(x), dh), dh


def test_a_histogram_forced_to_numeric_integration_is_summed_bin_by_bin(capsys: Any) -> None:
    """Over the whole range, a range that cuts bins, and a range inside one bin: ROOT's."""
    x, _ = _axes()
    hp, _ = _density(x)
    xs = RooArgSet(x)
    x.setVal(1.2)
    hp.setForceNumInt(True)
    capsys.readouterr()
    assert hp.getVal(xs) == 0.4
    x.setRange("r", 0.7, 3.2)
    assert hp.createIntegral(xs, Range="r").getVal() == 6.9
    x.setRange("n", 1.2, 1.7)
    assert hp.createIntegral(xs, Range="n").getVal() == 2.0
    out = capsys.readouterr().out
    for label in ("hp_Int[x]", "hp_Int[x|r]", "hp_Int[x|n]"):
        assert f"RooRealIntegral::init({label}) {BIN}(x)\n" in out


def test_a_two_dimensional_histogram_and_a_sum_of_one_are_summed_bin_by_bin(capsys: Any) -> None:
    """Both variables' bins, the last innermost; a sum's bins are its terms' bins."""
    x, y = _axes()
    rows = [(0.5, 0.5, 1.0), (1.5, 0.5, 2.0), (2.5, 1.5, 5.0), (3.5, 1.5, 1.0), (0.5, 1.5, 2.0)]
    hp2 = RooHistPdf("hp2", "hp2", RooArgSet(x, y), _hist("dh2", [x, y], rows))
    hp2.setForceNumInt(True)
    x.setVal(0.5)
    y.setVal(1.2)
    assert hp2.getVal(RooArgSet(x, y)) == 0.18181818181818182
    _, dh = _density(x)
    x.setVal(0.5)
    rs = RooRealSumPdf("rs", "rs", RooArgList(RooHistFunc("f1", "f1", RooArgSet(x), dh)),
                       RooArgList(RooRealVar("c1", "c1", 2.0)))  # fmt: skip
    rs.setForceNumInt(True)
    capsys.readouterr()
    assert rs.getVal(RooArgSet(x)) == 0.1
    assert f"RooRealIntegral::init(rs_Int[x]) {BIN}(x)\n" in capsys.readouterr().out
    assert rs.bin_boundaries("x") == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert rs.bin_boundaries("y") is None


def test_what_is_binned_and_where_its_bins_are() -> None:
    """Order zero is binned, interpolation is not; a product is binned where its factors are;
    a histogram's bin of a value, and its boundaries - none for another variable or when
    interpolated."""
    x, y = _axes()
    hp, dh = _density(x)
    hpy = RooHistPdf("hpy", "hpy", RooArgSet(y), _hist("hy", [y], [(0.5, 1.0), (1.5, 3.0)]))
    prod = RooProdPdf("prod", "prod", RooArgList(hp, hpy))
    assert hp.isBinnedDistribution(RooArgSet(x)) and prod.isBinnedDistribution(RooArgSet(x))
    smooth = RooHistPdf("hp3", "hp3", RooArgSet(x), dh, 2)
    assert not smooth.isBinnedDistribution(RooArgSet(x))
    assert smooth.bin_boundaries("x") is None
    assert hp.bin_boundaries("y") is None
    assert hp.bin_index({"x": 1.5}) == 1
    assert hp.bin_index({"x": np.array([0.5, 3.5])}).tolist() == [0, 3]


def test_an_integrand_with_no_bins_is_given_a_hundred(capsys: Any) -> None:
    """``RooBinIntegrator``'s fallback for an integrand that says nothing of its bins."""
    x, _ = _axes()
    g = RooGaussian("g", "g", x, RooRealVar("m", "m", 1), RooRealVar("s", "s", 1, 0.5, 2))
    capsys.readouterr()
    edges = integration._edges(g, "x", 0.0, 4.0)
    assert len(edges) == 101 and edges[0] == 0.0 and edges[-1] == 4.0
    assert capsys.readouterr().out == (
        "[#0] WARNING:Integration -- RooBinIntegrator::RooBinIntegrator WARNING: integrand "
        "provide no binning definition observable #0 substituting default binning of 100 bins\n"
    )
