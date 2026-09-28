"""A verbose fit, ``V``: Minuit2's own report, then ``FitResult::Print``'s block.

The reference is what ROOT 6.40.04 printed for the tutorial
``math/fit/FitHistoInFile.C``: a quadratic background and a normalised
Gaussian fitted with ``"VR+"``.
"""

from __future__ import annotations

import math
import re

import pytest

from refmachine import roots
from xrdroot import pyroot as ROOT
from xrdroot.fit.hfit import max_calls

#: The histogram FitHistoInFile.C fits.
DATA = [
    6, 1, 10, 12, 6, 13, 23, 22, 15, 21, 23, 26, 36, 25, 27, 35, 40, 44, 66, 81,
    75, 57, 43, 37, 36, 31, 35, 36, 43, 32, 40, 37, 38, 33, 36, 44, 42, 37, 32, 32,
    43, 44, 35, 33, 33, 39, 29, 41, 32, 44, 26, 39, 29, 35, 32, 21, 21, 15, 25, 15,
]  # fmt: skip

#: What ROOT printed, from ``Minuit2Minimizer: Minimize`` to the block's last parameter.
ROOTS_TEXT = """\
Minuit2Minimizer: Minimize with max-calls 1780 convergence for edm < 0.01 strategy 1
Minuit2Minimizer : Valid minimum - status = 0
FVAL  = 20.0330977485550434
Edm   = 1.41976030358321436e-07
Nfcn  = 251
Bkg0\t  = -1.72294\t +/-  3.20827
Bkg1\t  = 55.7228\t +/-  8.6473
Bkg2\t  = -19.4083\t +/-  4.14403
Gauss scale\t  = 8.41522\t +/-  1.33872
Gauss #sigma\t  = 0.0724347\t +/-  0.0116855
Gauss #mu\t  = 0.984485\t +/-  0.0109757

Covariance Matrix:

            \t        Bkg0        Bkg1        Bkg2 Gauss scaleGauss #sigma   Gauss #mu
Bkg0        \t      10.293     -25.273      11.233      1.4782   0.0095082   0.0011726
Bkg1        \t     -25.273      74.776     -35.237      -5.827   -0.037636  0.00042072
Bkg2        \t      11.233     -35.237      17.173      2.8018    0.018113 -0.00087613
Gauss scale \t      1.4782      -5.827      2.8018      1.7922    0.008954 -0.00050823
Gauss #sigma\t   0.0095082   -0.037636    0.018113    0.008954  0.00013655 -1.1754e-05
Gauss #mu   \t   0.0011726  0.00042072 -0.00087613 -0.00050823 -1.1754e-05  0.00012047

Correlation Matrix:

            \t        Bkg0        Bkg1        Bkg2 Gauss scaleGauss #sigma   Gauss #mu
Bkg0        \t           1    -0.91099     0.84489     0.34417     0.25362    0.033301
Bkg1        \t    -0.91099           1    -0.98333    -0.50335    -0.37246   0.0044329
Bkg2        \t     0.84489    -0.98333           1     0.50504     0.37404   -0.019263
Gauss scale \t     0.34417    -0.50335     0.50504           1     0.57237   -0.034589
Gauss #sigma\t     0.25362    -0.37246     0.37404     0.57237           1   -0.091642
Gauss #mu   \t    0.033301   0.0044329   -0.019263   -0.034589   -0.091642           1
****************************************
Minimizer is Minuit2 / Migrad
Chi2                      =      20.0331
NDf                       =           30
Edm                       =  1.41976e-07
NCalls                    =          251
Bkg0                      =     -1.72294   +/-   3.20827
Bkg1                      =      55.7228   +/-   8.6473
Bkg2                      =     -19.4083   +/-   4.14403
Gauss scale               =      8.41522   +/-   1.33872
Gauss #sigma              =    0.0724347   +/-   0.0116855
Gauss #mu                 =     0.984485   +/-   0.0109757
"""

#: A number as the report prints one.
NUMBER = re.compile(r"-?\d+(?:\.\d*)?(?:e[-+]\d+)?")


def fit_function(x: list[float], par: list[float]) -> float:
    background = par[0] + par[1] * x[0] + par[2] * x[0] * x[0]
    return background + par[3] * ROOT.TMath.Gaus(x[0], par[5], par[4], True)


def fitted_verbosely() -> None:
    """FitHistoInFile.C's fit, which prints as it goes."""
    histo = ROOT.TH1D("histo", "Gauss Peak on Quadratic Background;x;Events/0.05", 60, 0, 3)
    for i, value in enumerate(DATA):
        histo.SetBinContent(i + 1, value)
    fit = ROOT.TF1("fitFcn", fit_function, 0.2, 2.7, 6)
    names = ["Bkg0", "Bkg1", "Bkg2", "Gauss scale", "Gauss #sigma", "Gauss #mu"]
    for index, name in enumerate(names):
        fit.SetParName(index, name)
    fit.SetParameters(30, 0, 0, 50.0, 0.1, 1.0)
    histo.GetXaxis().SetRange(2, 40)
    histo.Fit("fitFcn", "VR+", "ep")


def numbers_of(line: str) -> list[float]:
    return [float(text) for text in NUMBER.findall(line)]


def wanted(line: str) -> object:
    """ROOT's numbers on a line, held as close as they can be.

    Edm, to eighteen figures, is MIGRAD's last estimate: the step before it
    rounded otherwise moves it by 7e-8 even on ROOT's machine. Off that
    machine the fit moves by FIT_REL, which can turn the sixth figure a value
    is printed to: one in its last place, 1e-5 of it.
    """
    if line.startswith("Edm"):
        return pytest.approx(numbers_of(line), rel=1e-6)
    return roots(numbers_of(line), rel=1e-5)


def test_a_verbose_fit_prints_minuit2s_report_then_the_results_block(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fitted_verbosely()
    # The block's lines end in blanks, as ROOT's do; the reference is kept without them.
    printed = [line.rstrip(" ") for line in capsys.readouterr().out.splitlines()]
    expected = ROOTS_TEXT.splitlines()
    assert [NUMBER.sub("#", line) for line in printed] == [
        NUMBER.sub("#", line) for line in expected
    ]
    assert [numbers_of(line) for line in printed] == [wanted(line) for line in expected]


def test_the_call_limit_is_fitters_and_an_invalid_minimum_reports_no_parameters() -> None:
    """``1000 + 100 n + 5 n^2`` calls; an invalid minimum says its values to six figures."""
    from xrdroot.fit import FitResult

    assert max_calls(6) == 1780
    one = {"parameters": [1.0], "errors": [0.5], "covariance": [[0.25]], "names": ["a"]}
    bad = FitResult(**one, fcn=math.pi, edm=2.0, valid=False, status=3)
    assert bad.minuit2_report(max_calls(1), 0.01, 1).splitlines()[1:] == [
        "Minuit2Minimizer : Invalid minimum - status = 3",
        "FVAL  = 3.14159",
        "Edm   = 2",
        "Nfcn  = 0",
    ]
    fixed = FitResult(
        parameters=[1.0, 2.0],
        errors=[0.5, 0.0],
        covariance=[[0.25, 0], [0, 0]],
        names=["a", "b"],
        fcn=1.0,
        fixed=[False, True],
    )
    assert "b\t  = 2\t (fixed)" in fixed.minuit2_report(1220, 0.01, 1).splitlines()
    limited = FitResult(**one, fcn=1.0, bounded=[True])
    assert "a\t  = 1\t +/-  0.5\t(limited)" in limited.minuit2_report(1105, 0.01, 1)
