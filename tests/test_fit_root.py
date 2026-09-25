"""Fits and fills checked against what ROOT itself wrote and documented.

``tgme.root``'s multigraph was fitted by ROOT 6.24 with ``mg->Fit("pol1",
"FQ")``: Minuit, on the effective-variance chi-square of asymmetric errors
with x errors, from ``gROOT``'s ``pol1`` at its standard parameters of one.
Refitted here the same way, the parameters must be ROOT's to a part in
10^9 and the chi-square to the last digits, and the chi-square at ROOT's
parameters must be ROOT's exactly. ``dirs-6.14.00.root`` and
``embedded-tbox.root`` hold histograms ROOT filled with
``h->FillRandom("gaus", 5)`` straight after starting, so from ``gRandom`` at
its default seed: filled here the same way they must be the same doubles.
The rest are the numbers ROOT's own tests and tutorials pin: ``stress.cxx``'s
integral of a fitted triple Gaussian, the Minuit example of ``Ifit.C`` that
PyROOT's tests check, and PyROOT's Gaussian fit of a callable.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest

from xrdroot import Function, Histogram, TRandom3, create, open_root
from xrdroot.fit import cost, minimize
from xrdroot.fit.data import DataOptions, from_graphs

DATA = pathlib.Path(__file__).parent / "data"


def opened(name):
    return open_root(str(DATA / name))


def rooted_pol1():
    """The fit as ROOT stored it on the multigraph, and the multigraph."""
    with opened("tgme.root") as root:
        mg = root["mg"]
    return mg, mg.functions[0]


# -- the fit ROOT made -----------------------------------------------------------------------


def test_the_chi_square_at_roots_parameters_is_the_chi_square_root_stored():
    mg, theirs = rooted_pol1()
    data = from_graphs(list(mg), DataOptions(), None)
    # The TGraph without errors gives nothing to data with asymmetric errors.
    assert (data.size, data.kind) == (10, 3)
    fcn = cost.effective_chi2(data, Function("pol1", "pol1"))
    assert fcn(theirs.parameters) == theirs.fit_result["chi2"]


def test_refitting_roots_multigraph_gives_roots_parameters_errors_and_chi_square(capsys):
    mg, theirs = rooted_pol1()
    ours, rest = theirs.parameters.copy(), theirs.parameter_errors.copy()
    stored = theirs.fit_result
    result = mg.fit("pol1", "FQ")
    assert capsys.readouterr().out == ""
    np.testing.assert_allclose(result.parameters, ours, rtol=1e-9)
    np.testing.assert_allclose(result.errors, rest, rtol=1e-5)
    assert result.chi2 == pytest.approx(stored["chi2"], rel=1e-14)
    assert (result.ndf, result.npoints, result.valid, result.status) == (8, 10, True, 0)
    refit = mg.functions[0]
    assert len(mg.functions) == 1 and refit is result.function and refit is not theirs
    assert refit.fit_result["ndf"] == stored["ndf"] and refit.fit_result["npfits"] == 10
    assert refit.range == theirs.range == (-0.53, 4.53)
    np.testing.assert_allclose(refit.members["fSave"], theirs.members["fSave"], rtol=1e-9)


def test_without_f_the_same_fit_goes_to_minuit_anyway_for_asymmetric_errors(capsys):
    mg, theirs = rooted_pol1()
    result = mg.fit("pol1", "Q")
    assert result.minimizer == "Minuit2 / Migrad"
    np.testing.assert_allclose(result.parameters, theirs.parameters, rtol=1e-9)


def test_a_fitted_histogram_writes_and_reads_back_with_its_fit_as_root_records_it():
    h = Histogram.book("h", (40, -4, 4))
    h.fill_random("gaus", 2000, rng=TRandom3(7))
    result = h.fit("gaus", "Q")
    buffer = io.BytesIO()
    with create(buffer) as out:
        out["h"] = h
    with open_root(io.BytesIO(buffer.getvalue())) as back:
        (again,) = back["h"].functions
    np.testing.assert_array_equal(again.parameters, result.parameters)
    np.testing.assert_array_equal(again.parameter_errors, result.errors)
    assert again.fit_result["chi2"] == result.chi2 and again.fit_result["ndf"] == result.ndf
    # GetBinLowEdge(last) + GetBinWidth(last), as ROOT adds them up: a hair past 4.
    assert again.range == (-4.0, -4.0 + 39 * 0.2 + 0.2)
    low, high = again.parameter_limits[2]  # InitGaus's bound on the width stays on the fit
    assert low == 0.0 and high == pytest.approx(10 * h.std(), rel=0.01)
    np.testing.assert_allclose(again(np.linspace(-4, 4, 9)), result.function(np.linspace(-4, 4, 9)))


# -- histograms ROOT filled ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "key", "axis"),
    [
        ("dirs-6.14.00.root", "dir1/dir11/h1", (100, 0.0, 100.0)),
        ("embedded-tbox.root", "h1", (10, 0.0, 10.0)),
    ],
)
def test_fill_random_from_gaus_at_roots_seed_is_roots_histogram(name, key, axis):
    with opened(name) as root:
        theirs = root[key]
    mine = Histogram.book("h1", axis, kind="F")
    mine.fill_random("gaus", 5, rng=TRandom3())
    np.testing.assert_array_equal(mine.values(flow=True), theirs.values(flow=True))
    for member in ("fEntries", "fTsumw", "fTsumw2", "fTsumwx", "fTsumwx2"):
        assert mine._core[member] == theirs._core[member], member


def test_stress_cxx_integral_of_a_fitted_triple_gaussian_is_within_roots_tolerance():
    # stress.cxx: FillRandom 10000 from gaus(0)+gaus(3)+gaus(6) at seed 65539,
    # fit "q0", and the integral over [-8, 6] "must be = 1923.74578" within 10.
    params = [100, -3, 3, 60, 0, 0.5, 40, 4, 0.7]
    f = Function("f1form", "gaus(0)+gaus(3)+gaus(6)", range=(-10, 10), parameters=params)
    h = Histogram.book("h1form", (100, -10, 10), kind="F")
    h.fill_random(f, 10000, rng=TRandom3(65539))
    result = h.fit(f, "q0")
    assert result.valid and result.ndf == 100 - 9 - 14  # the empty bins are left out
    assert abs(f.integral(-8, 6) - 1923.74578) < 1.0


def test_pyroots_fit_of_a_python_gaussian_finds_its_mean_and_width():
    # PyROOT_functiontests.py: FillRandom("gaus", 200000) into (100, -4, 4),
    # fitted by a four-parameter Python function with "0Q": 96 degrees of
    # freedom, the mean 0 and the width 1 to one decimal place.
    def pygaus(x, par):
        arg = (x - par[1]) / par[2]
        return par[0] / (1 + par[3]) * (0.01 * 0.39894228) / par[2] * np.exp(-0.5 * arg * arg)

    h = Histogram.book("h", (100, -4, 4), kind="F")
    h.fill_random("gaus", 200000, rng=TRandom3())
    result = h.fit(pygaus, "0Q", npar=4, parameters=[600, 0.43, 0.35, 600])
    assert result.function.fit_result["ndf"] == 96
    assert round(result.parameters[1], 1) == 0 and round(result.parameters[2] - 1.0, 1) == 0


def test_the_minuit_example_pyroots_tests_check_gives_its_numbers():
    # Ifit.C, as PyROOT_functiontests.py runs it through TMinuit: MIGRAD
    # from (3, 1, 0.1, 0.01) with steps (0.1, 0.1, 0.01, 0.001).
    z = np.array([1.0, 0.96, 0.89, 0.85, 0.78], dtype=np.float32).astype(float)
    x = np.array([1.5751, 1.5825, 1.6069, 1.6339, 1.6706], dtype=np.float32).astype(float)
    y = np.array([1.0642, 0.97685, 1.13168, 1.128654, 1.44016], dtype=np.float32).astype(float)
    error = float(np.float32(0.01))

    def fcn(par):
        value = ((par[0] * par[0]) / (x * x) - 1) / (par[1] + par[2] * y - par[3] * y * y)
        return float(np.sum(((z - value) / error) ** 2))

    # "MIGRAD 500 1." is a tolerance of one, and no HESSE: MIGRAD's own errors.
    result = minimize(fcn, [3, 1, 0.1, 0.01], errors=[0.1, 0.1, 0.01, 0.001], tolerance=1.0)
    assert result.valid
    expected = [(2.15, 0.10), (0.81, 0.25), (0.17, 0.40), (0.10, 0.16)]
    for (value, error_), found, err in zip(expected, result.parameters, result.errors):
        assert round(found - value, 2) == 0 and round(err - error_, 2) == 0
