"""``BayesianCalculator``: RooStats' credible intervals in one parameter, as ROOT 6.40 finds them.

The models are rs701's counting experiment - a signal and a background of
uniform shapes, three events seen - without nuisance parameters, and a
Poisson count whose background is a nuisance integrated over. Each
interval - central from the cumulative posterior, left- or right-sided,
from a scan, the shortest - and each plot's curves are the ones ROOT
printed for the same calls. The failures the calculator reports are
provoked for what it says, and its numerical failures by a root finder or
an integral made to fail.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.numerics.roots import brent_root
from xrdroot.roostats import bayesian
from xrdroot.roostats.bayesian import BayesianCalculator

CENTRAL = (0.8815733114447084, 7.252299555036391)
#: The 95% central interval from a scan of 100 bins.
SCAN100 = (0.6114153842025912, 8.263371722723951)


def counting(n: int = 3) -> tuple[Any, Any]:
    """rs701's model, its background fixed at 0.5 and not integrated over, ``n`` events seen."""
    w = ROOT.RooWorkspace("w")
    w.factory("SUM::pdf(s[0.001,15]*Uniform(x[0,1]),b[0.5,0,2]*Uniform(x))")
    w.factory("Uniform::priorPOI(s)")
    w.factory(f"n[{n}]")
    data = ROOT.RooDataSet("data", "", ROOT.RooArgSet(w["x"], w["n"]), WeightVar="n")
    data.add(ROOT.RooArgSet(w["x"]), w["n"].getVal())
    return w, data


def calculator(n: int = 3) -> BayesianCalculator:
    w, data = counting(n)
    found = BayesianCalculator(data, w["pdf"], ROOT.RooArgSet(w["s"]), w["priorPOI"])
    found.SetTestSize(0.1)
    return found


def nuisance() -> tuple[Any, Any]:
    """Three events of a Poisson count of signal and a background, flat in [0.5, 1.5]."""
    w = ROOT.RooWorkspace("w")
    w.factory("Poisson::pois(n[3], sum::nexp(s[0,6],b[1,0.5,1.5]))")
    w.factory("Uniform::prior_b(b)")
    w.factory("Uniform::prior_s(s)")
    w.factory("PROD::model(pois,prior_b)")
    data = ROOT.RooDataSet("data", "", ROOT.RooArgSet(w["n"]))
    data.add(ROOT.RooArgSet(w["n"]))
    return w, data


def with_nuisance() -> BayesianCalculator:
    w, data = nuisance()
    found = BayesianCalculator(data, w["model"], ROOT.RooArgSet(w["s"]), w["prior_s"],
                               ROOT.RooArgSet(w["b"]))  # fmt: skip
    found.SetTestSize(0.05)
    return found


def limits(interval: Any) -> tuple[float, float]:
    return interval.LowerLimit(), interval.UpperLimit()


def said(out: str, *parts: str) -> None:
    """Each of ``parts`` is in ``out``."""
    for part in parts:
        assert part in out


def test_the_central_interval_is_where_the_cumulative_posterior_crosses_the_tails(capfd) -> None:
    """Without nuisance parameters the posterior is RooFit's formula of the likelihood; the
    ends are GSL's Brent roots of its QAGS integral, ROOT's to the last few bits."""
    bc = calculator()
    interval = bc.GetInterval()
    assert limits(interval) == pytest.approx(CENTRAL, rel=1e-13)
    assert interval.GetName() == "BayesianInterval_a"
    assert interval.GetTitle() == "SimpleInterval from BayesianCalculator"
    assert interval.ConfidenceLevel() == pytest.approx(0.9)
    assert (bc.Size(), bc.ConfidenceLevel()) == pytest.approx((0.1, 0.9))
    said(capfd.readouterr().out, "GetPosteriorFunction :  nll value 1.76199 poi value = 7.5005",
         "minimum of NLL vs POI for POI =  2.4985 min NLL = -0.295836",
         "BayesianCalculator:GetInterval Compute the interval from the posterior cdf",
         "PosteriorCdfFunction - integral of posterior = 0.297017",
         "found a valid interval : [0.881573 , 7.2523 ]")  # fmt: skip


def test_asking_again_recomputes_the_interval_and_says_so(capfd) -> None:
    """A second ``GetInterval`` warns that it recomputes, and finds the same interval."""
    bc = calculator()
    first = limits(bc.GetInterval())
    capfd.readouterr()
    assert limits(bc.GetInterval()) == first
    assert "recomputing interval for the same CL and same model" in capfd.readouterr().out


def test_one_sided_intervals_run_to_the_end_of_the_range() -> None:
    """A left-side fraction of 0 is an upper limit from the range's low end, of 1 a lower
    limit to its high end; the mode is the highest bin of the 100-bin scan."""
    upper = calculator()
    upper.SetLeftSideTailFraction(0.0)
    assert limits(upper.GetInterval()) == pytest.approx((0.001, 6.181584660620197), rel=1e-13)
    assert upper.GetMode() == 2.475835
    lower = calculator()
    lower.SetLeftSideTailFraction(1.0)
    lower.SetConfidenceLevel(0.95)
    assert limits(lower.GetInterval()) == pytest.approx((CENTRAL[0], 15.0), rel=1e-13)


def test_a_scanned_posterior_is_roots_interpolated_tf1(capfd) -> None:
    """The scan is ROOT's cloned ``TF1``: the posterior at the bins' edges, joined by straight
    lines, and the interval ``TF1::GetQuantiles`` of it."""
    bc = calculator()
    bc.SetScanOfPosterior(20)
    assert limits(bc.GetInterval()) == pytest.approx((0.8514608037170621, 7.284440054763357),
                                                     rel=1e-13)  # fmt: skip
    assert "scan posterior function in nbins = 20" in capfd.readouterr().out
    assert bc.GetPosteriorFunction().GetName().endswith("_approx")
    hundred = calculator()
    hundred.SetScanOfPosterior(100)
    hundred.SetTestSize(0.05)
    assert limits(hundred.GetInterval()) == pytest.approx(SCAN100, rel=1e-13)


def test_a_finer_scan_resamples_the_coarser_one() -> None:
    """Asking for more bins than were scanned scans again - the scan before, which is what
    ROOT's posterior has become (where ROOT itself crashes, the old scan deleted); asking for
    fewer keeps the scan there is."""
    bc = calculator()
    bc.SetScanOfPosterior(10)
    assert limits(bc.GetInterval()) == pytest.approx((0.8323899891485302, 7.378298356159644),
                                                     rel=1e-13)  # fmt: skip
    bc.SetScanOfPosterior(20)
    finer = limits(bc.GetInterval())
    assert finer == pytest.approx((0.8323899891485302, 7.378298356159644), rel=1e-13)
    assert bc.GetPosteriorFunction().GetName().endswith("_approx_approx")
    bc.SetScanOfPosterior(5)
    assert limits(bc.GetInterval()) == finer
    assert bc.GetMode() == 2.625825


def test_the_shortest_interval_takes_the_highest_bins_as_root_reads_them(capfd) -> None:
    """The shortest interval gathers the scan's highest bins - each weighed, as ROOT's
    ``GetArray`` misreads it, by the bin below - until the probability is reached."""
    bc = calculator()
    bc.SetShortestInterval()
    assert limits(bc.GetInterval()) == (0.45097000000000004, 6.450570000000001)
    out = capfd.readouterr().out
    assert "computing shortest interval with CL = 0.9" in out
    assert "scan posterior function in nbins = 100" in out
    few = calculator()
    few.SetShortestInterval()
    few.SetScanOfPosterior(4)
    few.SetTestSize(0.05)
    assert limits(few.GetInterval()) == (0.001, 11.25025)


def test_a_shortest_interval_far_from_its_level_is_warned_of(capfd) -> None:
    """At 10% the bins overshoot the level by more than a tenth of it, and RooStats warns."""
    bc = calculator()
    bc.SetShortestInterval()
    bc.SetConfidenceLevel(0.1)
    assert limits(bc.GetInterval()) == (2.40084, 2.70082)
    assert ("actual interval CL = 0.0670973 differs more than 10% from desired CL value - must "
            "increase nbins 100 to an higher value") in capfd.readouterr().out  # fmt: skip


def test_too_few_bins_for_a_shortest_interval_leave_it_empty(capfd) -> None:
    """When the first bin taken already passes the level there is no interval: ROOT keeps its
    ``[0, 0]`` and says the bins are not sufficient."""
    bc = calculator(6)
    bc.SetShortestInterval()
    bc.SetScanOfPosterior(4)
    bc.SetConfidenceLevel(0.01)
    assert limits(bc.GetInterval()) == (0.0, 0.0)
    assert "ComputeShortestInterval 4 bins are not sufficient" in capfd.readouterr().out


def test_nuisance_parameters_are_integrated_by_qags_at_each_scanned_point(capfd) -> None:
    """With a nuisance parameter the posterior is ``PosteriorFunction``: the likelihood times
    the prior integrated over the background at each value of the signal."""
    bc = with_nuisance()
    bc.SetScanOfPosterior(10)
    assert limits(bc.GetInterval()) == pytest.approx((0.276314564847608, 5.624255838055471),
                                                     rel=1e-13)  # fmt: skip
    assert bc.GetPosteriorFunction().GetName() == "posteriorfunction_from_nll_model_data_approx"
    assert ("RooRealBinding: The function prior_s does not depend on the parameter b. Note that "
            "passing copies of the parameters is not supported.") in capfd.readouterr().out
    short = with_nuisance()
    short.SetShortestInterval()
    short.SetScanOfPosterior(10)
    assert limits(short.GetInterval()) == (0.0, 5.3999999999999995)
    assert short.GetMode() == 2.1


def test_roofit_can_integrate_the_nuisance_parameters_instead() -> None:
    """``SetIntegrationType("ROOFIT")`` integrates the formula over the background by RooFit."""
    bc = with_nuisance()
    bc.SetIntegrationType("roofit")
    bc.SetScanOfPosterior(10)
    assert limits(bc.GetInterval()) == pytest.approx((0.27631456484896594, 5.62425583805563),
                                                     rel=1e-12)  # fmt: skip


def test_other_integrations_of_the_nuisance_parameters_are_refused() -> None:
    """VEGAS, MISER and the toy integrations are ROOT's; here they are refused by name."""
    bc = with_nuisance()
    bc.SetIntegrationType("VEGAS")
    with pytest.raises(UnsupportedFeatureError, match="its VEGAS integration is not here yet"):
        bc.GetPosteriorFunction()


def test_a_model_config_gives_the_calculator_its_model() -> None:
    """From a ``ModelConfig`` the calculator takes the density, the prior, the parameter of
    interest and the nuisance parameters - and a new dataset or prior drops what it had."""
    from xrdroot.roostats.modelconfig import ModelConfig

    w, data = nuisance()
    config = ModelConfig("config", w)
    config.SetPdf(w["model"])
    config.SetPriorPdf(w["prior_s"])
    config.SetParametersOfInterest(ROOT.RooArgSet(w["s"]))
    config.SetNuisanceParameters(ROOT.RooArgSet(w["b"]))
    bc = BayesianCalculator(data, config)
    bc.SetTestSize(0.05)
    bc.SetScanOfPosterior(10)
    bc.SetBrfPrecision(1e-6)
    bc.SetNumIters(0)
    bc.ForceNuisancePdf(w["prior_b"])
    expected = (0.276314564847608, 5.624255838055471)
    assert limits(bc.GetInterval()) == pytest.approx(expected, rel=1e-13)
    bc.SetData(data)
    bc.SetPriorPdf(w["prior_s"])
    assert limits(bc.GetInterval()) == pytest.approx(expected, rel=1e-13)


def test_a_constant_nuisance_parameter_is_not_integrated_over() -> None:
    """Constant nuisance parameters are dropped, and with none left RooFit's formula is it."""
    w, data = nuisance()
    w["b"].setConstant(True)
    bc = BayesianCalculator(data, w["model"], ROOT.RooArgSet(w["s"]), w["prior_s"],
                            ROOT.RooArgSet(w["b"]))  # fmt: skip
    assert bc.GetPosteriorFunction().GetName() == "likelihood_times_prior_product_model_prior_s"
    alone = BayesianCalculator(data, w["model"], ROOT.RooArgSet(w["s"]))
    assert alone.GetPosteriorFunction().GetName() == "likelihood_times_prior_model"


def test_a_calculator_without_a_model_says_what_it_misses(capfd) -> None:
    """No density, no parameter of interest, or two: no posterior, and each said; with no
    parameter of interest no interval either."""
    w, data = counting()
    empty = BayesianCalculator(data)
    made = (empty.GetPosteriorFunction(), empty.GetPosteriorPdf(), empty.GetPosteriorPlot(),
            empty.GetInterval())  # fmt: skip
    assert made == (None, None, None, None)
    said(capfd.readouterr().out, "BayesianCalculator::GetPosteriorPdf - missing pdf model",
         "BayesianCalculator::GetInterval - no parameter of interest is set")  # fmt: skip
    unset = BayesianCalculator(data, w["pdf"], ROOT.RooArgSet())
    assert unset.GetPosteriorFunction() is None
    assert "missing parameter of interest" in capfd.readouterr().out
    two = BayesianCalculator(data, w["pdf"], ROOT.RooArgSet(w["s"], w["b"]))
    assert two.GetPosteriorFunction() is None
    assert "current implementation works only on 1D intervals" in capfd.readouterr().out


def test_an_interval_without_a_posterior_is_roots_dummy(capfd) -> None:
    """A parameter of interest but no density: the cdf, then the scan, fail, and the interval
    is ROOT's dummy ``[1, 0]``; the shortest interval fails the same way."""
    w, data = counting()
    bc = BayesianCalculator(data)
    bc.SetParameters(ROOT.RooArgSet(w["s"]))
    assert limits(bc.GetInterval()) == (1.0, 0.0)
    out = capfd.readouterr().out
    assert "BayesianCalculator::GetInterval() cannot make posterior Function" in out
    assert "computing integral from cdf failed - do a scan in 100 nbins" in out
    assert "cannot compute a valid interval - return a dummy [1,0] interval" in out
    short = BayesianCalculator(data)
    short.SetParameters(ROOT.RooArgSet(w["s"]))
    short.SetShortestInterval()
    assert limits(short.GetInterval()) == (1.0, 0.0)


def test_a_posterior_rising_to_the_end_leaves_the_bins_short_of_the_level(capfd) -> None:
    """All the bins but the highest - ROOT reads one below - fall short of 95%: the interval
    is the whole range, its level 0, and RooStats warns."""
    bc = calculator(30)
    bc.SetShortestInterval()
    bc.SetScanOfPosterior(4)
    bc.SetConfidenceLevel(0.95)
    assert limits(bc.GetInterval()) == (0.001, 15.0)
    assert "actual interval CL = 0 differs more than 10%" in capfd.readouterr().out


def test_a_failed_cdf_integral_falls_back_to_a_scan(capfd, monkeypatch) -> None:
    """A cumulative posterior whose normalisation failed is given up for a scan of 100 bins."""

    class Failing(bayesian.CdfFunction):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            self.error = True

    monkeypatch.setattr(bayesian, "CdfFunction", Failing)
    bc = calculator()
    bc.SetTestSize(0.05)
    assert limits(bc.GetInterval()) == pytest.approx(SCAN100, rel=1e-13)
    out = capfd.readouterr().out
    assert "Numerical error computing CDF integral - try a different method" in out
    assert "computing integral from cdf failed - do a scan in 100 nbins" in out


def failing_roots(monkeypatch: Any, answers: list[bool]) -> None:
    """The root finder, each call's success ``answers``' next, and the cdf in error after."""
    given = iter(answers)

    def fake(cdf: Any, *args: Any) -> Any:
        _ok, root, iterations = brent_root(cdf, *args)
        cdf.error = True
        return next(given), root, iterations

    monkeypatch.setattr(bayesian, "brent_root", fake)


def test_a_root_finder_failing_at_either_end_falls_back_to_a_scan(capfd, monkeypatch) -> None:
    """The root finder failing - the cdf in error - at the lower end, or at the upper, is said,
    and the interval is the scan's."""
    scan = (0.8804034309906532, 7.253628057060863)
    failing_roots(monkeypatch, [False])
    assert limits(calculator().GetInterval()) == pytest.approx(scan, rel=1e-13)
    out = capfd.readouterr().out
    assert "BayesianCalculator: Numerical error integrating the  CDF" in out
    assert "Error from root finder when searching lower limit !" in out
    failing_roots(monkeypatch, [True, False])
    assert limits(calculator().GetInterval()) == pytest.approx(scan, rel=1e-13)
    assert "Error from root finder when searching upper limit !" in capfd.readouterr().out


def curves(plot: Any) -> list[tuple[str, int, float]]:
    """Each curve's name, number of points and the value at its middle point."""
    found = []
    for k in range(int(plot.numItems())):
        curve = plot.getObject(k)
        found.append((curve.GetName(), curve.GetN(), curve.GetPointY(curve.GetN() // 2)))
    return found


def test_the_posterior_plot_fills_the_interval_behind_the_curve() -> None:
    """``GetPosteriorPlot``: the interval filled grey behind the posterior, ROOT's curves point
    for point; normalised, the posterior pdf of the scan."""
    bc = calculator()
    plot = bc.GetPosteriorPlot()
    assert plot.GetTitle() == 'Posterior probability of parameter "s"'
    assert plot.GetYaxis().GetTitle() == "posterior function"
    name = "likelihood_times_prior_product_pdf_priorPOI_Norm[s]"
    assert curves(plot) == [(name, 104, 0.7200806647010937), (name, 116, 0.1842029595141564)]
    assert plot.getDrawOptions(name) == "F"
    bc.SetScanOfPosterior(10)
    plot = bc.GetPosteriorPlot(True, 0.1)
    name = "_posteriorPdf_likelihood_times_prior_product_pdf_priorPOI_approx_Norm[s]"
    assert curves(plot) == [(name, 104, 0.02421332048668701), (name, 106, 0.00408645841625247)]


def test_the_quantiles_of_a_parabola_take_roots_every_way() -> None:
    """``TF1::GetQuantiles``' last step: the parabola's root, a straight line where the bin's
    curvature is nil or too great, the bin's edge where its slope is nil, and through bins
    whose cumulative is ``r`` within 1e-12."""
    quantile = bayesian._quantile
    cumulative, alpha = [0.0, 0.5, 1.0], [0.0, 1.0]
    assert quantile(cumulative, alpha, [0.5, 0.5], [0.0, 0.0], 2, 2.0, 0.25) == 0.5
    assert quantile(cumulative, alpha, [0.5, 0.5], [-10.0, 0.0], 2, 2.0, 0.25) == 0.5
    assert quantile(cumulative, alpha, [0.0, 0.5], [0.0, 0.0], 2, 2.0, 0.25) == 0.0
    assert quantile(cumulative, alpha, [0.5, 0.5], [0.5, 0.0], 2, 2.0, 0.0) == 0.0
    assert quantile(cumulative, alpha, [0.5, 0.5], [0.5, 0.0], 2, 2.0, 1.0) == 2.0
    flat = [0.0, 0.5, 0.5 * (1 + 1e-13), 0.5 * (1 + 2e-13), 1.0]
    assert quantile(flat, [0.0, 1.0, 2.0, 3.0], [1.0] * 4, [0.0] * 4, 4, 4.0, 0.5) == (
        2.0 + (0.5 - flat[2]))


def test_the_saved_posterior_is_nothing_outside_its_range() -> None:
    """``TF1::GetSave`` is zero outside the saved range, and a straight line inside it."""
    line = bayesian._saved(lambda x: x * x, 0.0, 2.0, 2)
    assert (line(-0.5), line(2.5), line(0.5), line(2.0)) == (0.0, 0.0, 0.5, 4.0)
