"""The binned likelihood RooFit counts for a ``RooRealSumPdf`` with the ``BinnedLikelihood``
attribute - HistFactory's channels - against ROOT 6.40.

Every number was printed by ROOT through PyROOT for the same two histograms
of four bins: the Poisson sum bin by bin, empty bins included; the same
likelihood given other data; a fit, plain and with ``SumW2Error``; the sum
taken out of a product of constraints; two channels of a ``RooSimultaneous``;
and bins with events but no expectation, which RooFit leaves out of the sum
and reports as evaluation errors, a fit backing out of them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgList, RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.data.dataset import RooDataSet
from xrdroot.roofit.fitting import fit
from xrdroot.roofit.fitting.binned import binned_part
from xrdroot.roofit.fitting.nll import RooNLLVar
from xrdroot.roofit.pdfs.basic import RooGaussian, ref
from xrdroot.roofit.pdfs.histpdf import RooHistFunc
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.pdfs.realsum import RooRealSumPdf
from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous
from xrdroot.roofit.variables import RooRealVar


class Channel:
    """Signal and background histograms in ``x``, their sum scaled by ``mu``, and data."""

    def __init__(self, signal: tuple[float, ...] = (1.0, 4.0, 2.0, 0.0),
                 mu: float = 1.0) -> None:  # fmt: skip
        self.x = RooRealVar("x", "x", 0, 4)
        self.x.setBins(4)
        self.w = RooRealVar("w", "w", 1)
        self.sig = RooHistFunc("sig", "sig", RooArgSet(self.x), self.hist("sig_dh", signal))
        bkg_dh = self.hist("bkg_dh", (3.0, 2.0, 1.0, 0.0))
        self.bkg = RooHistFunc("bkg", "bkg", RooArgSet(self.x), bkg_dh)
        self.mu = RooRealVar("mu", "mu", mu, -5, 10)
        self.model = self.sum("model", RooRealVar("one", "one", 1.0))

    def hist(self, name: str, counts: tuple[float, ...]) -> RooDataHist:
        """A histogram of ``counts``, filled as weighted events at the bins' centres."""
        data = RooDataSet(f"s{name}", "s", RooArgSet(self.x, self.w), RooCmdArg("WeightVar", "w"))
        for i, count in enumerate(counts):
            self.x.setVal(i + 0.5)
            data.add(RooArgSet(self.x), count)
        return RooDataHist(name, name, RooArgSet(self.x), data)

    def sum(self, name: str, background: Any) -> RooRealSumPdf:
        made = RooRealSumPdf(name, name, RooArgList(self.sig, self.bkg),
                             RooArgList(self.mu, background), True)  # fmt: skip
        made.setAttribute("BinnedLikelihood")
        return made


def test_the_binned_likelihood_is_the_poisson_sum_bin_by_bin(capsys: Any) -> None:
    """Counted bin by bin, the empty bin as nothing; named after the sum, not normalised."""
    ch = Channel()
    nll = ch.model.createNLL(ch.hist("obs", (5.0, 7.0, 2.0, 0.0)))
    assert nll.GetName() == "nll_model_obs"
    assert "RooAbsPdf::fitTo(model) fixing normalization" in capsys.readouterr().out
    assert nll.getVal() == 5.334787616875348
    ch.mu.setVal(2.0)
    assert nll.getVal() == 6.621639246410384


def test_a_likelihood_given_other_data_counts_those(capsys: Any) -> None:
    """``setData``: the same density's likelihood of other bins - one with events but no
    expectation left out, as RooFit leaves it."""
    ch = Channel(mu=2.0)
    nll = ch.model.createNLL(ch.hist("obs", (5.0, 7.0, 2.0, 0.0)))
    assert nll.setData(ch.hist("obs2", (4.0, 5.0, 3.0, 1.0)))
    assert nll.getVal() == 6.978314190349114


def test_the_likelihood_at_the_parameters_a_context_gives() -> None:
    """``compute`` with values of ``mu``: one likelihood per value, ``mu`` put back after."""
    ch = Channel()
    nll = ch.model.createNLL(ch.hist("obs", (5.0, 7.0, 2.0, 0.0)))
    found = nll.compute({"mu": np.array([1.0, 2.0])})
    assert found.tolist() == [5.334787616875348, 6.621639246410384]
    assert nll.compute({"mu": 2.0}) == 6.621639246410384
    assert ch.mu.getVal() == 1.0


def test_a_binned_fit_is_roots_plain_and_with_sumw2_errors() -> None:
    ch = Channel()
    obs = ch.hist("obs", (5.0, 7.0, 2.0, 0.0))
    result = ch.model.fitTo(obs, Save=True, PrintLevel=-1)
    assert (ch.mu.getVal(), ch.mu.getError(), result.minNll()) == pytest.approx(
        (1.05811116760782, 0.49895650153300536, 5.32736671115317), rel=1e-6)  # fmt: skip
    ch.mu.setVal(1.0)
    ch.model.fitTo(obs, Save=True, PrintLevel=-1, SumW2Error=True)
    assert (ch.mu.getVal(), ch.mu.getError()) == pytest.approx(
        (1.0581108592968218, 1.1818362505391165), rel=1e-6)  # fmt: skip


def test_the_first_likelihood_says_which_library_it_computes_with(capsys: Any,
                                                                  monkeypatch: Any) -> None:
    """RooFit says once per session that it uses the generic CPU library."""
    monkeypatch.setattr(fit, "_SAID_LIBRARY", [False])
    ch = Channel()
    capsys.readouterr()
    ch.model.createNLL(ch.hist("obs", (5.0, 7.0, 2.0, 0.0)))
    assert "using generic CPU library compiled with no vectorizations" in capsys.readouterr().out


def _constrained(ch: Channel) -> tuple[Any, Any, Any]:
    """The sum with a background yield ``nb``, times a Gaussian constraint of it."""
    nb = RooRealVar("nb", "nb", 1.0, 0, 5)
    nbnom = RooRealVar("nbnom", "nbnom", 1.0)
    cons = RooGaussian("cons", "cons", nbnom, nb, RooRealVar("sd", "sd", 0.1))
    model2 = ch.sum("model2", nb)
    return RooProdPdf("prod", "prod", RooArgList(model2, cons)), nb, nbnom


def test_a_binned_sum_is_taken_out_of_its_product_of_constraints(capsys: Any) -> None:
    """The product's binned factor is the likelihood's - named so - its constraint added."""
    ch = Channel()
    prod, nb, nbnom = _constrained(ch)
    capsys.readouterr()
    nll = prod.createNLL(ch.hist("obs", (5.0, 7.0, 2.0, 0.0)), GlobalObservables=RooArgSet(nbnom))
    out = capsys.readouterr().out
    assert " Including the following constraint terms in minimization: (cons)" in out
    assert "RooAbsPdf::fitTo(model2) fixing normalization" in out
    assert nll.getVal() == 3.951141057085975
    nb.setVal(1.3)
    assert nll.getVal() == 8.378645218863602


def test_a_likelihood_made_directly_normalises_a_constraint_over_its_variables() -> None:
    """With no set given the constraints are normalised over, each takes all its variables."""
    ch = Channel()
    nb = RooRealVar("nb", "nb", 1.3, 0, 5)
    cons = RooGaussian("cons", "cons", ref(1.0), nb, ref(0.1))
    prod = RooProdPdf("prod", "prod", RooArgList(ch.sum("model2", nb), cons))
    obs = ch.hist("obs", (5.0, 7.0, 2.0, 0.0))
    made = prod.createNLL(obs)
    direct = RooNLLVar(prod, obs, extended=True, constraints=[cons])
    assert direct.getVal() == pytest.approx(made.getVal(), rel=1e-14)


def test_a_main_measurement_before_the_sum_keeps_the_product_whole() -> None:
    """``getBinnedL`` stops at a factor marked ``MAIN_MEASUREMENT``: no binned channel."""
    ch = Channel()
    main = RooGaussian("main", "main", RooRealVar("n0", "n0", 1.0), RooRealVar("n", "n", 1, 0, 5),
                       RooRealVar("s0", "s0", 0.1))  # fmt: skip
    main.setAttribute("MAIN_MEASUREMENT")
    assert binned_part(RooProdPdf("pm", "pm", RooArgList(main, ch.model))) is None
    assert binned_part(RooProdPdf("pn", "pn", RooArgList(ch.sig, ch.bkg))) is None
    assert binned_part(ch.model) is ch.model


def test_a_simultaneous_likelihood_of_binned_channels_is_roots() -> None:
    """Each state's bins, the empty-expectation bin of one left out, and ``N log 2`` added."""
    ch = Channel()
    prod, _, nbnom = _constrained(ch)
    cat = RooCategory("cat", "cat")
    cat.defineType("a")
    cat.defineType("b")
    sim = RooSimultaneous("sim", "sim", cat)
    sim.addPdf(ch.model, "a")
    sim.addPdf(prod, "b")
    comb = RooDataSet("comb", "comb", RooArgSet(ch.x, ch.w), RooCmdArg("Index", cat),
                      RooCmdArg("Import", "a", ch.hist("obs", (5.0, 7.0, 2.0, 0.0))),
                      RooCmdArg("Import", "b", ch.hist("obs2", (4.0, 5.0, 3.0, 1.0))),
                      RooCmdArg("WeightVar", "w"))  # fmt: skip
    assert (comb.numEntries(), comb.sumEntries()) == (8, 27.0)
    nll = sim.createNLL(comb, GlobalObservables=RooArgSet(nbnom))
    assert nll.getVal() == 26.930461137378433


def test_a_fit_backs_out_of_bins_with_events_but_no_expectation(capfd: Any) -> None:
    """Started where some bins expect nothing: each such point an evaluation error - its bins
    listed as RooFit lists them - and the maximum so far handed to MIGRAD, as ROOT does."""
    ch = Channel(signal=(1.0, 4.0, 2.0, 1.0), mu=-0.5)
    ch.mu.setError(1.0)
    obs = ch.hist("obs", (3.0, 2.0, 1.0, 1.0))
    nll = ch.model.createNLL(obs)
    assert nll.getVal() == 1.5428872736055896
    capfd.readouterr()
    result = ch.model.fitTo(obs, Save=True, PrintLevel=-1)
    out = capfd.readouterr()
    text = out.out + out.err
    assert text.count("RooAbsMinimizerFcn: Minimized function has error status.") == 30
    assert (
        "Returning maximum FCN so far (-inf) to force MIGRAD to back out of this region. Error "
        "log follows.\nParameter values: \tmu=-0.510053\nRooFit::Detail::RooNLLVarNew::"
        "RooNLLVarNew[ func=model weightVar=_weight _weight_sumW2=_weight_sumW2 ]\n"
        "     Observed 2.000000 events in bin 1 with zero event yield @ func=model=-0.510053, "
        "weightVar=_weight=1, _weight_sumW2=_weight_sumW2=1\n"
        "     Observed 1.000000 events in bin 2 with zero event yield @ func=model=-0.510053, "
        "weightVar=_weight=1, _weight_sumW2=_weight_sumW2=1\n"
    ) in text
    assert (ch.mu.getVal(), result.minNll(), result.status()) == pytest.approx(
        (-0.5, 6.031228219081424, 300), rel=1e-12)  # fmt: skip
