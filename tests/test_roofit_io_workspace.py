"""Reading a ``RooWorkspace`` ROOT wrote into the engine's own objects, held to ROOT's numbers.

The files in ``tests/data`` were written by ROOT 6.40: ``roofit-workspace-zoo.root`` a
workspace of one of each class the reader makes - densities, functions, a category, a
conditional product, a simultaneous density, datasets weighted and binned, a snapshot, named
sets, a ``ModelConfig`` - and a second workspace whose dataset lives in a ``TTree``; the
``hf-*`` files HistFactory's ``hf001`` workspaces (Gaussian, Gamma and log-normal
constraints). Every value asserted is what ROOT printed for the same file.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import Any

import pytest

from xrdroot import open_root
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.messages import service

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def workspace(filename: str, name: str) -> Any:
    with open_root(str(DATA / filename)) as f:
        return f[name]


@pytest.fixture(scope="module")
def zoo() -> Any:
    return workspace("roofit-workspace-zoo.root", "w")


@pytest.mark.parametrize(("name", "value"), [
    ("g", 0.004431850952732091), ("e", 0.155047391905797), ("u", 0.1),
    ("l", 0.18173789816466834), ("bw", 0.018371639320712643), ("ext", 0.004431850952732091),
    ("p", 0.2241073225066671), ("rs", 0.09010989010989011), ("pp", 0.007915457849071123),
    ("pq", 0.006192813962108764), ("sim", 0.004431850952732091),
    ("gm", 0.2708058066838553),
])  # fmt: skip
def test_every_density_read_back_has_roots_normalised_value(zoo: Any, name: str,
                                                            value: float) -> None:  # fmt: skip
    """Each density of the zoo, normalised over ``x`` at the written ``x = 2``, is ROOT's."""
    assert zoo.pdf(name).getVal(RooArgSet(zoo.var("x"))) == pytest.approx(value, rel=1e-9)


def test_the_functions_read_back_have_roots_values(zoo: Any) -> None:
    """A polynomial, a sum and a product of what the file holds evaluate as ROOT's."""
    assert [zoo.function(n).getVal() for n in ("pv", "add", "prd")] == [2.0, 6.0, 5.0]


def test_the_workspace_prints_as_root_printed_it(zoo: Any, capsys: Any) -> None:
    """Its variables, densities, functions, datasets, snapshot, sets and ModelConfig, listed."""
    zoo.Print()
    out = capsys.readouterr().out
    assert "(a0,a1,a2,bb,c,cat,gg,k,m,m0,mb,ml,mp,mu0,n,s,sl,wb,x,y)" in out
    for line in ("RooExtendPdf::ext[ pdf=g n=n ] = 0.011109",
                 "RooLognormal::ln[ x=x m0=m0 k=k ] = 0.491956",
                 "RooSimultaneous::sim[ indexCat=cat A=g B=e ] = 0.011109",
                 "RooUniform::u[ x=(x) ] = 1",
                 "RooRealSumPdf::rs[ a2 * pv + [%] * prd ] = 4.1",
                 "RooPolyVar::pv[ x=x coefList=(a0,a1) ] = 2",
                 "RooDataSet::cd(x,cat)", "RooDataHist::dh(x)",
                 "snap = (x=2,m=5[C],s=1,c=-0.2,ml=3,sl=1,m0=2,k=1.5,mb=5,wb=1,n=100,mp=3,a0=1[C],"
                 "a1=0.5[C],a2=0.3,gg=2,bb=1,mu0=0[C],y=1[C])",
                 "mc_POI:(s)", "RooStats::ModelConfig::mc"):  # fmt: skip
        assert line in out


def test_a_variable_keeps_its_bins_named_binnings_ranges_unit_and_attributes(zoo: Any) -> None:
    """``x``'s four bins, its variable binning ``var`` and range ``side``; ``m``'s attributes."""
    x, m = zoo.var("x"), zoo.var("m")
    assert x.getBins() == 4
    assert list(x.getBinning("var").array()) == [0.0, 2.0, 7.0, 10.0]
    assert (x.getMin("side"), x.getMax("side"), x.getUnit()) == (1.0, 3.0, "GeV")
    assert (m.isConstant(), m.getAttribute("marked"), m.getStringAttribute("key")) == (
        True, True, "value")  # fmt: skip


def test_the_datasets_keep_their_events_weights_and_global_observables(zoo: Any) -> None:
    """Five events; two weighted three and six in all; none with a category; four bins."""
    counts = [(zoo.data(n).numEntries(), zoo.data(n).sumEntries()) for n in ("d", "wd", "cd", "dh")]
    assert counts == [(5, 5.0), (2, 6.0), (0, 0.0), (4, 5.0)]
    assert [one.GetName() for one in zoo.data("wd").getGlobalObservables()] == ["m"]


def test_the_model_config_names_the_workspaces_density_and_sets(zoo: Any) -> None:
    """The ``ModelConfig`` read back points at the workspace it was read with."""
    config = zoo.obj("mc")
    assert config.GetPdf() is zoo.pdf("g")
    assert [one.GetName() for one in config.GetParametersOfInterest()] == ["s"]


def test_a_dataset_kept_in_a_tree_is_refused_by_name() -> None:
    """``RooTreeDataStore`` is not the vector store datasets are read from, and says so."""
    with pytest.raises(UnsupportedFeatureError, match="stored in a TTree"):
        workspace("roofit-workspace-zoo.root", "tree")


@pytest.mark.parametrize(("filename", "nll", "moved"), [
    ("hf-gauss-combined.root", 4.549054846130797, 4.6584307966316025),
    ("hf-gamma-combined.root", 3.3582388752454717, 3.747675274930624),
])  # fmt: skip
def test_a_histfactory_workspace_read_back_has_roots_likelihood(filename: str, nll: float,
                                                                 moved: float) -> None:  # fmt: skip
    """``hf001``'s combined model: ROOT's density and likelihood, before and after a pull."""
    ws = workspace(filename, "combined")
    config = ws.obj("ModelConfig")
    pdf = config.GetPdf()
    assert pdf.getVal(config.GetObservables()) == pytest.approx(1.0434782608695652, rel=1e-12)
    likelihood = pdf.createNLL(ws.data("obsData"))
    assert likelihood.getVal() == pytest.approx(nll, rel=1e-12)
    ws.var("alpha_syst2" if "gauss" in filename else "beta_syst2").setVal(
        0.5 if "gauss" in filename else 1.2)  # fmt: skip
    assert likelihood.getVal() == pytest.approx(moved, rel=1e-12)


def test_a_gamma_constrained_systematic_reads_back_its_polynomial_and_gamma(capsys: Any) -> None:
    """At ``beta = 1.2``, ``alphaOfBeta_syst2`` and ``beta_syst2Constraint`` print as ROOT's."""
    ws = workspace("hf-gamma-combined.root", "combined")
    ws.var("beta_syst2").setVal(1.2)
    ws.function("alphaOfBeta_syst2").Print()
    ws.pdf("beta_syst2Constraint").Print()
    assert capsys.readouterr().out == (
        "RooPolyVar::alphaOfBeta_syst2[ x=beta_syst2 coefList=(-3.33333,3.33333) ] = 0.666667\n"
        "RooGamma::beta_syst2Constraint[ x=beta_syst2 gamma=k_nom_beta_syst2 beta=theta_syst2 "
        "mu=0 ] = 0.88062\n"
    )


def test_the_histfactory_functions_print_their_proxies_twice_as_root_does(capsys: Any) -> None:
    """A ``RooHistFunc`` and a ``RooBinWidthFunction`` read back list their proxy twice."""
    ws = workspace("hf-gauss-combined.root", "combined")
    ws.function("signal_channel1_Hist_alphanominal").Print()
    ws.function("channel1_model_binWidth").Print()
    ws.function("signal_channel1_epsilon").Print()
    assert capsys.readouterr().out == (
        "RooHistFunc::signal_channel1_Hist_alphanominal[ depList=(obs_x_channel1) "
        "depList=(obs_x_channel1) ] = 20\n"
        "RooBinWidthFunction::channel1_model_binWidth[ HistFuncForBinWidth=signal_channel1_Hist_"
        "alphanominal HistFuncForBinWidth=signal_channel1_Hist_alphanominal ] = 2\n"
        "RooStats::HistFactory::FlexibleInterpVar::signal_channel1_epsilon[ paramList=(alpha_"
        "syst1) ] = 1\n"
    )


def test_a_log_normal_systematics_formula_is_refused_by_its_class() -> None:
    """``alphaOfBeta_syst2`` is a ``RooFormulaVar``, a class the reader has no maker for."""
    with pytest.raises(UnsupportedFeatureError, match="RooFormulaVar \\('alphaOfBeta_syst2'\\)"):
        workspace("hf-lognorm-channel.root", "channel1")
