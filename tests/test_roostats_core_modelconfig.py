"""RooStats' ModelConfig: the names of a model's density and sets, kept in its workspace.

What a ModelConfig prints, the sets it defines in its workspace, the
messages it gives when it has no workspace or is handed something that is
not a variable, and its snapshot, are held to what ROOT 6.40 printed for the
same calls through PyROOT.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import ERROR, WARNING, service
from xrdroot.roostats.modelconfig import ModelConfig, quieted


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def workspace() -> Any:
    """``g(x | mu, s)`` times a constraint ``c(m0 | mu, 0.5)``, and a flat prior on ``mu``."""
    w = ROOT.RooWorkspace("w")
    w.factory("Gaussian::g(x[0,-5,5],mu[1,-3,3],s[1,0.1,3])")
    w.factory("Gaussian::c(m0[1,-3,3],mu,0.5)")
    w.factory("PROD::model(g,c)")
    w.factory("Uniform::prior(mu)")
    return w


def proto(w: Any) -> Any:
    """Three entries of ``x``, named ``proto``."""
    x = w.var("x")
    data = ROOT.RooDataSet("proto", "proto", ROOT.RooArgSet(x))
    for value in (0.5, -0.25, 1.0):
        x.setVal(value)
        data.add(ROOT.RooArgSet(x))
    x.setVal(0.0)
    return data


FULL = """
=== Using the following for full ===
Observables:             RooArgSet:: = (x,m0)
Parameters of Interest:  RooArgSet:: = (mu)
Nuisance Parameters:     RooArgSet:: = (s)
Global Observables:      RooArgSet:: = (m0)
Constraint Parameters:   RooArgSet:: = (s)
Conditional Observables: RooArgSet:: = (x)
Proto Data:              RooDataSet::proto[x] = 3 entries
PDF:                     RooProdPdf::model[ g * c ] = 0.606531
Prior PDF:               RooUniform::prior[ x=(mu) ] = 1

"""


def test_a_full_model_config_prints_every_set_and_density_as_root_does(capsys: Any) -> None:
    """Every set named, the non-variables refused, the rest guessed: ROOT 6.40's printout."""
    w = workspace()
    data = proto(w)
    mc = ROOT.RooStats.ModelConfig("full", "a title", w)
    assert (mc.GetName(), mc.GetTitle()) == ("full", "a title")
    mc.SetPdf(w.pdf("model"))
    mc.SetPriorPdf(w.pdf("prior"))
    mc.SetProtoData(data)
    mc.SetParametersOfInterest(ROOT.RooArgSet(w.var("mu")))
    mc.SetObservables(ROOT.RooArgSet(w.var("x"), w.var("m0")))
    mc.SetConstraintParameters("s")
    mc.SetConditionalObservables("x")
    mc.SetExternalConstraints(ROOT.RooArgSet(w.pdf("c")))
    mc.SetParametersOfInterest(ROOT.RooArgSet(w.pdf("g")))
    mc.SetNuisanceParameters(ROOT.RooArgSet(w.pdf("g")))
    mc.SetObservables("g")
    mc.GuessObsAndNuisance(data, False)
    mc.Print()
    assert capsys.readouterr().out == (
        "ModelConfig::SetParametersOfInterest ERROR: specified set contains non-parameters: (g)\n"
        "ModelConfig::SetNuisanceParameters ERROR: specified set contains non-parameters: (g)\n"
        "ModelConfig::SetObservables ERROR: specified set contains non-parameters: (g)\n" + FULL
    )
    assert w.var("m0").isConstant()
    assert [p.GetName() for p in mc.GetExternalConstraints()] == ["c"]
    assert mc.GetPriorPdf() is w.pdf("prior")
    assert mc.GetProtoData() is w.data("proto")
    assert sorted(w._sets) == [
        "full_ConditionalObservables", "full_ConstrainedParams", "full_ExternalConstraints",
        "full_GlobalObservables", "full_NuisParams", "full_Observables", "full_POI",
    ]


def test_parameters_of_interest_by_name_are_refused_as_set_parameters(capsys: Any) -> None:
    """``SetParametersOfInterest("g")`` is ``SetParameters``' own, and says so, as in ROOT."""
    w = workspace()
    mc = ModelConfig("mc", w)
    mc.SetParametersOfInterest("g")
    mc.SetParameters(ROOT.RooArgSet(w.pdf("c")))
    assert capsys.readouterr().out == (
        "ModelConfig::SetParameters ERROR: specified set contains non-parameters: (g)\n"
        "ModelConfig::SetParameters ERROR: specified set contains non-parameters: (c)\n"
    )
    assert mc.GetParametersOfInterest() is None
    mc.SetParameters("mu")
    assert mc.GetParametersOfInterest().names() == ["mu"]


def test_a_model_config_is_made_from_a_workspace_a_name_or_nothing(capsys: Any) -> None:
    """``ModelConfig(ws)``, ``(name, ws)``, ``(name, title)`` and ``()``: a RooStats ``TNamed``."""
    w = workspace()
    assert ModelConfig(w).GetWS() is w
    assert ModelConfig(w).GetName() == ""
    named = ModelConfig("mc", w)
    assert (named.GetName(), named.GetTitle(), named.GetWorkspace()) == ("mc", "mc", w)
    titled = ModelConfig("mc", "title")
    assert (titled.GetName(), titled.GetTitle()) == ("mc", "title")
    titled.SetName("other")
    titled.SetTitle("another")
    assert (titled.GetName(), titled.GetTitle()) == ("other", "another")
    assert titled.ClassName() == "RooStats::ModelConfig"
    assert all(titled.InheritsFrom(k) for k in ("RooStats::ModelConfig", "TNamed", "TObject"))
    assert not titled.InheritsFrom("RooAbsPdf")
    assert capsys.readouterr().out == ""


def test_a_clone_names_the_same_sets_in_the_same_workspace() -> None:
    """``Clone`` keeps the workspace and every name, and takes a new name if given one."""
    w = workspace()
    mc = ModelConfig("mc", w)
    mc.SetPdf("model")
    mc.SetParametersOfInterest("mu")
    same, renamed = mc.Clone(), mc.Clone("copy")
    assert same.GetName() == "mc" and renamed.GetName() == "copy"
    assert renamed.GetWS() is w and renamed.GetPdf() is w.pdf("model")
    assert renamed.GetParametersOfInterest().names() == ["mu"]


def test_without_a_workspace_every_call_says_so_and_does_nothing(capsys: Any) -> None:
    """ROOT 6.40: ``workspace not set``, once for each call that needs one."""
    mc = ModelConfig()
    assert (mc.GetName(), mc.GetTitle()) == ("", "")
    assert mc.GetPdf() is None
    assert mc.GetParametersOfInterest() is None
    assert mc.GetPriorPdf() is None
    assert mc.GetProtoData() is None
    mc.SetPdf("model")
    mc.SetProtoData("data")
    mc.SetParametersOfInterest("mu")
    mc.SetGlobalObservables("m0")
    mc.SetSnapshot(ROOT.RooArgSet())
    mc.LoadSnapshot()
    said = capsys.readouterr().out.splitlines()
    assert said == ["[#0] ERROR:ObjectHandling -- workspace not set"] * 10
    var = ROOT.RooRealVar("v", "v", 1.0, 0.0, 2.0)
    mc.SetObservables(ROOT.RooArgSet(var))
    mc.SetGlobalObservables(ROOT.RooArgSet(var))
    assert var.isConstant()
    assert capsys.readouterr().out.count("workspace not set") == 2


def test_a_second_workspace_is_refused_but_can_replace_the_first(capsys: Any) -> None:
    """``SetWS`` once only, as ROOT says; ``ReplaceWS`` whenever."""
    first, second = workspace(), workspace()
    mc = ModelConfig("mc")
    mc.SetWorkspace(first)
    mc.SetWS(second)
    assert capsys.readouterr().out == (
        "[#0] ERROR:ObjectHandling -- ModelConfig::SetWS(): workspace already set, "
        "not doing anything\n"
    )
    assert mc.GetWS() is first
    mc.ReplaceWS(second)
    assert mc.GetWS() is second


def test_a_density_or_dataset_the_workspace_lacks_is_imported_or_refused(capsys: Any) -> None:
    """An object is imported quietly; a name the workspace does not have raises, as ROOT's."""
    w = ROOT.RooWorkspace("w")
    x = ROOT.RooRealVar("x", "x", 0.0, -5.0, 5.0)
    gauss = ROOT.RooGaussian("gauss", "gauss", x, ROOT.RooFit.RooConst(0.0),
                             ROOT.RooFit.RooConst(1.0))  # fmt: skip
    mc = ModelConfig("mc", w)
    mc.SetPdf(gauss)
    assert w.pdf("gauss") is mc.GetPdf()
    mc.SetPdf(gauss)
    assert capsys.readouterr().out == ""
    with pytest.raises(RuntimeError, match="pdf nothing does not exist in workspace"):
        mc.SetPriorPdf("nothing")
    with pytest.raises(RuntimeError, match="dataset nothing does not exist in workspace"):
        mc.SetProtoData("nothing")
    assert capsys.readouterr().out == (
        "[#0] ERROR:ObjectHandling -- pdf nothing does not exist in workspace\n"
        "[#0] ERROR:ObjectHandling -- dataset nothing does not exist in workspace\n"
    )


SNAPSHOT = re.compile(r"  1\) 0x[0-9a-f]+ RooRealVar:: mu = 1  L\(-3 - 3\)  \"mu\"\n\n\Z")


def test_a_snapshot_keeps_its_values_and_leaves_the_workspaces_alone(capsys: Any) -> None:
    """``mc__snapshot`` holds ``mu = 1``; reading it back does not move ``mu`` from 2."""
    w = workspace()
    mc = ModelConfig("mc", w)
    mc.SetPdf("model")
    mc.SetSnapshot(ROOT.RooArgSet(w.var("mu")))
    w.var("mu").setVal(2.0)
    assert mc.GetSnapshot().getRealValue("mu") == 1.0
    assert w.var("mu").getVal() == 2.0
    mc.SetSnapshot(ROOT.RooArgSet(w.var("mu")))  # a second time replaces the set, quietly
    w.var("mu").setVal(1.0)
    mc.SetSnapshot([w.var("mu")])
    w.var("mu").setVal(2.0)
    mc.LoadSnapshot()
    assert w.var("mu").getVal() == 1.0
    assert w.set("mc__snapshot").names() == ["mu"]
    capsys.readouterr()
    mc.Print()
    out = capsys.readouterr().out
    assert out.startswith(
        "\n=== Using the following for mc ===\n"
        "PDF:                     RooProdPdf::model[ g * c ] = 0.606531\n"
        "Snapshot:                \n"
    )
    assert SNAPSHOT.search(out)


def test_an_empty_snapshot_is_none_and_an_unnamed_one_is_just_snapshot() -> None:
    """ROOT 6.40: ``ModelConfig(w).SetSnapshot(RooArgSet())`` saves ``snapshot`` - and none."""
    w = workspace()
    mc = ModelConfig(w)
    assert mc.GetSnapshot() is None
    mc.SetSnapshot(ROOT.RooArgSet())
    assert w.set("snapshot") is not None
    assert mc.GetSnapshot() is None


def test_guessing_takes_observables_from_the_data_and_nuisances_from_the_rest(
    capsys: Any,
) -> None:
    """ROOT 6.40's guess of ``g * c``, printed to the INFO stream as ``Print`` prints it."""
    w = workspace()
    mc = ModelConfig("guess", w)
    mc.SetPdf(w.pdf("model"))
    mc.SetParametersOfInterest("mu")
    mc.GuessObsAndNuisance(proto(w))
    assert capsys.readouterr().out == (
        "\n=== Using the following for guess ===\n"
        "Observables:             RooArgSet:: = (x)\n"
        "Parameters of Interest:  RooArgSet:: = (mu)\n"
        "Nuisance Parameters:     RooArgSet:: = (m0,s)\n"
        "PDF:                     RooProdPdf::model[ g * c ] = 0.606531\n\n"
    )
    assert mc.GetGlobalObservables() is None


def test_guessing_from_a_set_finds_global_observables_and_may_find_no_nuisances(
    capsys: Any,
) -> None:
    """Observables the data lacks are global; with every parameter constant, no nuisances."""
    w = workspace()
    w.var("s").setConstant(True)
    mc = ModelConfig("guess", w)
    mc.SetPdf("model")
    mc.SetObservables("x,m0")
    mc.GuessObsAndNuisance(ROOT.RooArgSet(w.var("x")), False)
    assert mc.GetGlobalObservables().names() == ["m0"]
    assert mc.GetNuisanceParameters().names() == ["mu"]  # no parameters of interest named
    w.var("mu").setConstant(True)
    other = ModelConfig("other", w)
    other.SetPdf("model")
    other.SetObservables("x,m0")
    other.GuessObsAndNuisance(ROOT.RooArgSet(w.var("x")), False)
    assert other.GetGlobalObservables() is None  # m0 is constant already
    assert other.GetNuisanceParameters() is None
    assert capsys.readouterr().out == ""


def test_a_fit_through_the_model_config_carries_its_global_observables() -> None:
    """``fitTo`` and ``createNLL`` add the conditional and global observables and the
    external constraints the ModelConfig names."""
    w = workspace()
    data = proto(w)
    mc = ModelConfig("mc", w)
    mc.SetPdf("model")
    mc.SetGlobalObservables("m0")
    mc.SetConditionalObservables(ROOT.RooArgSet())
    mc.SetExternalConstraints(ROOT.RooArgSet())
    nll = mc.createNLL(data)
    assert nll.getVal() == pytest.approx(
        w.pdf("model").createNLL(data, ROOT.RooFit.GlobalObservables(w.var("m0"))).getVal()
    )
    result = mc.fitTo(data, ROOT.RooFit.PrintLevel(-1), Save=True)
    assert result.status() == 0


def test_quieted_holds_messages_back_and_puts_the_level_back(capsys: Any) -> None:
    """``setGlobalKillBelow`` for the block, and as it was after it - even after an error."""
    service().setGlobalKillBelow(WARNING)
    with pytest.raises(ValueError), quieted():
        assert service().globalKillBelow() == ERROR
        raise ValueError
    assert service().globalKillBelow() == WARNING


def test_guessing_keeps_whatever_was_named_already() -> None:
    """Named global observables and nuisances are left as they were named."""
    w = workspace()
    mc = ModelConfig("mc", w)
    mc.SetPdf("model")
    mc.SetObservables("x")
    mc.SetGlobalObservables("m0")
    mc.SetNuisanceParameters("mu")
    mc.GuessObsAndNuisance(ROOT.RooArgSet(w.var("x")), False)
    assert mc.GetNuisanceParameters().names() == ["mu"]
    plain = ModelConfig("plain", w)
    plain.SetPdf("model")
    data = proto(w)
    assert plain.createNLL(data).getVal() == w.pdf("model").createNLL(data).getVal()
