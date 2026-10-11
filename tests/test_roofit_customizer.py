"""``RooCustomizer``: a model's copies with nodes split by a category's states, or replaced.

The model stays as it was; each state's copy has its own ``_<state>`` stand-ins - the
caller's, or clones titled ``(<state>)`` - and its own clones of the branches above them,
reconnected to them, while every node they do not depend on is shared.
"""

from __future__ import annotations

from typing import Any

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.customizer import RooCustomizer
from xrdroot.roofit.functions import RooFormulaVar
from xrdroot.roofit.pdfs.addpdf import RooAddPdf
from xrdroot.roofit.pdfs.basic import RooGaussian, RooPolynomial
from xrdroot.roofit.variables import RooRealVar


def _model() -> dict[str, Any]:
    e = RooRealVar("E", "E", 0, 10)
    mean = RooRealVar("mean", "mean", 5, 0, 10)
    sigma = RooRealVar("sigma", "sigma", 1, 0.1, 5)
    gauss = RooGaussian("gauss", "gauss", e, mean, sigma)
    slope = RooRealVar("slope", "slope", 0.1, -1, 1)
    line = RooPolynomial("line", "line", e, [slope])
    ns, nb = RooRealVar("ns", "ns", 1, 0, 1e4), RooRealVar("nb", "nb", 1, 0, 1e4)
    model = RooAddPdf("model", "model", [gauss, line], [ns, nb])
    sample = RooCategory("sample", "sample")
    sample["A"] = 1
    sample["B"] = 2
    return {
        "e": e,
        "mean": mean,
        "gauss": gauss,
        "line": line,
        "ns": ns,
        "model": model,
        "sample": sample,
    }


def test_each_state_has_its_own_stand_ins_and_clones_above_them_the_model_its_own() -> None:
    m = _model()
    made, every = RooArgSet(), RooArgSet()
    double = RooRealVar("M", "M", 4, 0, 100)
    every.add(RooFormulaVar("ns_A", "ns in A", "M/2", [double]))
    cust = RooCustomizer(m["model"], m["sample"], made, every)
    cust.splitArgs([m["mean"]], m["sample"])
    cust.splitArg(m["ns"], m["sample"])
    a, b = cust.build("A"), cust.build("B", True)
    assert (a.GetName(), b.GetName(), a.pdfs[0].GetName()) == ("model_A", "model_B", "gauss_A")
    assert a.pdfs[1] is m["line"] and a.coefs[0] is every.find("ns_A")
    assert [one.GetName() for one in made] == ["mean_A", "mean_B", "ns_B"]
    assert made.find("mean_A").GetTitle() == "mean (A)" and len(every) == 4
    assert made.find("ns_B").getAttribute("ORIGNAME:ns") and m["gauss"].mean is m["mean"]
    made.find("mean_A").setVal(2.0)
    assert a.pdfs[0].getVal() != m["gauss"].getVal() == b.pdfs[0].getVal()
    assert (a.coefs[0].getVal(), b.coefs[0].GetTitle()) == (2.0, "ns (B)")
    assert [one.GetName() for one in cust.cloneBranchList()] == [
        "model_A",
        "gauss_A",
        "model_B",
        "gauss_B",
    ]
    assert cust.cloneLeafList() is made


def test_a_sterile_customiser_replaces_saying_so_and_refuses_to_split(capsys: Any) -> None:
    m = _model()
    wider = RooGaussian("wider", "wider", m["e"], m["mean"], RooRealVar("w", "w", 3, 1, 5))
    cust = RooCustomizer(m["model"], "cust")
    cust.replaceArg(m["gauss"], wider)
    cust.replaceArg(m["gauss"], m["line"])
    cust.splitArg(m["mean"], m["sample"])
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooCustomizer(model) ERROR: multiple replacement rules "
        "defined for gauss only using first rule\n"
        "[#0] ERROR:InputArguments -- RooCustomizer::splitArg(cust) ERROR cannot set spitting "
        "rules on this sterile customizer\n"
    )
    clone = cust.build(True)
    assert clone.GetName() == "model_cust" and clone.pdfs[0] is wider
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooCustomizer::build(model): tree node gauss will be "
        "replaced by wider\n"
        "[#1] INFO:ObjectHandling -- RooCustomizer::build(model) Branch node RooAddPdf::model "
        "cloned: depends on a replaced parameter\n"
        "[#1] INFO:ObjectHandling -- RooCustomizer::build(model) Branch node gauss is already "
        "replaced\n"
    )
    assert cust.build("A") is None and "sterile" in capsys.readouterr().out


def test_a_split_branch_is_cloned_with_a_warning_and_reused_and_refusals_are_said(
    capsys: Any,
) -> None:
    m = _model()
    cust = RooCustomizer(m["model"], m["sample"], RooArgSet())
    cust.splitArg(m["line"], m["sample"])
    cust.splitArg(m["line"], m["sample"])
    assert cust.build("C") is None
    assert capsys.readouterr().out == (
        "[#0] ERROR:InputArguments -- RooCustomizer(model) ERROR: multiple splitting rules "
        "defined for line only using first rule\n"
        "[#0] ERROR:InputArguments -- Trying to set invalid state label 'C' for category sample\n"
        "[#0] ERROR:InputArguments -- RooCustomizer::build(model): ERROR label 'C' not defined "
        "for master splitting category sample\n"
    )
    a = cust.build("A", True)
    out = capsys.readouterr().out
    assert a.pdfs[1].GetName() == "line_A" and a.pdfs[0] is m["gauss"]
    assert (
        "[#0] WARNING:ObjectHandling -- RooCustomizer::build(model): WARNING: branch node line "
        "is split but has no pre-defined specializations\n"
    ) in out and "RooCustomizer::build(model) Branch node line is already split\n" in out
    again = cust.build("A", True)
    assert again.pdfs[1] is a.pdfs[1] and again is not a
    assert (
        "Adding existing node specialization line_A to clonedMasterNodes" in capsys.readouterr().out
    )
    assert RooCustomizer(m["model"], "nothing").build() is m["model"]
