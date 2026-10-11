"""``printLatex``: rf407's parameters as the LaTeX tables ROOT 6.40 printed for them.

The variables have rf407's fitted values and errors to the precision the
table shows them: ``a0 = 0.6 +/- 0.2`` and so on, ``alpha``, ``mean`` and
the widths constant; the initial values are the sibling's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xrdroot.roofit.categories import RooCategory
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.variables import RooRealVar

FITTED = {"a0": (0.62, 0.21), "a1": (0.19, 0.18), "bkgfrac": (0.452, 0.031),
          "sig1frac": (0.712, 0.061)}  # fmt: skip
CONSTANT = {"alpha": -1.0, "mean": 5.0, "sigma1": 0.5, "sigma2": 1.0}
INITIAL = {"a0": 0.5, "a1": 0.2, "alpha": -1.0, "bkgfrac": 0.5, "mean": 5.0, "sig1frac": 0.8,
           "sigma1": 0.5, "sigma2": 1.0}  # fmt: skip


def params(values: dict[str, Any]) -> RooArgSet:
    made = []
    for name in sorted(INITIAL):
        var = RooRealVar(name, name, INITIAL[name])
        if name in FITTED and values is FITTED:
            var.setVal(FITTED[name][0])
            var.setError(FITTED[name][1])
            var.setConstant(False)
        made.append(var)
    return RooArgSet(*made)


TWO_COLUMNS = [
    "\\begin{tabular}{lc|lc}",
    "$\\verb+a0+ $ & $  0.6\\pm 0.2$ & $\\verb+mean+ $ & $  5$\\\\",
    "$\\verb+a1+ $ & $  0.2\\pm 0.2$ & $\\verb+sig1frac+ $ & $  0.71\\pm 0.06$\\\\",
    "$\\verb+alpha+ $ & $ -1.00$ & $\\verb+sigma1+ $ & $  0.5$\\\\",
    "$\\verb+bkgfrac+ $ & $  0.45\\pm 0.03$ & $\\verb+sigma2+ $ & $  1$\\\\",
    "\\end{tabular}",
]
WITH_SIBLING = [
    "\\begin{tabular}{lcc}",
    "$\\verb+a0+ $ & $  0.6\\pm 0.2$ & $ 0.5$\\\\",
    "$\\verb+a1+ $ & $  0.2\\pm 0.2$ & $ 0.2$\\\\",
    "$\\verb+alpha+ $ & $ -1.00$ & $-1.00$\\\\",
]


def test_a_collection_prints_as_a_latex_table_in_columns(capsys: Any) -> None:
    capsys.readouterr()
    params(FITTED).printLatex(RooCmdArg("Columns", 2))
    assert capsys.readouterr().out.splitlines() == TWO_COLUMNS


def test_a_sibling_collection_is_printed_beside_without_names(capsys: Any) -> None:
    capsys.readouterr()
    params(FITTED).printLatex(RooCmdArg("Sibling", params(INITIAL)))
    assert capsys.readouterr().out.splitlines()[:4] == WITH_SIBLING


def test_a_format_and_a_file_are_taken_and_other_kinds_skipped(tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "t.tex"
    mixed = RooArgSet(RooRealVar("x", "x", 1.25), RooCategory("c", "c"))
    mixed.printLatex(RooCmdArg("Format", "NE", 3), RooCmdArg("OutputFile", str(out)))
    assert out.read_text().splitlines() == [
        "\\begin{tabular}{lc}",
        "x =  1.25\\\\",
        "\\end{tabular}",
    ]
    assert "skipping non-RooRealVar object named c" in capsys.readouterr().out
    auto = RooCmdArg("AutoPrecision", 1)
    params(FITTED).printLatex(
        RooCmdArg("Format", "NEYVU", auto), RooCmdArg("Sibling", params(INITIAL))
    )
    assert (
        capsys.readouterr().out.splitlines()[1]
        == "$\\verb+a0+ $ & $  0.6\\pm 0.2$ & $ 0.5$\\\\"
    )


def test_siblings_of_another_length_are_refused(capsys: Any) -> None:
    capsys.readouterr()
    params(FITTED).printLatex(RooCmdArg("Sibling", RooArgSet(RooRealVar("z", "z", 1))))
    assert "sibling list(s) must have same length" in capsys.readouterr().out
