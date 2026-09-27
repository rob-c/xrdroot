"""``RooRealVar::format``: a variable as the text ``paramOn`` and ``printLatex`` show.

Every expected string came from ROOT 6.40's ``RooRealVar::format`` on a
variable ``mean`` titled ``Mean #mu`` in ``GeV``, with the same value,
errors and options. The value is rounded to as many decimals as the error
has significant figures, so each case below is one of ROOT's roundings.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.formatting import format_command, format_var
from xrdroot.roofit.variables import RooRealVar


def var(value: float, error: Any = None, asym: Any = None, const: bool = False) -> RooRealVar:
    made = RooRealVar("mean", "Mean #mu", value, -10, 10, "GeV")
    if error is not None:
        made.setError(error)
    if asym is not None:
        made.setAsymError(*asym)
    made.setConstant(const)
    return made


FITTED = (1.01746, 0.0300144)
ASYM = (-0.03, 0.031)


@pytest.mark.parametrize(
    ("made", "sig", "options", "expected"),
    [
        (var(*FITTED), 2, "NEU", "mean =  1.017 +/- 0.030 GeV"),
        (var(*FITTED), 2, "NELU", "mean =  1.017 #pm 0.030 GeV"),
        (var(*FITTED), 2, "TEXU", "$Mean #mu =  1.017\\pm 0.030 GeV$"),
        (var(*FITTED), 3, "NE", "mean =  1.0175 +/- 0.0300"),
        (var(*FITTED, ASYM), 2, "NEAU", "mean =  1.017 +/-  (-0.030, 0.031) GeV"),
        (var(*FITTED, ASYM), 2, "NEALU", "mean =  1.017 #pm _{-0.030}^{+0.031} GeV"),
        (var(*FITTED, ASYM), 2, "XEA", "$ 1.017\\pm _{-0.030}^{+0.031}$"),
        (var(*FITTED, ASYM), 2, "NA", "mean =  1.0"),
        (var(*FITTED), 2, "NH", "mean =  "),
        (var(*FITTED), 2, "NHE", "mean =   +/- 0.030"),
        (var(*FITTED), 2, "NP", "mean =  1.017"),
        (var(*FITTED), 4, "NEF", "mean =  1.017 +/- 0.03001"),
        (var(*FITTED), 2, "", " 1.0"),
        (var(*FITTED, const=True), 2, "NE", "mean =  1.0 +/- 0.030"),
        (var(-2.345, 0.012), 2, "NE", "mean = -2.3450 +/- 0.012"),
        (var(-2.345), 2, "N", "mean = -2.35"),
        (var(0.0, 0.0), 2, "NEP", "mean =  0.0"),
        (var(0.0), 2, "N", "mean =  0.0"),
        (var(123.456), 3, "NU", "mean =  10.0 GeV"),
        (var(1.5), 0, "N", "mean =  2"),
        (var(*FITTED), 2, "YE", "$ 1.017\\pm 0.030$"),
    ],
)
def test_a_variable_formats_as_root_formats_it(
    made: RooRealVar, sig: int, options: str, expected: str
) -> None:
    """Each option letter, and each rounding the error or value calls for, is ROOT's."""
    assert made.format(sig, options) == expected
    assert format_var(made, sig, options) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            RooCmdArg("Format", "NEU", RooCmdArg("AutoPrecision", 1)),
            "mean =  1.02 +/- 0.03 GeV",
        ),
        (
            RooCmdArg("Format", "NEU", RooCmdArg("FixedPrecision", 3)),
            "mean =  1.02 +/- 0.0300 GeV",
        ),
        (RooCmdArg("Format", "NE", RooCmdArg("TLatexStyle")), "mean =  1.017 #pm 0.030"),
        (RooCmdArg("Format", "NE", RooCmdArg("LatexStyle")), "$mean =  1.017\\pm 0.030$"),
        (RooCmdArg("Format", "NE"), "mean =  1.017 +/- 0.030"),
    ],
)
def test_a_format_command_formats_as_root_formats_it(command: RooCmdArg, expected: str) -> None:
    """``Format("NEU", AutoPrecision(1))`` is how ``paramOn`` is told what to show."""
    assert var(*FITTED).format(command) == expected


def test_a_format_command_ignores_what_it_does_not_know() -> None:
    """Only the precision and style commands change the text; anything else passes by."""
    command = RooCmdArg("Format", "NE", RooCmdArg("Unknown"), 7)
    assert format_command(var(*FITTED), command) == "mean =  1.017 +/- 0.030"


def test_a_latex_table_style_puts_the_name_and_value_in_two_columns() -> None:
    """``printLatex`` tables separate the name from the value with ``$ & $``, as ROOT does."""
    command = RooCmdArg("Format", "NE", RooCmdArg("LatexTableStyle"))
    assert var(*FITTED).format(command) == "$mean $ & $  1.017\\pm 0.030$"


def test_a_verbatim_name_is_printed_as_root_prints_it() -> None:
    """ROOT 6.40 prints ``mean+`` for ``Format("NE", VerbatimName())``."""
    command = RooCmdArg("Format", "NE", RooCmdArg("VerbatimName"))
    assert var(*FITTED).format(command) == "mean+ =  1.017 +/- 0.030"
