"""``RooRealVar::format``: a variable as text - ``mean = 1.02 +/- 0.03`` - to the precision its
error says.

The value is rounded to as many decimals as its error has significant
figures (two by default), which ``paramOn`` boxes and ``printLatex`` tables
show. The options are ROOT's letters: ``N`` name, ``T`` title, ``E`` error,
``A`` asymmetric error, ``U`` unit, ``H`` hide the value, ``L`` TLatex,
``X`` LaTeX, ``P`` precision from the error, ``F`` fixed precision.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = ["format_var", "format_command"]


def _digits(var: Any, sig: int, by_error: bool) -> tuple[int, int]:
    value, error = var.getVal(), var.getError()
    lead_val = (
        math.floor(math.log10(abs(error + 1e-10)))
        if by_error
        else math.floor(math.log10(abs(value + 1e-10)))
    )
    if (by_error and value == 0 and error == 0) or (not by_error and value == 0):
        lead_val = 0
    lead_err = math.floor(math.log10(abs(error + 1e-10)))
    where_val = lead_val - sig + 1 - (1 if value < 0 else 0)
    where_err = lead_err - sig + 1
    return max(-where_val, 0), max(-where_err, 0)


def format_var(var: Any, sig: int = 2, options: str = "") -> str:
    """``RooRealVar::format(sigDigits, options)``."""
    opts = options.lower()
    show_error = "e" in opts and var.hasError(False)
    latex = "x" in opts or "y" in opts
    by_error = (
        (("e" in opts) and var.hasError(False) and not var.isConstant()) or "p" in opts
    ) and "f" not in opts
    val_digits, err_digits = _digits(var, max(sig, 1), by_error)
    text = "$" if latex else ""
    label = var.GetTitle() if "t" in opts else (var.getPlotLabel() if "n" in opts else "")
    if label:
        text += label + " = "
    if var.getVal() >= 0:
        text += " "
    if "h" not in opts:
        text += f"{var.getVal():.{val_digits}f}"
    asym = "a" in opts and var.hasAsymError(False)
    if show_error and not asym:
        sign = " #pm " if "l" in opts else ("\\pm " if latex else " +/- ")
        text += f"{sign}{var.getError():.{err_digits}f}"
    if asym and "e" in opts:
        low, high = var.getAsymErrorLo(), var.getAsymErrorHi()
        text += _asymmetric(low, high, err_digits, opts, latex)
    if var.getUnit() and "u" in opts:
        text += " " + var.getUnit()
    return text + ("$" if latex else "")


def _asymmetric(low: float, high: float, digits: int, opts: str, latex: bool) -> str:
    lo, hi = f"{low:.{digits}f}", f"{high:.{digits}f}"
    if "l" in opts:
        return f" #pm _{{{lo}}}^{{+{hi}}}"
    if latex:
        return f"\\pm _{{{lo}}}^{{+{hi}}}"
    return f" +/-  ({lo}, {hi})"


def format_command(var: Any, command: Any) -> str:
    """``RooRealVar::format(RooCmdArg)``: ``Format("NEU", AutoPrecision(1))`` and its kin."""
    what = str(command.value(0, ""))
    sig = 2
    for extra in command.args[1:]:
        name = getattr(extra, "name", "")
        if name == "AutoPrecision":
            what, sig = what + "P", int(extra.value(0, 2))
        elif name == "FixedPrecision":
            what, sig = what + "F", int(extra.value(0, 2))
        elif name == "TLatexStyle":
            what += "L"
        elif name in ("LatexStyle", "LatexTableStyle"):
            what += "X"
        elif name == "VerbatimName":
            what += "V"
    return format_var(var, sig, what)
