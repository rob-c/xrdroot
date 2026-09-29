"""``RooAbsData::statOn``: a box of a dataset's statistics in the frame's variable, on a frame.

``data.statOn(frame)`` adds a ``TPaveText`` of the RMS, the mean and the
number of entries - ``What("MNR")`` in RooFit's order, R first - each a
variable formatted as ``paramOn`` formats a parameter: the mean's error the
RMS over the square root of the count, the RMS's that over the root of twice
the count, the RMS itself with Bessel's correction.
"""

from __future__ import annotations

import math
from typing import Any

from ..cmdargs import commands
from ..formatting import format_command, format_var
from .params import PAVE

__all__ = ["mean_var", "rms_var", "stat_on"]


def mean_var(data: Any, var: Any, cut: Any = None, rng: Any = None) -> Any:
    """``meanVar``: ``<var>Mean``, the mean and its error."""
    from ..variables import RooRealVar

    made = RooRealVar(f"{var.GetName()}Mean", f"Mean of {var.GetTitle()}", 0.0)
    made.setConstant(False)
    made.setPlotLabel(f"<{var.getPlotLabel()}>")
    mean, count = data.moment(var, 1, 0.0, cut, rng), data.sumEntries(cut, rng)
    rms = math.sqrt(data.moment(var, 2, mean, cut, rng) * count / (count - 1))
    made.setVal(mean)
    made.setError(rms / math.sqrt(count) if count > 0 else 0.0)
    return made


def rms_var(data: Any, var: Any, cut: Any = None, rng: Any = None) -> Any:
    """``rmsVar``: ``<var>RMS``, the RMS and its error."""
    from ..variables import RooRealVar

    made = RooRealVar(f"{var.GetName()}RMS", f"RMS         of {var.GetTitle()}", 0.0)
    made.setConstant(False)
    made.setPlotLabel(f"{var.getPlotLabel()}_{{RMS}}")
    mean, count = data.moment(var, 1, 0.0, cut, rng), data.sumEntries(cut, rng)
    rms = math.sqrt(data.moment(var, 2, mean, cut, rng) * count / (count - 1))
    made.setVal(rms)
    made.setError(rms / math.sqrt(2 * count))
    return made


def stat_on(data: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The box, added to ``frame``: ``What``, ``Label``, ``Layout``, ``Format``, ``CutSpec`` and
    ``CutRange`` as RooFit takes them."""
    from ..variables import RooRealVar

    options = commands([a for a in args if hasattr(a, "name")], kwargs)
    what = str(options.get("What", 0, "MNR")).upper()
    label = str(options.get("Label", 0, "") or "")
    cut, rng = options.get("CutSpec", 0, None), options.get("CutRange", 0, None)
    shown = [letter for letter in "RMN" if letter in what]
    xmin, xmax = float(options.get("Layout", 0, 0.65)), float(options.get("Layout", 1, 0.99))
    ymax = int(float(options.get("Layout", 2, 0.95)) * 10000) / 10000.0
    ymin = ymax - len(shown) * 0.06 - (0.06 if label else 0.0)
    box = PAVE[0](xmin, ymax, xmax, ymin, "BRNDC")
    box.SetName(f"{data.GetName()}_statBox")
    box.SetFillColor(0)
    box.SetBorderSize(1)
    box.SetTextAlign(12)
    box.SetTextSize(0.04)
    box.SetFillStyle(1001)
    count = RooRealVar("N", "Number of Events", data.sumEntries(cut, rng))
    count.setPlotLabel("Entries")
    var = frame.getPlotVar()
    mean, rms = mean_var(data, var, cut, rng), rms_var(data, var, cut, rng)
    mean.setPlotLabel("Mean")
    rms.setPlotLabel("RMS")
    command = options.every("Format")
    for letter in shown:
        one = {"R": rms, "M": mean, "N": count}[letter]
        box.AddText(format_command(one, command[-1]) if command else format_var(one, 2, "NELU"))
    if label:
        box.AddText(label)
    frame.addObject(box)
    return frame
