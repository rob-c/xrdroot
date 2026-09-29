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


def _box(data: Any, corners: tuple[float, ...]) -> Any:
    """The box's pave, empty, as ``statOn`` dresses it."""
    box = PAVE[0](*corners, "BRNDC")
    box.SetName(f"{data.GetName()}_statBox")
    box.SetFillColor(0)
    box.SetBorderSize(1)
    box.SetTextAlign(12)
    box.SetTextSize(0.04)
    box.SetFillStyle(1001)
    return box


def _statistics(data: Any, frame: Any, cut: Any, rng: Any) -> dict[str, Any]:
    """The variables the box may show, by their letters: the RMS, the mean, the count."""
    from ..variables import RooRealVar

    count = RooRealVar("N", "Number of Events", data.sumEntries(cut, rng))
    count.setPlotLabel("Entries")
    var = frame.getPlotVar()
    mean, rms = mean_var(data, var, cut, rng), rms_var(data, var, cut, rng)
    mean.setPlotLabel("Mean")
    rms.setPlotLabel("RMS")
    return {"R": rms, "M": mean, "N": count}


def _corners(options: Any, lines: int, label: bool) -> tuple[float, ...]:
    """``Layout(xmin, xmax, ymax)``: the box's corners, 0.06 high a line and the label's."""
    xmin, xmax = float(options.get("Layout", 0, 0.65)), float(options.get("Layout", 1, 0.99))
    ymax = int(float(options.get("Layout", 2, 0.95)) * 10000) / 10000.0
    return xmin, ymax, xmax, ymax - lines * 0.06 - (0.06 if label else 0.0)


def _fill(box: Any, shown: list[Any], command: list[Any], label: str) -> None:
    """The box's lines: each variable, as the last ``Format`` says or in full, then the label."""
    for one in shown:
        box.AddText(format_command(one, command[-1]) if command else format_var(one, 2, "NELU"))
    if label:
        box.AddText(label)


def stat_on(data: Any, frame: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    """The box, added to ``frame``: ``What``, ``Label``, ``Layout``, ``Format``, ``Cut`` and
    ``CutRange`` as RooFit takes them."""
    options = commands([a for a in args if hasattr(a, "name")], kwargs)
    what = str(options.get("What", 0, "MNR")).upper()
    label = str(options.get("Label", 0, "") or "")
    cut, rng = options.get("Cut", 0, None), options.get("CutRange", 0, None)  # CutSpec
    shown = [letter for letter in "RMN" if letter in what]
    box = _box(data, _corners(options, len(shown), bool(label)))
    values = _statistics(data, frame, cut, rng)
    _fill(box, [values[letter] for letter in shown], options.every("Format"), label)
    frame.addObject(box)
    return frame
