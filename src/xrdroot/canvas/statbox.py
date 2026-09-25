"""A histogram's stats box, as ``THistPainter::PaintStat`` writes it.

ROOT keeps a histogram's stats box in the histogram's list of functions -
a ``TPaveStats``, saved with the lines it was last drawn with - and, when a
histogram that has none is drawn, makes one from ``gStyle``: the name,
entries, mean and standard deviation (``SetOptStat(1111)``) at the top right
of the pad. A saved box draws the lines it was saved with; this is the
other case, and the lines are worked out the way ROOT works them out,
digit by digit of ``fOptStat`` and of ``fOptFit`` for a fitted histogram.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..function import Function
from ..stats import prob
from .model import Primitive, lookup
from .paves import columns, draw_box
from .scene import Scene

__all__ = ["default_stats", "stats_rows"]

#: ``gStyle``'s own ``fOptStat`` and ``fOptFit``: a fit is not described
#: unless the session that drew it asked, which it saved in the box.
OPT_STAT = 1111
OPT_FIT = 0
#: ``gStyle``'s place for a stats box: its top right corner and its size, in NDC.
STAT_X, STAT_Y, STAT_W = 0.98, 0.935, 0.2
#: How tall each line of a stats box made from ``gStyle`` is.
STAT_LINE = 0.04
#: ``TH1::kNoStats``: the bit a histogram drawn without a stats box carries.
NO_STATS = 1 << 9
#: ``TF1::kNotDraw``: the bit of a function that is not drawn with its histogram.
NOT_DRAW = 1 << 9


def _value(number: float) -> str:
    """A number as ``gStyle``'s ``"6.4g"`` writes it."""
    return f"{number:.4g}"


Rows = Callable[[Any, int], list[tuple[str, str]]]


def _moment(name: str, what: str, error: str) -> Rows:
    """One line per axis of a moment, with its error when the digit is 2."""

    def rows(h: Any, digit: int) -> list[tuple[str, str]]:
        axes = len(h.axes)
        made = []
        for axis in range(axes):
            label = f"{name} {'xyz'[axis]}" if axes > 1 else name
            value = _value(getattr(h, what)(axis))
            if digit > 1:
                value += " #pm " + _value(getattr(h, error)(axis))
            made.append((label, value))
        return made

    return rows


def _single(name: str, what: Callable[[Any], float], any_axes: bool = False) -> Rows:
    """One line of one number - of a one-dimensional histogram, unless ``any_axes``."""

    def rows(h: Any, _digit: int) -> list[tuple[str, str]]:
        if len(h.axes) > 1 and not any_axes:
            return []
        return [(name, _value(what(h)))]

    return rows


#: What each digit of ``fOptStat`` after the first asks for, the lowest first.
STAT_LINES: tuple[Rows, ...] = (
    _single("Entries", lambda h: h.entries, any_axes=True),
    _moment("Mean", "mean", "mean_error"),
    _moment("Std Dev", "std", "std_error"),
    _single("Underflow", lambda h: float(h.values(flow=True)[0])),
    _single("Overflow", lambda h: float(h.values(flow=True)[-1])),
    _single("Integral", lambda h: h.integral(), any_axes=True),
    _single("Skewness", lambda h: h.skewness()),
    _single("Kurtosis", lambda h: h.kurtosis()),
)


def _digits(option: int, count: int) -> list[int]:
    """``fOptStat`` a decimal digit at a time, the lowest first."""
    return [(option // 10**place) % 10 for place in range(count)]


def stats_rows(h: Any, option: int) -> list[tuple[str, str]]:
    """The lines ``fOptStat`` asks for, as names and values under the histogram's name."""
    name, *digits = _digits(option, 1 + len(STAT_LINES))
    rows: list[tuple[str, str]] = [(h.name, "")] if name else []
    for digit, lines in zip(digits, STAT_LINES):
        if digit:
            rows += lines(h, digit)
    return rows


def fit_rows(h: Any, option: int) -> list[tuple[str, str]]:
    """What ``fOptFit`` asks to be said of the fit hung on the histogram, if there is one."""
    fitted = [one for one in h.functions if isinstance(one, Function)]
    if not option or not fitted:
        return []
    function = fitted[0]
    parameters, probability, chi2 = _digits(option, 3)
    rows = _goodness(function.fit_result or {}, chi2, probability)
    if parameters:
        for label, value, error in zip(
            function.parameter_names, function.parameters, function.parameter_errors
        ):
            rows.append((label, f"{_value(value)} #pm {_value(error)}"))
    return rows


def _goodness(result: dict[str, Any], chi2: int, probability: int) -> list[tuple[str, str]]:
    """The fit's chi-square over its degrees of freedom, and its probability, as asked."""
    if "chi2" not in result:
        return []
    ndf = int(result.get("ndf", 0))
    rows = [("#chi^{2} / ndf", f"{_value(result['chi2'])} / {ndf}")] if chi2 else []
    if probability:
        rows.append(("Prob", _value(prob(result["chi2"], ndf))))
    return rows


def shows_stats(h: Any, option: str) -> bool:
    """Whether drawing ``h`` with ``option`` makes a stats box, as ``gStyle`` would."""
    if int(lookup(h, "fBits", 0)) & NO_STATS:
        return False
    upper = option.upper()
    return "SAMES" in upper or "SAME" not in upper


def default_stats(scene: Scene, h: Any) -> None:
    """A stats box for a histogram saved with none, placed and filled from ``gStyle``."""
    rows = stats_rows(h, OPT_STAT) + fit_rows(h, OPT_FIT)
    top = STAT_Y - scene.stats * (STAT_LINE * len(rows) + 0.01)
    corners = {
        "fX1NDC": STAT_X - STAT_W,
        "fY1NDC": top - STAT_LINE * len(rows),
        "fX2NDC": STAT_X,
        "fY2NDC": top,
        "fOption": "brNDC",
        "fBorderSize": 1,
        "fFillColor": 0,
        "fFillStyle": 1001,
        "fLineColor": 1,
        "fTextFont": 42,
        "fTextSize": 0.0,
    }
    box = Primitive("TPaveStats", corners)
    columns(scene, box, rows, draw_box(scene, box))
    scene.stats += 1
