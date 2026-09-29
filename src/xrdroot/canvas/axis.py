"""``TGaxis::PaintAxis``: an axis's line, ticks, labels and title, as ROOT places them.

A histogram's frame is dressed by ``THistPainter::PaintAxis``, which paints
each axis with a ``TGaxis`` - and so does a ``TGaxis`` drawn on its own.
Everything is placed in the pad's NDC: the axis from ``(x0, y0)`` to
``(x1, y1)``, its ticks a fraction of its own length long (``fTickSize`` of
it, half that for a secondary tick, a quarter for a tertiary), its labels
``fLabelOffset`` of the pad beyond them, and its title ``1.6`` times its size
times ``fTitleOffset`` further. The divisions are ``THLimitsFinder::Optimize``'s
round numbers, the labels printed in the fewest digits that tell them
apart, with a ``#times10^{n}`` of the whole axis when they would be too long, and a
minus sign ``#minus``. A logarithmic axis has a label per decade - ``1``,
``10``, ``10^{2}`` - and ticks at each whole multiple.

This works out what ROOT draws, as :class:`Painted` segments and labels;
:mod:`.frame` and :mod:`.gaxis` draw them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ..limits import optimize

__all__ = ["Axis", "Label", "Painted", "paint_axis"]

#: ``gStyle``'s ``fAxisMaxDigits``: the longest a label may be before it is scaled.
MAX_DIGITS = 5
#: How close two values are to be the same, as ``PaintAxis`` compares them.
EPSILON = 1e-5


@dataclass
class Axis:
    """What ``PaintAxis`` is given: where the axis is, its range, and its attributes."""

    x0: float
    y0: float
    x1: float
    y1: float
    wmin: float
    wmax: float
    ndiv: int = 510
    chopt: str = ""
    grid_length: float = 0.0
    label_font: int = 42
    label_size: float = 0.035
    label_color: int = 1
    label_offset: float = 0.005
    tick_size: float = 0.03
    title: str = ""
    title_offset: float = 1.0
    title_size: float = 0.035
    title_font: int = 42
    title_color: int = 1
    line_color: int = 1
    #: ``TAxis`` bits: a centred or rotated title, no exponent, more log labels, decimals.
    bits: frozenset[str] = frozenset()
    #: The pad's size in pixels, and ``gPad->GetWw() * GetWNDC()``.
    pad: tuple[float, float] = (1.0, 1.0)
    #: ``TGaxis::ChangeLabel``'s changes: each ``(number, angle, size, align, color, font,
    #: text)``, the number counted from one - or back from the last label, if negative - and
    #: each attribute left as it is where it is negative, or the text where it is empty.
    changed: tuple[tuple[Any, ...], ...] = ()


class Label(NamedTuple):
    """One string an axis writes, at a point of the pad's NDC."""

    text: str
    u: float
    v: float
    align: int
    angle: float
    font: int
    size: float
    color: int


@dataclass
class Painted:
    """Everything an axis paints: segments in NDC, of the axis and of its grid, and its text."""

    lines: list[tuple[float, float, float, float]] = field(default_factory=list)
    grid: list[tuple[float, float, float, float]] = field(default_factory=list)
    labels: list[Label] = field(default_factory=list)


def _divisions(ndiv: int) -> tuple[int, int, int]:
    """``fNdivisions`` as its primary, secondary and tertiary divisions."""
    n1a = ndiv % 100
    n2a = (ndiv % 10000 - n1a) // 100
    n3a = (ndiv % 1000000 - n2a - n1a) // 10000
    return n1a, n2a, n3a


class _Geometry(NamedTuple):
    """The axis's direction and length in NDC, and which side its ticks and labels are on."""

    cos: float
    sin: float
    length: float
    mside: int
    lside: int
    phil: float


def _snapped(value: float) -> float:
    """A cosine or sine as ``PaintAxis`` keeps it: nothing, when it is all but nothing."""
    return 0.0 if abs(value) <= EPSILON else value


def _pixel_angle(axis: Axis, pixel: Any) -> float:
    """``phil``: the axis's angle measured in whole pixels, which its title is turned by."""
    (px0, py0), (px1, py1) = pixel(axis.x0, axis.y0), pixel(axis.x1, axis.y1)
    if axis.x0 < axis.x1:
        return math.atan2(py0 - py1, px1 - px0)
    return math.atan2(py1 - py0, px0 - px1)


def _direction(axis: Axis, pixel: Any) -> tuple[float, float, float]:
    """The axis's cosine and sine in NDC, and its angle in pixels.

    An upright axis is straight up or down, with no pixels to measure it by.
    """
    x0, y0, x1, y1 = axis.x0, axis.y0, axis.x1, axis.y1
    if x0 == x1:
        phi = 0.5 * math.pi if y1 >= y0 else 1.5 * math.pi
        phil = phi
    else:
        phi = math.atan2(y1 - y0, x1 - x0)
        phil = _pixel_angle(axis, pixel)
    return _snapped(math.cos(phi)), _snapped(math.sin(phi)), phil


def _sides(axis: Axis, options: str) -> tuple[int, int]:
    """``mside`` and ``lside``: which side of the axis its ticks and its labels go.

    ``+`` and ``-`` put the ticks above or below, both of them either side, and
    ``=`` puts the labels on the ticks' side rather than the other; an upright
    axis going up has its ticks on its left unless told.
    """
    plus, minus = "+" in options, "-" in options
    if plus and minus:
        return 0, (1 if "=" in options else -1)
    upright = -1 if axis.x0 == axis.x1 and axis.y1 > axis.y0 else 1
    mside = 1 if plus else (-1 if minus else upright)
    return mside, (mside if "=" in options else -mside)


def _geometry(axis: Axis, options: str, pixel: Any) -> _Geometry:
    """Where the axis points and which sides it is dressed on, which all the rest is placed by."""
    cos, sin, phil = _direction(axis, pixel)
    mside, lside = _sides(axis, options)
    length = math.hypot(axis.x1 - axis.x0, axis.y1 - axis.y0)
    return _Geometry(cos, sin, length, mside, lside, phil)


def _rotate(x: float, y: float, geo: _Geometry, origin: tuple[float, float]) -> tuple[float, float]:
    """``TGaxis::Rotate``: along the axis and across it, into the pad's NDC."""
    return geo.cos * x - geo.sin * y + origin[0], geo.sin * x + geo.cos * y + origin[1]


@dataclass
class _Binning:
    """The divisions ``PaintAxis`` settles on, and where the optimised axis starts and ends."""

    noopt: bool
    wmin: float
    wmax: float
    n1a: int
    nn2: int
    nn3: int
    nticks: int
    start: tuple[float, float] = (0.0, 0.0)
    end: tuple[float, float] = (0.0, 0.0)
    before: float = 0.0
    after: float = 0.0
    length: float = 0.0


def _unoptimised(axis: Axis, options: str, log: bool, n1a: int) -> bool:
    """Whether the divisions are taken as they are: asked not to (``N``), or nothing to divide."""
    return "N" in options or axis.wmin == axis.wmax or axis.ndiv == 0 or n1a <= 1 or log


def _trimmed(axis: Axis, n1a: int) -> tuple[float, float, int, float]:
    """``THLimitsFinder::Optimize``'s round divisions, less any that fall off the axis's ends."""
    low, high, nbins, width = optimize(axis.wmin, axis.wmax, n1a)
    if axis.wmin - low > EPSILON:
        low, nbins = low + width, nbins - 1
    if high - axis.wmax > EPSILON:
        high, nbins = high - width, nbins - 1
    return low, high, nbins, width


def _subdivisions(low: float, width: float, n2a: int, n3a: int) -> tuple[int, int]:
    """``nn2`` and ``nn3``: the ticks per primary and per secondary division, themselves rounded.

    The secondaries are optimised within the first primary division, and the
    tertiaries within the first secondary, as ``PaintAxis`` does.
    """
    nb2, low2, width2 = n2a, low, 0.0
    if n2a > 1 and width > 0:
        low2, _high2, nb2, width2 = optimize(low, low + width, n2a)
    nb3 = n3a
    if n3a > 1 and width2 > 0:
        nb3 = optimize(low2, low2 + width2, n3a)[2]
    nn3 = max(nb3, 1)
    return max(nb2, 1) * nn3, nn3


def _measured(found: _Binning, axis: Axis) -> _Binning:
    """The optimised axis's length, and how far short of each end of the whole it stops."""
    start, end = found.start, found.end
    found.length = math.hypot(end[0] - start[0], end[1] - start[1])
    found.before = math.hypot(start[0] - axis.x0, start[1] - axis.y0)
    found.after = math.hypot(axis.x1 - end[0], axis.y1 - end[1])
    return found


def _optimised(axis: Axis, options: str, log: bool) -> _Binning:
    """``PaintAxis``'s optimisation: round primary divisions, and secondaries within them."""
    n1a, n2a, n3a = _divisions(axis.ndiv)
    nn3 = max(n3a, 1)
    nn2 = max(n2a, 1) * nn3
    plain = _Binning(True, axis.wmin, axis.wmax, n1a, nn2, nn3, max(n1a, 1) * nn2 + 1)
    if _unoptimised(axis, options, log, n1a):
        return plain
    low, high, nbins, width = _trimmed(axis, n1a)
    start, end = _optimised_ends(axis, low, high)
    nn2, nn3 = _subdivisions(low, width, n2a, n3a)
    found = _Binning(False, low, high, nbins, nn2, nn3, max(nbins, 1) * nn2 + 1, start, end)
    if _measured(found, axis).length < EPSILON:
        return _Binning(True, axis.wmin, axis.wmax, n1a, nn2, nn3, plain.nticks)
    return found


def _optimised_ends(
    axis: Axis, low: float, high: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the first and last round division fall along the axis, in NDC."""
    span = axis.wmax - axis.wmin
    if axis.x1 == axis.x0:
        slope = (axis.y1 - axis.y0) / span
        return (
            (axis.x0, slope * (low - axis.wmin) + axis.y0),
            (axis.x1, slope * (high - axis.wmin) + axis.y0),
        )
    slope = (axis.x1 - axis.x0) / span
    xs = (slope * (low - axis.wmin) + axis.x0, slope * (high - axis.wmin) + axis.x0)
    if axis.y1 == axis.y0:
        return (xs[0], axis.y0), (xs[1], axis.y1)
    alfa = (axis.y1 - axis.y0) / (axis.x1 - axis.x0)
    beta = (axis.y0 * axis.x1 - axis.y1 * axis.x0) / (axis.x1 - axis.x0)
    return (xs[0], alfa * xs[0] + beta), (xs[1], alfa * xs[1] + beta)


def _tick_length(axis: Axis, geo: _Geometry, options: str) -> float:
    """A primary tick's length in NDC, signed by its side: ``fTickSize`` of the axis if ``S``."""
    size = axis.tick_size if "S" in options else 0.03
    return (1 if geo.mside >= 0 else -1) * geo.length * size


def _grid_side(axis: Axis) -> int:
    """Which way the grid runs from the axis: back across the frame from an upright one going up."""
    return -1 if axis.x0 == axis.x1 and axis.y1 > axis.y0 else 1


def _span(axis: Axis, geo: _Geometry, binning: _Binning) -> tuple[tuple[float, float], float]:
    """Where the divisions start and how long they run: the whole axis, or its optimised part."""
    if binning.noopt:
        return (axis.x0, axis.y0), geo.length
    return binning.start, binning.length


def _beyond(distance: float, step: float) -> int:
    """How many whole steps fit in the stretch the optimisation left off an end, at most 1000."""
    return min(int(distance / step + EPSILON), 1000) if distance else 0


def _tick_places(binning: _Binning, step: float) -> list[tuple[int, float]]:
    """Each tick's index and how far along it is: the divisions, then those before and after."""
    before, after = _beyond(binning.before, step), _beyond(binning.after, step)
    places = [(k, k * step) for k in range(binning.nticks)]
    places += [(k, -k * step) for k in range(1, before + 1)]
    places += [(k, (binning.nticks - 1 + k) * step) for k in range(1, after + 1)]
    return places


def _level(index: int, binning: _Binning) -> int:
    """Whether tick ``index`` is primary (0), secondary (1) or tertiary (2)."""
    if index % binning.nn2 == 0:
        return 0
    return 1 if index % binning.nn3 == 0 else 2


def _ticks(axis: Axis, geo: _Geometry, binning: _Binning, options: str, out: Painted) -> None:
    """The linear axis's ticks, and its grid at each primary one."""
    first = _tick_length(axis, geo, options)
    ticks = (first, first / 2, first / 4)
    origin, length = _span(axis, geo, binning)
    step = length / (binning.nticks - 1)
    side = _grid_side(axis)
    for index, along in _tick_places(binning, step):
        level = _level(index, binning)
        _tick(out, geo, origin, along, ticks[level], options, side, axis.grid_length, level == 0)


# -- the numbers ----------------------------------------------------------------------------


class _Format(NamedTuple):
    """How the labels are printed: the format, the first value and step, and a ``#times10^n``."""

    format: str
    first: float
    step: float
    exponent: int


def _scaled(ww: float, first: float, step: float, up: bool) -> tuple[int, float, float, float]:
    """The labels divided (``up``) or multiplied by ten until they are short.

    The exponent goes by powers of a thousand, so it reads as the engineers' units do.
    """
    exponent = 0
    small = 1 / 10 ** (MAX_DIGITS - 2)
    while True:
        exponent += 1 if up else -1
        factor = 0.1 if up else 10.0
        ww, first, step = ww * factor, first * factor, step * factor
        if exponent % 3 == 0 and (ww <= 10 ** (MAX_DIGITS - 1) if up else ww >= small):
            return exponent, ww, first, step


def _tiny(ww: float) -> int | None:
    """For steps under the fewest digits' last: the power of a thousand the labels are shown times.

    ``None`` when the labels themselves are not small, and so need no exponent.
    """
    af = math.log10(ww) + EPSILON
    if af >= 0:
        return None
    iexe = abs(int(af))
    return iexe + {1: 2, 2: 1}.get(iexe % 3, 0)


def _finished(
    if1: int, if2: int, first: float, step: float, exponent: int, negative: bool = False
) -> _Format:
    """The format of ``if1`` digits, ``if2`` after the point, widened until the step shows."""
    if1 = min(if1 + (1 if negative else 0), 32)
    while step < 10.0 ** (-if2):
        if1, if2 = if1 + 1, if2 + 1
    if1, if2 = max(min(if1, 14), 0), min(if2, 14)
    return _Format(f"%{if1}.{if2}f" if if2 > 0 else f"%{if1 + 1}.1f", first, step, exponent)


def _tiny_format(
    wmin: float, wmax: float, n1a: int, ww: float, negative: bool
) -> _Format | None:
    """The format of steps too small for ``fAxisMaxDigits`` digits, times a power of a thousand.

    ``None`` when the steps are not that small, or the labels are not, and the
    ordinary format will do.
    """
    if abs(wmax - wmin) / n1a >= 10.0 ** (-MAX_DIGITS):
        return None
    iexe = _tiny(ww)
    if iexe is None:
        return None
    first, step = wmin * 10**iexe, (wmax - wmin) / n1a * 10**iexe
    return _finished(MAX_DIGITS, MAX_DIGITS - 2, first, step, -iexe, negative)


def _figures(ww: float) -> int:
    """``nf``: how many figures the largest label has before its point, or less its zeros after."""
    af = (math.log10(ww) if ww >= 1 else math.log10(ww * 0.0001)) + EPSILON
    return int(af) + 1


def _decimals(ww: float, wmin: float, wmax: float, n1a: int) -> int:
    """``na``: the figures after the point, more of them the smaller the labels and their steps."""
    na = max((MAX_DIGITS - i for i in range(MAX_DIGITS - 1, 0, -1) if abs(ww) < 10**i), default=0)
    ndyn = n1a
    while ndyn and abs((wmax - wmin) / ndyn) <= 0.999 and na < MAX_DIGITS - 2:
        na, ndyn = na + 1, ndyn // 10
    return na


def _label_format(wmin: float, wmax: float, n1a: int, no_exponent: bool) -> _Format:
    """``PaintAxis``'s choice of format for the labels of a linear axis."""
    first, step = wmin, (wmax - wmin) / n1a
    ww = max(abs(wmin), abs(wmax)) or 1.0
    negative = min(wmin, wmax) < 0
    tiny = None if no_exponent else _tiny_format(wmin, wmax, n1a, ww, negative)
    if tiny is not None:
        return tiny
    nf = _figures(ww)
    exponent = 0
    if not no_exponent and (nf > MAX_DIGITS or nf < -MAX_DIGITS):
        exponent, ww, first, step = _scaled(ww, first, step, nf > MAX_DIGITS)
    na = _decimals(ww, wmin, wmax, n1a)
    return _finished(max(nf + na, MAX_DIGITS) + 1, na, first, step, exponent, negative)


def _printed(value: float, form: str, dot: bool = False) -> str:
    """One label as ``PaintAxis`` prints it: the format, without its blanks and trailing zeros."""
    label = (form % value)[:28]
    first = next((i for i, char in enumerate(label) if char in "1234567890-+."), len(label))
    label = label[first:]
    label = label.rstrip("0") if "." in label else label
    if label.endswith(".") and not dot:
        label = label[:-1]
    if label == "-0":
        label = "0"
    return label or " "


class _Text(NamedTuple):
    """How the labels are written: their height in NDC, size, alignment and offset from the axis."""

    height: float
    size: float
    align: int
    offset: float


def _label_text(axis: Axis, geo: _Geometry, options: str, log: bool) -> _Text:
    """The labels' size, their alignment by the axis's direction, and how far off it they sit."""
    height = axis.label_size
    if axis.label_font % 10 > 2:
        height /= axis.pad[1]
    across = 3 if axis.x0 == axis.x1 else 2
    up = 2 if axis.y0 != axis.y1 else 1
    across = {"C": 2, "R": 3, "L": 1}.get(next((c for c in "LRC" if c in options), ""), across)
    if abs(geo.cos) > 0.9:
        across = 2
    elif geo.cos * geo.sin:
        across = 1 if geo.cos * geo.sin > 0 else 3
    offset = _label_offset(axis, geo, options, log, height)
    return _Text(height, axis.label_size, 10 * across + up, offset)


def _upright_offset(offset: float, tick: float, geo: _Geometry, options: str) -> float:
    """An upright axis's ``ylabel``: past its ticks on their side, or off the other."""
    if "+" in options and "-" not in options:
        return offset / 2 + tick if "=" in options else -offset
    return offset + (tick if geo.lside < 0 else 0.0)


def _level_offset(
    offset: float, tick: float, geo: _Geometry, options: str, log: bool, height: float
) -> float:
    """A level axis's ``ylabel``: below it, past its ticks if they hang down, or above it.

    A logarithmic axis's labels, with their raised exponents, sit half a label lower.
    """
    if "-" in options and "+" not in options:
        found = offset + 0.5 * height + abs(tick)
    else:
        found = -offset - (abs(tick) if geo.mside <= 0 else 0.0)
    return found - (0.5 * height if log else 0.0)


def _label_offset(axis: Axis, geo: _Geometry, options: str, log: bool, height: float) -> float:
    """``ylabel``: how far across the axis its labels are put, in NDC."""
    tick = _tick_length(axis, geo, options)
    offset = axis.label_offset
    if axis.x0 == axis.x1:
        return _upright_offset(offset, tick, geo, options)
    if axis.y0 == axis.y1:
        return _level_offset(offset, tick, geo, options, log, height)
    return offset if geo.mside + geo.lside >= 0 else -offset


def _exponent_label(axis: Axis, geo: _Geometry, text: _Text, exponent: int) -> Label:
    """The ``#times10^{n}`` the labels are all shown times, just past the axis's far end.

    It is written upright in the labels' font, made precise-size if it is not.
    """
    along = (geo.length if axis.x0 != axis.x1 else axis.y1 - axis.y0) + 0.1 * text.height
    u, v = _rotate(along, 0.0, geo, (axis.x0, axis.y0))
    font = axis.label_font if axis.label_font % 10 >= 2 else axis.label_font // 10 * 10 + 2
    shown = f"#times10^{{{exponent}}}".replace("-", "#minus")
    return Label(shown, u, v, 11, 0.0, font, text.size, axis.label_color)


def _changed(axis: Axis, label: Label, number: int, nlabels: int) -> Label:
    """``label``, the ``number``-th, as ``TGaxis::ChangeLabelAttributes`` restyles it.

    A change numbered back from the end, ``-1`` the last, is of the label
    numbered ``change + 2 + nlabels``, as ``FindModLab`` matches it; a size of
    zero erases the label. The first change that matches is the one made.
    """
    backwards = number - 2 - nlabels
    found = next((c for c in axis.changed if c[0] in (number, min(backwards, 0))), None)
    if found is None:
        return label
    return label._replace(**_restyled(found[1:]))


def _restyled(change: tuple[Any, ...]) -> dict[str, Any]:
    """What a change sets: each attribute not left at ``-1``, and the text if it has one."""
    angle, size, align, color, font, text = change
    given = {
        "angle": (angle, angle >= 0), "size": (size, size >= 0), "align": (align, align > 0),
        "color": (color, color >= 0), "font": (font, font > 0), "text": (text, bool(text)),
    }  # fmt: skip
    return {name: value for name, (value, used) in given.items() if used}


def _written(out: Painted, label: Label) -> None:
    """A label written, its minus signs ROOT's ``#minus`` - unless erased, at size zero."""
    if label.size:
        out.labels.append(label._replace(text=label.text.replace("-", "#minus")))


def _linear_labels(
    axis: Axis, geo: _Geometry, binning: _Binning, text: _Text, out: Painted, options: str
) -> None:
    """The labels of a linear axis, one per primary division, and its ``#times10^n`` if any."""
    if not binning.n1a:
        return
    form = _label_format(binning.wmin, binning.wmax, binning.n1a, "noexponent" in axis.bits)
    origin, length = _span(axis, geo, binning)
    step = length / binning.n1a
    centred = "M" in options or "centerlabels" in axis.bits
    shift = 0.5 * step if centred else 0.0
    drop = 0.80 * text.height if axis.y0 == axis.y1 else 0.0
    value = form.first
    nlabels = binning.n1a - int(centred)
    for k in range(nlabels + 1):
        label = _printed(value, form.format, "." in options)
        value += form.step
        u, v = _rotate(step * k + shift, text.offset, geo, origin)
        made = Label(label, u, v - drop, text.align, 0.0, axis.label_font, text.size,
                     axis.label_color)  # fmt: skip
        _written(out, _changed(axis, made, k + 1, nlabels))
    if form.exponent:
        out.labels.append(_exponent_label(axis, geo, text, form.exponent))


# -- logarithmic axes -------------------------------------------------------------------------


class _Decades(NamedTuple):
    """A logarithmic axis's decades: the first, how many, NDC per decade, and the first label."""

    low: float
    first: int
    total: int
    scale: float
    number: int


def _decades(axis: Axis, geo: _Geometry) -> _Decades:
    """The decades a logarithmic axis spans, its ends nudged a millionth outward as ROOT does.

    The nudge keeps an end that is a whole decade from being lost to rounding.
    """
    low = math.log10(axis.wmin)
    low += 1e-6 if low > 0 else -1e-6
    high = math.log10(axis.wmax)
    edge = high + (1e-6 if high > 0 else -1e-6)
    first, last = int(low), 1 + int(edge)
    number = first + (1 if low > 0 and low - first > 0 else 0)
    return _Decades(low, first, last - first + 1, geo.length / (high - low), number)


def _log_label(
    axis: Axis, geo: _Geometry, text: _Text, number: int, at: tuple[float, float]
) -> Label:
    """One decade's label: ``1``, ``10``, ``10^{n}``, or the number itself without exponents."""
    u, v = at
    if axis.x0 == axis.x1:
        u += 0.25 * text.height
        if geo.lside < 0:
            u += (1 if number == 0 else 2) * text.height if geo.mside < 0 else 0.25 * text.height
    no_exponent = "noexponent" in axis.bits
    if axis.y0 == axis.y1 and no_exponent:
        v += 0.33 * text.height
    if no_exponent:
        shown = f"{10.0**number:f}".rstrip("0").rstrip(".")
    else:
        shown = {0: "1", 1: "10"}.get(number, f"10^{{{number}}}")
    return Label(shown.replace("-", "#minus"), u, v, text.align, 0.0,
                 axis.label_font, text.size, axis.label_color)  # fmt: skip


def _labelled(axis: Axis, options: str) -> bool:
    """Whether the axis is labelled: not when told ``U``, nor when its labels are off the pad."""
    return "U" not in options and axis.label_offset <= 1.1


def _log_shift(axis: Axis) -> float:
    """How many label heights a logarithmic axis's labels move off it, beyond ``ylabel``."""
    if axis.x0 == axis.x1:
        return 0.33
    return -0.65 if axis.y0 == axis.y1 else 0.0


@dataclass
class _LogWalk:
    """``PaintAxis``'s walk along a logarithmic axis, decade by decade.

    It keeps what the walk carries from one decade to the next - how far off the
    axis the labels are, and the number of the next one - so each step can be
    taken on its own.
    """

    axis: Axis
    geo: _Geometry
    text: _Text
    options: str
    out: Painted
    decades: _Decades
    first: float
    offset: float
    number: int

    def decade(self, j: int) -> bool:
        """Decade ``j``'s long tick, label and short ticks; whether the axis goes on past it."""
        decade = self.decades.first - 2 + j
        if j == 1:
            self.offset += self.text.height * _log_shift(self.axis)
        along = self.decades.scale * (decade - self.decades.low)
        if along >= 0 and not self._primary(j, along):
            return False
        n1a = self.axis.ndiv % 100
        return _multiples(self.axis, self.geo, self.decades, decade, self.first / 2,
                          self.options, self.out, n1a)  # fmt: skip

    def _primary(self, j: int, along: float) -> bool:
        """The decade's own tick and label, at ``along``; whether the axis goes on past it."""
        axis, geo, origin = self.axis, self.geo, (self.axis.x0, self.axis.y0)
        if along - geo.length > EPSILON:
            return False
        _tick(self.out, geo, origin, along, self.first, self.options, _grid_side(axis),
              axis.grid_length)  # fmt: skip
        if not _labelled(axis, self.options):
            return True
        n1a = axis.ndiv % 100
        if not n1a:
            return False
        if _shows(self.decades.total, n1a, j):
            at = _rotate(along, self.offset, geo, origin)
            self.out.labels.append(_log_label(axis, geo, self.text, self.number, at))
        self.number += 1
        return True


def _log_ticks(axis: Axis, geo: _Geometry, text: _Text, options: str, out: Painted) -> None:
    """A logarithmic axis: a long tick and a label at each decade, short ones at its multiples."""
    if axis.wmin <= 0 or axis.wmax <= 0 or axis.wmin == axis.wmax:
        return
    decades = _decades(axis, geo)
    first = _tick_length(axis, geo, options)
    walk = _LogWalk(axis, geo, text, options, out, decades, first, text.offset, decades.number)
    for j in range(1, decades.total + 1):
        if not walk.decade(j):
            return


def _shows(count: int, n1a: int, j: int) -> bool:
    """Whether decade ``j`` of ``count`` is labelled, as ``PaintAxis`` thins the labels out."""
    every = count // n1a or 1000000
    return count <= n1a or j == 1 or j == count or (count > n1a and j % every == 0)


def _tick(out: Painted, geo: _Geometry, origin: tuple[float, float], along: float, length: float,
          options: str, side: int, grid: float, gridded: bool = True) -> None:  # fmt: skip
    """One tick ``length`` long at ``along``, and its grid line if the axis has a grid."""
    across = 0.0 if geo.mside else -length
    out.lines.append((*_rotate(along, length, geo, origin), *_rotate(along, across, geo, origin)))
    if gridded and "W" in options:
        far, near = _rotate(along, side * grid, geo, origin), _rotate(along, 0.0, geo, origin)
        out.grid.append((*far, *near))


def _multiples(axis: Axis, geo: _Geometry, decades: _Decades, decade: int, length: float,
               options: str, out: Painted, n1a: int) -> bool:  # fmt: skip
    """The ticks at 2 to 9 times a decade, all of them or only 5 when there are many decades.

    Returns whether the axis goes on past them.
    """
    origin, side = (axis.x0, axis.y0), _grid_side(axis)
    for k in range(2, 10):
        along = decades.scale * (math.log10(k) + decade - decades.low)
        if along < 0:
            continue
        if along > geo.length:
            return False
        if decades.total <= n1a * 2 or k == 5:
            gridded = decades.total <= 5 and axis.ndiv > 100
            _tick(out, geo, origin, along, length, options, side, axis.grid_length, gridded)
    return True


# -- the title ---------------------------------------------------------------------------------


def _title_height(axis: Axis) -> float:
    """The title's height in NDC: a precise size in pixels is a fraction of the pad across it."""
    height = axis.title_size
    if axis.title_font % 10 > 2:
        height /= axis.pad[0] if axis.x1 == axis.x0 else axis.pad[1]
    return height


def _title_turn(axis: Axis, geo: _Geometry, centre: bool) -> tuple[int, float]:
    """The title's alignment and angle: along the axis, or turned about (``rotatetitle``).

    A title at the far end is aligned by that end, which a turn or a reversed
    axis puts on its other side.
    """
    forward = axis.x1 >= axis.x0
    if "rotatetitle" in axis.bits:
        return (22 if centre else (12 if forward else 32)), geo.phil + math.pi
    return (22 if centre else (32 if forward else 12)), geo.phil


def _title(axis: Axis, geo: _Geometry, out: Painted) -> None:
    """The axis's title, beyond its labels, at its far end or its middle, turned along it."""
    height = _title_height(axis)
    offset = axis.title_offset or 1.0
    widen = 1.6 if axis.x1 == axis.x0 or axis.y1 == axis.y0 else 1.3
    across = geo.lside * widen * height * offset
    centre = "centertitle" in axis.bits
    along = 0.5 * geo.length if centre else geo.length
    align, angle = _title_turn(axis, geo, centre)
    u, v = _rotate(along, across, geo, (axis.x0, axis.y0))
    out.labels.append(Label(axis.title, u, v, align, math.degrees(angle),
                            axis.title_font, axis.title_size, axis.title_color))  # fmt: skip


def paint_axis(axis: Axis, pixel: Any) -> Painted:
    """Everything ``TGaxis::PaintAxis`` paints for ``axis``.

    ``pixel`` turns a point of NDC into whole pixels, which the title's angle is measured in.
    """
    options = axis.chopt
    log = "G" in options
    geo = _geometry(axis, options, pixel)
    out = Painted()
    if "B" not in options:
        out.lines.append((axis.x0, axis.y0, axis.x1, axis.y1))
    if axis.ndiv == 0 or axis.wmin == axis.wmax:
        return out
    text = _label_text(axis, geo, options, log)
    if log:
        _log_ticks(axis, geo, text, options, out)
    else:
        binning = _optimised(axis, options, log)
        _ticks(axis, geo, binning, options, out)
        if _labelled(axis, options):
            _linear_labels(axis, geo, binning, text, out, options)
    if axis.title:
        _title(axis, geo, out)
    return out
