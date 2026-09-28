"""``TGaxis::PaintAxis``: an axis's line, ticks, labels and title, as ROOT places them.

A histogram's frame is dressed by ``THistPainter::PaintAxis``, which paints
each axis with a ``TGaxis`` - and so does a ``TGaxis`` drawn on its own.
Everything is placed in the pad's NDC: the axis from ``(x0, y0)`` to
``(x1, y1)``, its ticks a fraction of its own length long (``fTickSize`` of
it, half that for a secondary tick, a quarter for a tertiary), its labels
``fLabelOffset`` of the pad beyond them, and its title ``1.6`` times its size
times ``fTitleOffset`` further. The divisions are ``THLimitsFinder::Optimize``'s
round numbers, the labels printed in the fewest digits that tell them
apart, with a ``×10^n`` of the whole axis when they would be too long, and a
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


def _geometry(axis: Axis, options: str, pixel: Any) -> _Geometry:
    x0, y0, x1, y1 = axis.x0, axis.y0, axis.x1, axis.y1
    if x0 == x1:
        phi = 0.5 * math.pi if y1 >= y0 else 1.5 * math.pi
        phil = phi
    else:
        phi = math.atan2(y1 - y0, x1 - x0)
        (px0, py0), (px1, py1) = pixel(x0, y0), pixel(x1, y1)
        phil = math.atan2(py0 - py1, px1 - px0) if x0 < x1 else math.atan2(py1 - py0, px0 - px1)
    cos, sin = math.cos(phi), math.sin(phi)
    cos, sin = (0.0 if abs(cos) <= EPSILON else cos), (0.0 if abs(sin) <= EPSILON else sin)
    plus, minus = "+" in options, "-" in options
    mside = -1 if x0 == x1 and y1 > y0 else 1
    mside = 0 if plus and minus else (1 if plus else (-1 if minus else mside))
    lside = -mside
    if "=" in options:
        lside = mside
    if plus and minus:
        lside = 1 if "=" in options else -1
    length = math.hypot(x1 - x0, y1 - y0)
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


def _optimised(axis: Axis, options: str, log: bool) -> _Binning:
    """``PaintAxis``'s optimisation: round primary divisions, and secondaries within them."""
    n1a, n2a, n3a = _divisions(axis.ndiv)
    nn3 = max(n3a, 1)
    nn2 = max(n2a, 1) * nn3
    plain = _Binning(True, axis.wmin, axis.wmax, n1a, nn2, nn3, max(n1a, 1) * nn2 + 1)
    if "N" in options or axis.wmin == axis.wmax or axis.ndiv == 0 or n1a <= 1 or log:
        return plain
    low, high, nbins, width = optimize(axis.wmin, axis.wmax, n1a)
    if axis.wmin - low > EPSILON:
        low, nbins = low + width, nbins - 1
    if high - axis.wmax > EPSILON:
        high, nbins = high - width, nbins - 1
    start, end = _optimised_ends(axis, low, high)
    nb2, low2, width2 = n2a, low, 0.0
    if n2a > 1 and width > 0:
        low2, _high2, nb2, width2 = optimize(low, low + width, n2a)
    nb3 = n3a
    if n3a > 1 and width2 > 0:
        nb3 = optimize(low2, low2 + width2, n3a)[2]
    nn3 = max(nb3, 1)
    nn2 = max(nb2, 1) * nn3
    found = _Binning(False, low, high, nbins, nn2, nn3, max(nbins, 1) * nn2 + 1, start, end)
    found.length = math.hypot(end[0] - start[0], end[1] - start[1])
    found.before = math.hypot(start[0] - axis.x0, start[1] - axis.y0)
    found.after = math.hypot(axis.x1 - end[0], axis.y1 - end[1])
    if found.length < EPSILON:
        return _Binning(True, axis.wmin, axis.wmax, n1a, nn2, nn3, plain.nticks)
    return found


def _optimised_ends(axis: Axis, low: float, high: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """Where the first and last round division fall along the axis, in NDC."""
    span = axis.wmax - axis.wmin
    if axis.x1 == axis.x0:
        slope = (axis.y1 - axis.y0) / span
        return (axis.x0, slope * (low - axis.wmin) + axis.y0), (axis.x1, slope * (high - axis.wmin) + axis.y0)
    slope = (axis.x1 - axis.x0) / span
    xs = (slope * (low - axis.wmin) + axis.x0, slope * (high - axis.wmin) + axis.x0)
    if axis.y1 == axis.y0:
        return (xs[0], axis.y0), (xs[1], axis.y1)
    alfa = (axis.y1 - axis.y0) / (axis.x1 - axis.x0)
    beta = (axis.y0 * axis.x1 - axis.y1 * axis.x0) / (axis.x1 - axis.x0)
    return (xs[0], alfa * xs[0] + beta), (xs[1], alfa * xs[1] + beta)


def _ticks(axis: Axis, geo: _Geometry, binning: _Binning, options: str, out: Painted) -> None:
    """The linear axis's ticks, and its grid at each primary one."""
    size = axis.tick_size if "S" in options else 0.03
    first = (1 if geo.mside >= 0 else -1) * geo.length * size
    ticks = (first, first / 2, first / 4)
    origin = (axis.x0, axis.y0) if binning.noopt else binning.start
    length = geo.length if binning.noopt else binning.length
    step = length / (binning.nticks - 1)
    before = min(int(binning.before / step + EPSILON), 1000) if binning.before else 0
    after = min(int(binning.after / step + EPSILON), 1000) if binning.after else 0
    places = [(k, k * step) for k in range(binning.nticks)]
    places += [(k, -k * step) for k in range(1, before + 1)]
    places += [(k, (binning.nticks - 1 + k) * step) for k in range(1, after + 1)]
    side = -1 if axis.x0 == axis.x1 and axis.y1 > axis.y0 else 1
    for index, along in places:
        level = 0 if index % binning.nn2 == 0 else (1 if index % binning.nn3 == 0 else 2)
        across = 0.0 if geo.mside else -ticks[level]
        out.lines.append((*_rotate(along, ticks[level], geo, origin), *_rotate(along, across, geo, origin)))
        if "W" in options and level == 0:
            out.grid.append((*_rotate(along, side * axis.grid_length, geo, origin), *_rotate(along, 0.0, geo, origin)))


# -- the numbers ----------------------------------------------------------------------------


class _Format(NamedTuple):
    """How the labels are printed: the format, the first value and the step, and a ``×10^n``."""

    format: str
    first: float
    step: float
    exponent: int


def _scaled(ww: float, first: float, step: float, up: bool) -> tuple[int, float, float, float]:
    """The labels divided (``up``) or multiplied by ten until they are short, by powers of a thousand."""
    exponent = 0
    small = 1 / 10 ** (MAX_DIGITS - 2)
    while True:
        exponent += 1 if up else -1
        factor = 0.1 if up else 10.0
        ww, first, step = ww * factor, first * factor, step * factor
        if exponent % 3 == 0 and (ww <= 10 ** (MAX_DIGITS - 1) if up else ww >= small):
            return exponent, ww, first, step


def _tiny(ww: float) -> int | None:
    """For steps under the fewest digits' last: the power of a thousand small labels are shown times."""
    af = math.log10(ww) + EPSILON
    if af >= 0:
        return None
    iexe = abs(int(af))
    return iexe + {1: 2, 2: 1}.get(iexe % 3, 0)


def _finished(if1: int, if2: int, first: float, step: float, exponent: int, negative: bool = False) -> _Format:
    """The format of ``if1`` digits, ``if2`` after the point, widened until the step shows."""
    if1 = min(if1 + (1 if negative else 0), 32)
    while step < 10.0 ** (-if2):
        if1, if2 = if1 + 1, if2 + 1
    if1, if2 = max(min(if1, 14), 0), min(if2, 14)
    return _Format(f"%{if1}.{if2}f" if if2 > 0 else f"%{if1 + 1}.1f", first, step, exponent)


def _label_format(wmin: float, wmax: float, n1a: int, no_exponent: bool) -> _Format:
    """``PaintAxis``'s choice of format for the labels of a linear axis."""
    first, step = wmin, (wmax - wmin) / n1a
    ww = max(abs(wmin), abs(wmax)) or 1.0
    negative = min(wmin, wmax) < 0
    if not no_exponent and abs(wmax - wmin) / n1a < 10.0 ** (-MAX_DIGITS):
        iexe = _tiny(ww)
        if iexe is not None:
            return _finished(MAX_DIGITS, MAX_DIGITS - 2, first * 10**iexe, step * 10**iexe, -iexe, negative)
    af = (math.log10(ww) if ww >= 1 else math.log10(ww * 0.0001)) + EPSILON
    nf = int(af) + 1
    exponent = 0
    if not no_exponent and (nf > MAX_DIGITS or nf < -MAX_DIGITS):
        exponent, ww, first, step = _scaled(ww, first, step, nf > MAX_DIGITS)
    na = max((MAX_DIGITS - i for i in range(MAX_DIGITS - 1, 0, -1) if abs(ww) < 10**i), default=0)
    ndyn = n1a
    while ndyn and abs((wmax - wmin) / ndyn) <= 0.999 and na < MAX_DIGITS - 2:
        na, ndyn = na + 1, ndyn // 10
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
    return _Text(height, axis.label_size, 10 * across + up, _label_offset(axis, geo, options, log, height))


def _label_offset(axis: Axis, geo: _Geometry, options: str, log: bool, height: float) -> float:
    """``ylabel``: how far across the axis its labels are put, in NDC."""
    plus, minus = "+" in options, "-" in options
    tick = (1 if geo.mside >= 0 else -1) * geo.length * (axis.tick_size if "S" in options else 0.03)
    offset = axis.label_offset
    if axis.x0 == axis.x1:
        if plus and not minus:
            return offset / 2 + tick if "=" in options else -offset
        return offset + (tick if geo.lside < 0 else 0.0)
    if axis.y0 == axis.y1:
        if minus and not plus:
            found = offset + 0.5 * height + abs(tick)
        else:
            found = -offset - (abs(tick) if geo.mside <= 0 else 0.0)
        return found - (0.5 * height if log else 0.0)
    return offset if geo.mside + geo.lside >= 0 else -offset


def _linear_labels(axis: Axis, geo: _Geometry, binning: _Binning, text: _Text, out: Painted, options: str) -> None:
    """The labels of a linear axis, one per primary division, and its ``×10^n`` if it has one."""
    if not binning.n1a:
        return
    form = _label_format(binning.wmin, binning.wmax, binning.n1a, "noexponent" in axis.bits)
    origin = (axis.x0, axis.y0) if binning.noopt else binning.start
    length = geo.length if binning.noopt else binning.length
    step = length / binning.n1a
    centred = "M" in options or "centerlabels" in axis.bits
    value = form.first
    for k in range((binning.n1a - 1 if centred else binning.n1a) + 1):
        along = step * k + (0.5 * step if centred else 0.0)
        label = _printed(value, form.format, "." in options)
        value += form.step
        u, v = _rotate(along, text.offset, geo, origin)
        if axis.y0 == axis.y1:
            v -= 0.80 * text.height
        out.labels.append(Label(label.replace("-", "#minus"), u, v, text.align, 0.0,
                                axis.label_font, text.size, axis.label_color))  # fmt: skip
    if form.exponent:
        along = (geo.length if axis.x0 != axis.x1 else axis.y1 - axis.y0) + 0.1 * text.height
        u, v = _rotate(along, 0.0, geo, (axis.x0, axis.y0))
        font = axis.label_font if axis.label_font % 10 >= 2 else axis.label_font // 10 * 10 + 2
        out.labels.append(Label(f"#times10^{{{form.exponent}}}".replace("-", "#minus"), u, v, 11, 0.0,
                                font, text.size, axis.label_color))  # fmt: skip


# -- logarithmic axes -------------------------------------------------------------------------


class _Decades(NamedTuple):
    """A logarithmic axis's decades: the first, how many, the NDC per decade, and the first label."""

    low: float
    first: int
    count: int
    scale: float
    number: int


def _decades(axis: Axis, geo: _Geometry) -> _Decades:
    low = math.log10(axis.wmin)
    low += 1e-6 if low > 0 else -1e-6
    high = math.log10(axis.wmax)
    edge = high + (1e-6 if high > 0 else -1e-6)
    first, last = int(low), 1 + int(edge)
    number = first + (1 if low > 0 and low - first > 0 else 0)
    return _Decades(low, first, last - first + 1, geo.length / (high - low), number)


def _log_label(axis: Axis, geo: _Geometry, text: _Text, number: int, at: tuple[float, float]) -> Label:
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
    return Label(shown.replace("-", "#minus"), u, v, text.align, 0.0, axis.label_font, text.size, axis.label_color)


def _log_ticks(axis: Axis, geo: _Geometry, text: _Text, options: str, out: Painted) -> None:
    """A logarithmic axis: a long tick and a label at each decade, short ones at its multiples."""
    if axis.wmin <= 0 or axis.wmax <= 0 or axis.wmin == axis.wmax:
        return
    size = axis.tick_size if "S" in options else 0.03
    first = (1 if geo.mside >= 0 else -1) * geo.length * size
    decades = _decades(axis, geo)
    n1a = axis.ndiv % 100
    offset, number = text.offset, decades.number
    origin = (axis.x0, axis.y0)
    side = -1 if axis.x0 == axis.x1 and axis.y1 > axis.y0 else 1
    labelled = "U" not in options and axis.label_offset <= 1.1
    for j in range(1, decades.count + 1):
        decade = decades.first - 2 + j
        if j == 1:
            offset += text.height * (0.33 if axis.x0 == axis.x1 else -0.65 if axis.y0 == axis.y1 else 0.0)
        along = decades.scale * (decade - decades.low)
        if along >= 0:
            if along - geo.length > EPSILON:
                return
            _tick(out, geo, origin, along, first, options, side, axis.grid_length)
            if labelled:
                if not n1a:
                    return
                if _shows(decades.count, n1a, j):
                    out.labels.append(_log_label(axis, geo, text, number, _rotate(along, offset, geo, origin)))
                number += 1
        if not _multiples(axis, geo, decades, decade, first / 2, options, out, n1a):
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
        out.grid.append((*_rotate(along, side * grid, geo, origin), *_rotate(along, 0.0, geo, origin)))


def _multiples(axis: Axis, geo: _Geometry, decades: _Decades, decade: int, length: float, options: str,
               out: Painted, n1a: int) -> bool:  # fmt: skip
    """The ticks at 2 to 9 times a decade, all of them or only 5 when there are many decades.

    Returns whether the axis goes on past them.
    """
    origin = (axis.x0, axis.y0)
    side = -1 if axis.x0 == axis.x1 and axis.y1 > axis.y0 else 1
    for k in range(2, 10):
        along = decades.scale * (math.log10(k) + decade - decades.low)
        if along < 0:
            continue
        if along > geo.length:
            return False
        if decades.count <= n1a * 2 or k == 5:
            gridded = decades.count <= 5 and axis.ndiv > 100
            _tick(out, geo, origin, along, length, options, side, axis.grid_length, gridded)
    return True


# -- the title ---------------------------------------------------------------------------------


def _title(axis: Axis, geo: _Geometry, out: Painted) -> None:
    """The axis's title, beyond its labels, at its far end or its middle, turned along it."""
    height = axis.title_size
    if axis.title_font % 10 > 2:
        height /= axis.pad[0] if axis.x1 == axis.x0 else axis.pad[1]
    offset = axis.title_offset or 1.0
    across = geo.lside * (1.6 if axis.x1 == axis.x0 or axis.y1 == axis.y0 else 1.3) * height * offset
    centre = "centertitle" in axis.bits
    along = 0.5 * geo.length if centre else geo.length
    forward = axis.x1 >= axis.x0
    angle = geo.phil
    if "rotatetitle" in axis.bits:
        align = 22 if centre else (12 if forward else 32)
        angle += math.pi
    else:
        align = 22 if centre else (32 if forward else 12)
    u, v = _rotate(along, across, geo, (axis.x0, axis.y0))
    out.labels.append(Label(axis.title, u, v, align, math.degrees(angle), axis.title_font, axis.title_size,
                            axis.title_color))  # fmt: skip


def paint_axis(axis: Axis, pixel: Any) -> Painted:
    """Everything ``TGaxis::PaintAxis`` paints for ``axis``; ``pixel`` turns NDC into whole pixels."""
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
        if "U" not in options and axis.label_offset <= 1.1:
            _linear_labels(axis, geo, binning, text, out, options)
    if axis.title:
        _title(axis, geo, out)
    return out
