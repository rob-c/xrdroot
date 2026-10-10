"""``TCandle``: a distribution summarised as a candle or a violin, and ROOT's options for one.

A candle is a slice of a two-dimensional histogram - the distribution along
one axis in one bin of the other - summed up as ``TCandle::Calculate`` sums
it: a box round the median holding ``BoxRange`` of the entries (the quartiles
by default), whiskers to the farthest bins within ``WhiskerRange`` inter-
quartile ranges of the box or to the distribution's ends, the mean, a notch
of ``1.57 IQR / sqrt(n)`` either side of the median, and the bins beyond the
whiskers as outliers. The option's digits ``zhpawMmb`` say what is drawn:
the box, the median (line, notched, circle), the mean (line, circle), the
whiskers (all, 1.5 IQR), anchors on their ends, points (outliers, all, all
scattered), the histogram (left, right, violin) and a zero line; ``CANDLEX1``
to ``6`` and ``VIOLINX1`` to ``2`` are ROOT's presets of them, ``Y`` lays the
candles along y. Points scattered at random are scattered by a generator of
this module's own, so that a macro's own random numbers are not consumed.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

__all__ = ["SETTINGS", "PRESETS", "Candle", "candle_of", "parse_option", "part"]

#: ``TCandle``'s settings for every candle: ``SetWhiskerRange``, ``SetBoxRange``,
#: ``SetScaledCandle`` and ``SetScaledViolin``.
SETTINGS: dict[str, Any] = {
    "whisker_range": 1.5, "box_range": 0.5, "scaled_candle": False, "scaled_violin": True,
}  # fmt: skip
#: The places of the option's digits ``zhpawMmb``, lowest first, and the horizontal bit.
BOX, MEDIAN, MEAN, WHISKER, ANCHOR, POINTS, HISTO, ZERO = (10**place for place in range(8))
HORIZONTAL = 10**8
#: ROOT's presets, by their digits.
PRESETS = {
    "CANDLE1": 112311, "CANDLE2": 112321, "CANDLE3": 111311, "CANDLE4": 111321,
    "CANDLE5": 212311, "CANDLE6": 312311, "VIOLIN1": 3001010, "VIOLIN2": 13300330,
}  # fmt: skip
#: ``kNMAXPOINTS``: the most points one candle draws.
MOST_POINTS = 2010
#: The probabilities at the distribution's ends, which ``GetQuantiles`` lands just inside.
ENDS = 1e-15


def parse_option(spelling: str) -> int:
    """The option's digits from how it was spelled: a preset's, those in brackets, or the
    first preset's; ``CANDLEY`` adds the horizontal bit."""
    text = spelling.upper()
    kind, rest = text[:6], text[6:]
    horizontal = rest.startswith("Y")
    rest = rest.lstrip("XY")
    if rest.startswith("("):
        code = int(rest[1:-1] or "0")
    else:
        code = PRESETS.get(f"{kind}{rest}", PRESETS[f"{kind}1"])
    return code + (HORIZONTAL if horizontal else 0)


def part(code: int, place: int) -> int:
    """``GetCandleOption``: the digit at ``place`` of the option."""
    return (code // place) % 10


class Candle(NamedTuple):
    """One candle, worked out: where it stands, how wide, and its distribution's summary."""

    pos: float
    width: float
    mean: float
    median: float
    median_err: float
    box_up: float
    box_down: float
    whisker_up: float
    whisker_down: float
    entries: float
    #: The slice's bin edges and contents, for a violin.
    edges: Any
    contents: Any
    #: The points to draw, along the candle's axis and across it.
    points: tuple[Any, Any]


def _whiskers(edges: Any, contents: Any, box: tuple[float, float]) -> tuple[float, float]:
    """``kWhisker15``: the farthest non-empty bins within ``WhiskerRange`` IQRs of the box."""
    reach = SETTINGS["whisker_range"] * (box[1] - box[0])
    centres = 0.5 * (edges[:-1] + edges[1:])
    last = len(contents) - 1
    low = min(max(int(np.searchsorted(edges, box[0] - reach, side="right")) - 1, 0), last)
    while contents[low] == 0 and low < last:
        low += 1
    high = min(max(int(np.searchsorted(edges, box[1] + reach, side="right")) - 1, 0), last)
    while contents[high] == 0 and high > 0:
        high -= 1
    return float(centres[low]), float(centres[high])


def _scattered(code: int, candle: tuple[float, float], bins: Any, rng: Any) -> tuple[Any, Any]:
    """``kPointsAll`` and ``kPointsAllScat``: a point per entry, at the bin's centre or
    scattered over the bin and across the candle, up to ``kNMAXPOINTS`` of them."""
    edges, contents = bins
    counts = np.minimum(np.maximum(contents, 0), MOST_POINTS).astype(int)
    running = np.cumsum(counts)
    kept = np.minimum(running, MOST_POINTS) - np.concatenate(([0], running[:-1]))
    index = np.repeat(np.arange(len(contents)), np.maximum(kept, 0))
    pos, width = candle
    if part(code, POINTS) == 3:
        along = edges[index] + (edges[index + 1] - edges[index]) * rng.random(len(index))
        return pos - width / 2 + width * rng.random(len(index)), along
    return np.full(len(index), pos), 0.5 * (edges[index] + edges[index + 1])


def _points(code: int, candle: tuple[float, float], bins: Any, whiskers: Any, rng: Any) -> Any:
    """The points the option asks for: the bins beyond the whiskers, or every entry."""
    kind = part(code, POINTS)
    if kind in (2, 3):
        return _scattered(code, candle, bins, rng)
    if kind != 1:
        return np.zeros(0), np.zeros(0)
    edges, contents = bins
    centres = 0.5 * (edges[:-1] + edges[1:])
    beyond = (contents > 0) & ((centres < whiskers[0]) | (centres > whiskers[1]))
    return np.full(int(beyond.sum()), candle[0]), centres[beyond]


def candle_of(proj: Any, pos: float, width: float, code: int, rng: Any) -> Candle | None:
    """``TCandle::Calculate`` on a slice: ``None`` for one ROOT dismisses, empty or with its
    quantiles out of order."""
    edges = np.asarray(proj.edges(0), dtype=float)
    contents = np.asarray(proj.values(), dtype=float)
    total = float(contents.sum())
    if total <= 0:
        return None
    box = SETTINGS["box_range"]
    q = [float(v) for v in proj.quantiles([ENDS, 0.5 - box / 2, 0.5, 0.5 + box / 2, 1 - ENDS])]
    if q[0] >= q[4] or q[1] > q[3]:
        return None
    whiskers = (q[0], q[4])
    if part(code, WHISKER) == 2:
        whiskers = _whiskers(edges, contents, (q[1], q[3]))
    points = _points(code, (pos, width), (edges, contents), whiskers, rng)
    return Candle(
        pos, width, float(proj.mean()), q[2], 1.57 * (q[3] - q[1]) / math.sqrt(total),
        q[3], q[1], whiskers[1], whiskers[0], total, edges, contents, points,
    )  # fmt: skip
