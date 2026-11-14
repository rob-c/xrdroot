"""The fillers of an ``RHist``: one that finds its axis from the first entries, and one through
which threads fill a histogram they share.

``RHistAutoAxisFiller`` keeps entries until it has enough (or is asked), then
makes a regular axis from their least to their greatest value, widened by a
margin, and fills them in. ``RHistConcurrentFiller`` hands out contexts;
a context's fill is made whole - its bin and its statistics - before
another thread's, under the lock every atomic operation takes. ROOT's
context keeps its statistics until it is flushed, at the latest when it
is destroyed; Python destroys nothing at a scope's end, so here each
fill reaches the histogram's statistics at once, which is where ROOT's
are once every context is gone.
"""

from __future__ import annotations

import math
from typing import Any

from .hist import HIST, RHist
from .stats import ATOMIC, RWeight

__all__ = ["RHistAutoAxisFiller", "RHistConcurrentFiller", "RHistFillContext"]


class RHistAutoAxisFiller:
    """``RHistAutoAxisFiller<T>(nNormalBins, maxBufferSize, marginFraction)``."""

    #: The bin content type, ``T``: set on each class ``RHistAutoAxisFiller[T]`` makes.
    content: Any = "double"

    def __init__(self, nNormalBins: Any, maxBufferSize: Any = 1024,
                 marginFraction: Any = 0.05) -> None:  # fmt: skip
        self._normal, self._size = int(nNormalBins), int(maxBufferSize)
        self._margin = float(marginFraction)
        if self._normal == 0:
            raise ValueError("nNormalBins must be > 0")
        if self._size == 0:
            raise ValueError("maxBufferSize must be > 0")
        if self._margin <= 0:
            raise ValueError("marginFraction must be > 0")
        self._hist: RHist | None = None
        self._buffer: list[tuple[float, RWeight | None]] = []
        self._minimum, self._maximum = math.inf, -math.inf

    def GetNNormalBins(self) -> int:
        return self._normal

    def GetMaxBufferSize(self) -> int:
        return self._size

    def GetMarginFraction(self) -> float:
        return self._margin

    def Fill(self, x: Any, weight: RWeight | None = None) -> None:
        """``Fill(x[, weight])``: into the histogram once it is made, else kept until then."""
        extra = () if weight is None else (weight,)
        if self._hist is not None:
            self._hist.Fill(x, *extra)
            return
        x = float(x)
        self._buffer.append((x, weight))
        # std::min and std::max: a NaN, which compares false, changes neither
        self._minimum = x if x < self._minimum else self._minimum
        self._maximum = x if self._maximum < x else self._maximum
        if len(self._buffer) >= self._size:
            self.Flush()

    def Flush(self) -> None:
        """The histogram made, on ``[min - margin, max + margin)``, and the entries kept filled
        into it."""
        if self._hist is not None:
            return
        if not self._buffer:
            raise RuntimeError("buffer is empty, cannot create histogram")
        if not (math.isfinite(self._minimum) and math.isfinite(self._maximum)):
            raise RuntimeError("could not determine axis interval")
        if self._minimum == self._maximum:
            raise RuntimeError("axis interval is empty")
        margin = self._margin * (self._maximum - self._minimum)
        hist = HIST[self.content](self._normal, (self._minimum - margin, self._maximum + margin))
        for x, weight in self._buffer:
            hist.Fill(x, *(() if weight is None else (weight,)))
        self._hist, self._buffer = hist, []

    def GetHist(self) -> RHist:
        self.Flush()
        assert self._hist is not None
        return self._hist


class RHistFillContext:
    """``RHistFillContext<T>``: one thread's filling of the histogram its filler shares."""

    def __class_getitem__(cls, kind: Any) -> type:
        """``RHistFillContext<T>``: one class serves every ``T``."""
        return cls

    def __init__(self, hist: RHist) -> None:
        self._hist = hist

    def Fill(self, *args: Any) -> None:
        with ATOMIC:
            self._hist.GetEngine().FillAtomic(*args)
            self._hist.GetStats().Fill(*args)

    def Flush(self) -> None:
        """``Flush``: each fill went to the histogram's statistics as it was made."""


class RHistConcurrentFiller:
    """``RHistConcurrentFiller<T>(hist)``: contexts through which threads fill one histogram."""

    def __class_getitem__(cls, kind: Any) -> type:
        """``RHistConcurrentFiller<T>``: one class serves every ``T``."""
        return cls

    def __init__(self, hist: RHist | None) -> None:
        if hist is None:
            raise ValueError("hist must not be nullptr")
        self._hist = hist

    def GetHist(self) -> RHist:
        return self._hist

    def CreateFillContext(self) -> RHistFillContext:
        return RHistFillContext(self._hist)
