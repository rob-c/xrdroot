"""``RDataFrame::Hist``: an ``RHist`` booked on a frame, filled from its columns when first read.

The forms are ROOT's: ``Hist(nNormalBins, {low, high}, column[, weight])``,
``Hist(axes, columns[, weight])`` and ``Hist(hist, columns[, weight])`` for
an ``RHist`` made already. The columns are booked with ``Take``, so the
histogram fills in the frame's one event loop with everything else booked.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .hist import RHistEngine
from .stats import RWeight

__all__ = ["BookedHist", "book"]


class BookedHist:
    """What ``Hist`` gives back: the histogram, filled the first time it is asked for."""

    def __init__(self, hist: RHistEngine, taken: list[Any], weight: Any) -> None:
        self._hist, self._taken, self._weight = hist, taken, weight
        self._filled = False

    def GetValue(self) -> RHistEngine:
        if not self._filled:
            columns = [np.asarray(result.GetValue(), dtype=np.float64) for result in self._taken]
            weights = (None if self._weight is None else
                       np.asarray(self._weight.GetValue(), dtype=np.float64))  # fmt: skip
            for at, values in enumerate(zip(*columns, strict=False)):
                extra = () if weights is None else (RWeight(float(weights[at])),)
                self._hist.Fill(*values, *extra)
            self._filled = True
        return self._hist

    def IsReady(self) -> bool:
        return self._filled

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.GetValue(), name)


def _made(args: tuple[Any, ...]) -> tuple[RHistEngine, list[str], str | None]:
    """The histogram a ``Hist`` call fills, its columns, and its weight's column."""
    from . import Experimental

    if isinstance(args[0], (int, np.integer)):
        hist: Any = Experimental.RHist["double"](int(args[0]), args[1])
        return hist, [str(args[2])], (str(args[3]) if len(args) > 3 else None)
    columns = [str(name) for name in ([args[1]] if isinstance(args[1], str) else args[1])]
    weight = str(args[2]) if len(args) > 2 else None
    if isinstance(args[0], RHistEngine):
        return args[0], columns, weight
    axes = list(args[0])
    if len(axes) != len(columns):
        raise ValueError(f"Wrong number of columns for the specified number of histogram axes: "
                         f"expected {len(axes)}, got {len(columns)}")  # fmt: skip
    return Experimental.RHist["double"](axes), columns, weight


def book(frame: Any, args: tuple[Any, ...]) -> BookedHist:
    """``frame.Hist(args...)``: the histogram booked on the frame."""
    hist, columns, weight = _made(args)
    taken = [frame.Take(name) for name in columns]
    return BookedHist(hist, taken, None if weight is None else frame.Take(weight))

