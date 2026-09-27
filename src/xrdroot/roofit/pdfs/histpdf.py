"""``RooHistPdf`` and ``RooHistFunc``: a density - or a function - that is a binned dataset's shape.

``RooHistPdf("h", "h", x, datahist)`` is the data's weight in the bin of
``x``, over the bin's width - a density whose integral over the full range
is the data's sum of weights; ``RooHistFunc`` is the weight itself. Only
interpolation order 0 - the bin's own value - is RooFit's exactly; a higher
order interpolates linearly between bin centres, and says so.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import as_list
from ..messages import WARNING, log
from ..pdf import RooAbsPdf
from ..real import Context, RooAbsReal

__all__ = ["RooHistFunc", "RooHistPdf"]


class _Histogram:
    """What both classes share: the data, its variables in order, and a value per point."""

    def _setup(self, variables: Any, data: Any, order: int) -> None:
        self.observables = self._list_proxy("pdfObs", as_list(variables))  # type: ignore[attr-defined]
        self.data = data
        self.order = int(order)
        if self.order > 1:
            log(
                self,
                WARNING,
                "Eval",
                f"{self.ClassName()}::{self.GetName()} interpolates to order "  # type: ignore[attr-defined]
                f"{self.order} as xrdroot does - linearly - not as RooFit's polynomial does",
            )

    def _weights(self, ctx: Context, density: bool) -> Any:
        columns = {
            mine.GetName(): np.atleast_1d(np.asarray(obs.compute(ctx), dtype=np.float64))
            for mine, obs in zip(self.data.get(), self.observables)
        }
        if self.order >= 1 and len(columns) == 1:
            return self._interpolated(next(iter(columns.values())), density)
        bins = self.data._bin_of(columns)
        weights = self.data.weights()
        volumes = self.data.binVolumes() if density else np.ones(len(weights))
        found = np.where(
            bins >= 0, weights[np.clip(bins, 0, None)] / volumes[np.clip(bins, 0, None)], 0.0
        )
        return np.maximum(found, 0.0)

    def _interpolated(self, x: Any, density: bool) -> Any:
        var = next(iter(self.data.get()))
        centres = self.data.column(var.GetName())
        values = self.data.weights() / (self.data.binVolumes() if density else 1.0)
        return np.maximum(np.interp(x, centres, values), 0.0)

    def dataHist(self) -> Any:
        return self.data


class RooHistPdf(_Histogram, RooAbsPdf):
    """A binned dataset's shape, as a density."""

    def __init__(self, name: Any, title: Any, variables: Any, data: Any, intOrder: int = 0) -> None:
        RooAbsPdf.__init__(self, name, title)
        self._setup(variables, data, intOrder)

    def compute(self, ctx: Context) -> Any:
        found = self._weights(ctx, True)
        return found if np.ndim(self.observables[0].compute(ctx)) else float(found[0])

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        mine = frozenset(one.GetName() for one in self.observables)
        return mine if mine <= names and not rng and self.order == 0 else frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.data.sum(False)


class RooHistFunc(_Histogram, RooAbsReal):
    """A binned dataset's weights, as a function."""

    def __init__(self, name: Any, title: Any, variables: Any, data: Any, intOrder: int = 0) -> None:
        RooAbsReal.__init__(self, name, title)
        self._setup(variables, data, intOrder)

    def compute(self, ctx: Context) -> Any:
        found = self._weights(ctx, False)
        return found if np.ndim(self.observables[0].compute(ctx)) else float(found[0])
