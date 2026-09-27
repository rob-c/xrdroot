"""``RooHistPdf`` and ``RooHistFunc``: a density - or a function - that is a binned dataset's shape.

``RooHistPdf("h", "h", x, datahist)`` is the data's weight in the bin of
``x``, over the bin's width - a density whose integral over the full range
is the data's sum of weights; ``RooHistFunc`` is the weight itself. With an
interpolation order the value between bin centres is the polynomial
through the nearest centres (:mod:`..data.interpolate`), in one or two
dimensions, as ``RooDataHist::weightInterpolated`` computes it - and the
integral over the full range is still the sum of weights, as RooFit has it.

The density's observables may be functions of the histogram's -
``RooHistPdf(name, title, pdfObs, histObs, datahist)`` - evaluated and then
looked up; a value outside the histogram's range weighs nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..collections import as_list
from ..data.interpolate import Axis, clone_rows, weights_interpolated
from ..messages import ERROR, log
from ..pdf import RooAbsPdf
from ..real import Context, RooAbsReal

__all__ = ["RooHistFunc", "RooHistPdf"]


def _split(args: tuple[Any, ...]) -> tuple[Any, Any, Any, int]:
    """``(vars, data[, order])`` or ``(pdfObs, histObs, data[, order])``: what the classes keep."""
    if len(args) >= 3 and hasattr(args[2], "numEntries"):
        return args[0], args[1], args[2], int(args[3]) if len(args) > 3 else 0
    return args[0], args[0], args[1], int(args[2]) if len(args) > 2 else 0


class _Histogram:
    """What both classes share: the data, its variables in order, and a value per point."""

    #: Whether the value is the weight over the bin's volume - a density - or the weight.
    density = True

    def _setup(self, variables: Any, histObs: Any, data: Any, order: int) -> None:
        proxy = self._list_proxy  # type: ignore[attr-defined]
        self.observables = proxy("pdfObs", as_list(variables))
        self.data = data
        #: The histogram's variable each observable is looked up as, in the observables' order.
        self._hist_vars = [data.get().find(one.GetName()) for one in as_list(histObs)]
        #: The variables as given, whose own binnings the density's copies of them keep.
        self._given = {one.GetName(): one for one in as_list(histObs)}
        self.order = int(order)
        self._cdf = False
        real = [one for one in data.get() if not hasattr(one, "lookupIndex")]
        if self.order > 0 and len(real) > 2:
            log(self, ERROR, "InputArguments", f"RooDataHist::weight({data.GetName()}) "
                "interpolation in "
                f"{len(real)} dimensions not yet implemented")  # fmt: skip

    def setInterpolationOrder(self, order: int) -> None:
        self.order = int(order)

    def getInterpolationOrder(self) -> int:
        return self.order

    def setCdfBoundaries(self, flag: bool) -> None:
        self._cdf = bool(flag)

    def getCdfBoundaries(self) -> bool:
        return self._cdf

    def _columns(self, ctx: Context) -> dict[str, Any]:
        """The histogram's variables' values: its observables' values at ``ctx``, as arrays."""
        return {mine.GetName(): np.asarray(obs.compute(ctx), dtype=np.float64)
                for mine, obs in zip(self._hist_vars, self.observables)}  # fmt: skip

    def _weights(self, ctx: Context) -> Any:
        columns = self._columns(ctx)
        shape = np.broadcast_shapes(*(one.shape for one in columns.values()))
        columns = {k: np.broadcast_to(v, shape).reshape(-1) for k, v in columns.items()}
        inside = np.ones(int(np.prod(shape)), dtype=bool)
        for var in self.data.get():
            values = columns[var.GetName()]
            inside &= (values >= var.getMin()) & (values <= var.getMax())
        weights = self.data.weights() / (self.data.binVolumes() if self.density else 1.0)
        if self.order > 0 and len(columns) <= 2:
            found = self._interpolated(columns, weights)
        else:
            bins = self.data._bin_of(columns)
            found = weights[np.clip(bins, 0, None)]
        found = np.where(inside, found, 0.0)
        return (np.maximum(found, 0.0) if self.density else found).reshape(shape)

    def _interpolated(self, columns: dict[str, Any], weights: Any) -> Any:
        axes = [Axis.of(var.getBinning()) for var in self.data.get()]
        grid = weights.reshape([a.count for a in axes])
        points = [columns[var.GetName()] for var in self.data.get()]
        last = list(self.data.get())[-1]
        rows = clone_rows(self._given.get(last.GetName(), last), axes[-1].low, axes[-1].high)
        return weights_interpolated(axes, grid, points, self.order, self._cdf, rows)

    def _value(self, ctx: Context) -> Any:
        found = self._weights(ctx)
        return found if found.ndim else float(found)

    def dataHist(self) -> Any:
        return self.data

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        """All the observables over their full range: the sum of the weights, RooFit's code 1."""
        mine = frozenset(one.GetName() for one in self.observables)
        if not (mine <= names and all(one.isFundamental() for one in self.observables)):
            return frozenset()
        pairs = zip(self.observables, self._hist_vars)
        full = all(_full_range(obs, var, rng) for obs, var in pairs)
        return mine if full else frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        return self.data.sum(not self.density)


def _full_range(obs: Any, var: Any, rng: Any) -> bool:
    """``fullRange``: whether ``rng`` of the observable is the histogram variable's whole range."""
    return bool(obs.getMin(rng) == var.getMin() and obs.getMax(rng) == var.getMax())


class RooHistPdf(_Histogram, RooAbsPdf):
    """A binned dataset's shape, as a density."""

    def __init__(self, name: Any, title: Any, *args: Any, **kwargs: Any) -> None:
        RooAbsPdf.__init__(self, name, title)
        variables, hist_obs, data, order = _split(args)
        self._setup(variables, hist_obs, data, kwargs.get("intOrder", order))

    def compute(self, ctx: Context) -> Any:
        return self._value(ctx)


class RooHistFunc(_Histogram, RooAbsReal):
    """A binned dataset's weights, as a function."""

    density = False

    def __init__(self, name: Any, title: Any, *args: Any, **kwargs: Any) -> None:
        RooAbsReal.__init__(self, name, title)
        variables, hist_obs, data, order = _split(args)
        self._setup(variables, hist_obs, data, kwargs.get("intOrder", order))

    def compute(self, ctx: Context) -> Any:
        return self._value(ctx)
