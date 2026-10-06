"""``RooKeysPdf``: a smooth density estimated from a dataset, one adaptive Gaussian per event.

``RooKeysPdf("k", "k", x, data, RooKeysPdf.MirrorBoth)`` puts a Gaussian
kernel on every event of ``data``, each as wide as the density near it is
thin - Cranmer's adaptive kernel estimate - optionally with the data
mirrored at either end of the range so that the estimate does not fall off
there. RooFit does not evaluate the kernels at every call: it tabulates
their sum at a thousand and one points once, when made, and interpolates
that table linearly; its integral is the trapezoid rule on the same table.
Both are ROOT's to the letter, including the table's grid - its first
point in a bin is placed as if the table had ``1000`` intervals, its step
as if ``999`` - because the numbers are what must agree.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooKeysPdf"]

#: ``RooKeysPdf::_nPoints``: the lookup table has this many intervals - and one more point.
N_POINTS = 1000
#: ``RooKeysPdf::_nSigma``: how many widths out a Gaussian is still above a double's precision.
N_SIGMA = math.sqrt(-2.0 * math.log(np.finfo(np.float64).eps))
#: How many kernel sums are taken at once, to keep the pairwise arrays small.
CHUNK = 512

#: The mirror modes, in ROOT's order: whether each mirrors left, right, subtracts left, right.
MODES = {
    0: (False, False, False, False), 1: (True, False, False, False), 2: (False, True, False, False),
    3: (True, True, False, False), 4: (False, False, True, False), 5: (False, True, True, False),
    6: (False, False, False, True), 7: (True, False, False, True), 8: (False, False, True, True),
}  # fmt: skip


def _kernel_sums(
    points: np.ndarray[Any, Any], at: np.ndarray[Any, Any], sigma: float
) -> np.ndarray[Any, Any]:
    """``RooKeysPdf::g``: the fixed-width kernel estimate at ``at``, from the sorted ``points``.

    Only the events within ``N_SIGMA`` widths count, found by bisection as
    ROOT finds them; every event counts once, whatever its weight.
    """
    found = np.empty(len(at))
    for start in range(0, len(at), CHUNK):
        x = at[start : start + CHUNK]
        low = np.searchsorted(points, x - N_SIGMA * sigma, side="left")
        high = np.searchsorted(points, x + N_SIGMA * sigma, side="right")
        index = np.arange(len(points))
        inside = (index[None, :] >= low[:, None]) & (index[None, :] < high[:, None])
        r = (x[:, None] - points[None, :]) / sigma
        found[start : start + CHUNK] = np.sum(np.where(inside, libm.exp(-0.5 * r * r), 0.0), axis=1)
    return found / (sigma * math.sqrt(2.0 * math.pi))


def _options(data: Any, mirror: int = 0, rho: float = 1.0) -> tuple[Any, int, float]:
    """The constructor's last arguments, with ROOT's defaults: no mirroring, ``rho`` of one."""
    return data, int(mirror), float(rho)


def _spread(values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> float:
    """The weighted standard deviation of the events, as ROOT accumulates it."""
    total = float(np.sum(weights))
    mean = float(np.sum(weights * values)) / total
    return math.sqrt(float(np.sum(weights * values * values)) / total - mean * mean)


class RooKeysPdf(RooAbsPdf):
    """The adaptive kernel estimate of one variable's density in a dataset."""

    #: ``RooKeysPdf::Mirror``: at which ends the data are mirrored - or mirrored and subtracted.
    NoMirror, MirrorLeft, MirrorRight, MirrorBoth = 0, 1, 2, 3
    MirrorAsymLeft, MirrorAsymLeftRight, MirrorAsymRight, MirrorLeftAsymRight, MirrorAsymBoth = (
        4,
        5,
        6,
        7,
        8,
    )

    def __init__(self, name: Any, title: Any, x: Any, *args: Any) -> None:
        super().__init__(name, title)
        xdata, rest = (args[0], args[1:]) if hasattr(args[0], "getMin") else (x, args)
        data, mirror, rho = _options(*rest)
        if int(mirror) not in MODES:
            raise ValueError(f"RooKeysPdf has no mirror mode {mirror}: the modes are 0 to 8.")
        self.x = self._proxy("x", x)
        self._mirror = MODES[int(mirror)]
        self._lo, self._hi = float(xdata.getMin()), float(xdata.getMax())
        self._bin_width = (self._hi - self._lo) / (N_POINTS - 1)
        self._rho = float(rho)
        self._var_name = xdata.GetName()
        self.LoadDataSet(data)

    # -- the lookup table ---------------------------------------------------------

    def LoadDataSet(self, data: Any) -> None:
        """Take the events - mirrored as asked - choose each kernel's width, and fill the table."""
        values = np.asarray(data.column(self._var_name), dtype=np.float64)
        weights = np.asarray(data.weights(), dtype=np.float64)
        points, point_weights = self._mirrored(values, weights)
        copies = 1 + int(self._mirror[0]) + int(self._mirror[1])
        self._sum_weights = float(np.sum(copies * weights))
        self._points, self._point_weights = points, point_weights
        self._widths = self._adaptive_widths(_spread(values, weights))
        table = self._kernels(points, 1.0)
        for subtract, at in zip(self._mirror[2:], (self._lo, self._hi), strict=False):
            if subtract:
                table -= self._kernels(2.0 * at - points, -1.0)
        self._table = table / (math.sqrt(2.0 * math.pi) * self._sum_weights)

    def _adaptive_widths(self, sigma: float) -> np.ndarray[Any, Any]:
        """Each kernel's width: wider where a fixed-width estimate is thin, not below a floor."""
        h = (4.0 / 3.0) ** 0.2 * self._sum_weights**-0.2 * self._rho
        norm = h * math.sqrt(sigma * self._sum_weights) / (2.0 * math.sqrt(3.0))
        density = _kernel_sums(self._points, self._points, h * sigma)
        widths = norm / np.sqrt(self._point_weights * density)
        return np.maximum(widths, h * sigma * math.sqrt(2.0) / 10)

    def _mirrored(
        self, values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
    ) -> tuple[Any, Any]:
        """The events and their reflections, sorted - each copy after its event, as in ROOT."""
        columns = [values]
        if self._mirror[0]:
            columns.append(2.0 * self._lo - values)
        if self._mirror[1]:
            columns.append(2.0 * self._hi - values)
        points = np.stack(columns, axis=1).reshape(-1)
        repeated = np.repeat(weights, len(columns))
        order = np.argsort(points, kind="stable")
        return points[order], repeated[order]

    def _kernels(self, centres: np.ndarray[Any, Any], sign: float) -> np.ndarray[Any, Any]:
        """Every kernel at the table's points, each only where it is above a double's precision.

        ``sign`` is -1 for the subtracted reflections, whose window ROOT
        computes with the width's sign flipped - so it is empty, and they
        subtract nothing: ROOT's own behaviour, kept.
        """
        reach = sign * N_SIGMA * self._widths
        xlo = np.minimum(self._hi, np.maximum(self._lo, centres - reach))
        xhi = np.maximum(self._lo, np.minimum(self._hi, centres + reach))
        keep = xlo < xhi
        centres, widths, xlo, xhi = centres[keep], self._widths[keep], xlo[keep], xhi[keep]
        ratio = self._point_weights[keep] / widths
        chi, inside = self._chi(centres, widths, xlo, xhi)
        return np.sum(np.where(inside, ratio[:, None] * libm.exp(-chi * chi), 0.0), axis=0)

    def _chi(self, centres: Any, widths: Any, xlo: Any, xhi: Any) -> tuple[Any, Any]:
        """Each kernel's argument at the table's points - stepped from its first bin, as ROOT
        steps it, from a start placed as if the table had ``N_POINTS`` intervals - and where,
        between ``xlo`` and ``xhi``, the kernel is taken."""
        binlo = np.floor((xlo - self._lo) / self._bin_width).astype(np.int64)
        binhi = (N_POINTS - np.floor((self._hi - xhi) / self._bin_width)).astype(np.int64)
        start = ((N_POINTS - binlo) * self._lo + binlo * self._hi) / N_POINTS
        steps = np.arange(N_POINTS + 1)[None, :] - binlo[:, None]
        first = (start - centres) / widths / math.sqrt(2.0)
        step = self._bin_width / widths / math.sqrt(2.0)
        chi = first[:, None] + steps * step[:, None]
        return chi, (steps >= 0) & (steps + binlo[:, None] <= binhi[:, None])

    # -- values and integrals -----------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        x = np.asarray(self.x.compute(ctx), dtype=np.float64)
        i = np.clip(np.floor((x - self._lo) / self._bin_width), 0, N_POINTS - 1).astype(np.int64)
        dx = (x - (self._lo + i * self._bin_width)) / self._bin_width
        found = np.maximum(self._table[i] + dx * (self._table[i + 1] - self._table[i]), 0.0)
        return found if found.ndim else float(found)

    def analytic_names(self, names: frozenset[str], rng: Any) -> frozenset[str]:
        mine = self.x.GetName()
        return frozenset([mine]) if self.x.isFundamental() and mine in names else frozenset()

    def analytic(self, names: frozenset[str], ctx: Context, rng: Any) -> Any:
        """The trapezoid rule on the table, from the range's low end to its high end."""
        xmin = max(self._lo, self.x.getMin(rng))
        xmax = min(self._hi, self.x.getMax(rng))
        imin = math.floor((xmin - self._lo) / self._bin_width)
        imax = min(math.floor((xmax - self._lo) / self._bin_width), N_POINTS - 1)
        return float(self._whole_bins(imin, imax) + self._part_bins(xmin, xmax, imin, imax))

    def _whole_bins(self, imin: int, imax: int) -> float:
        """The trapezoids of the table's intervals wholly inside the range."""
        t = self._table
        total = t[imin + 1] + t[imax] if imin + 1 < imax else 0.0
        if imax > imin + 2:
            total += 2.0 * float(np.sum(t[imin + 2 : imax]))
        return float(total) * self._bin_width * 0.5

    def _along(self, x: float, i: int) -> tuple[float, float]:
        """How far into interval ``i`` the value ``x`` is, as a fraction, and the table there."""
        t, w = self._table, self._bin_width
        dx = (x - (self._lo + i * w)) / w
        return dx, t[i] + dx * (t[i + 1] - t[i])

    def _part_bins(self, xmin: float, xmax: float, imin: int, imax: int) -> float:
        """The pieces of the intervals the range's ends fall in - or of the one it lies in."""
        t, w = self._table, self._bin_width
        if imin > imax:
            return 0.0
        dxmin, low = self._along(xmin, imin)
        dxmax, high = self._along(xmax, imax)
        if imin < imax:
            return float(
                w * (1.0 - dxmin) * 0.5 * (t[imin + 1] + low) + w * dxmax * 0.5 * (t[imax] + high)
            )
        return float(w * (dxmax - dxmin) * 0.5 * (low + high))

    def maxVal(self, code: int = 1) -> float:
        """``maxVal``: the table's highest point, which no interpolation between points exceeds."""
        return float(np.max(self._table))
