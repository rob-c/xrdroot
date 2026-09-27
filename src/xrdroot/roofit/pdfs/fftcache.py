"""One cache of a ``RooFFTConvPdf``: both densities sampled, convolved by FFT, kept as a histogram.

``RooFFTConvPdf::FFTCacheElem`` and ``fillCacheSlice``, step for step. The
convolution observable's "cache" binning - ``N`` bins - is widened by
``Nbuf = int(N * bufferFraction / 2 + 0.5)`` bins at either end; each
density is sampled at the ``N2 = N + 2 Nbuf`` centres of the widened range
(the second shifted by the middle of the range, so that the result lands
back in place), divided by its integral over the cache's observables *as it
was when the cache was made* - RooFit computes that once per cache - and
rotated so that the bin holding zero comes first. FFTW's unnormalised
real-to-complex and back again multiply the two transforms; ``numpy.fft``
does the same up to rounding, and the ``N2`` FFTW leaves in is kept. The
middle ``N`` values are the histogram's weights, the rest dropped.

Any other observable of the cache is a slice: every bin of its "cache"
binning is a convolution of its own, all of them transformed at once.
The histogram is read as ``RooHistPdf`` reads it - weight over bin volume,
interpolated to the order asked for (:mod:`..data.interpolate`), nothing
outside its range - and integrated as ``RooHistPdf`` integrates: a sum of
weights over the observables it holds, when those are the density's own.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from ..data.interpolate import Axis, clone_rows, weights_interpolated
from ..integration import announce
from ..messages import WARNING, log
from ..pdf import normalized
from ..real import Context

__all__ = ["FFTCache", "buffered", "cache_binning", "scan_layout", "sums_over"]

#: Below this many bins RooFit warns that a difficult convolution will be inaccurate.
FEW_BINS = 900
#: ``RooFFTConvPdf::BufStrat``: sample the buffers, reflect the range into them, or repeat its ends.
EXTEND, MIRROR, FLAT = 0, 1, 2


def cache_binning(var: Any) -> Any:
    """The binning a cache histogram has for ``var``: its "cache" binning, or its default."""
    return var.getBinning("cache") if var.hasBinning("cache") else var.getBinning()


def scan_layout(n: int, low: float, high: float, fraction: float, shift: float,
                extend: bool = True) -> tuple[int, int, int]:  # fmt: skip
    """``scanPdf``'s bookkeeping: the buffer bins either side, the sampled bins, the bin of zero.

    The bin holding zero - or, for a range not containing it, where zero
    would be counted from the nearer end - is moved by the shift, in bins of
    the widened range, and taken modulo the number of bins. The range it is
    counted in is the widened one when the buffer extends the range, the
    observable's own otherwise - with the widened range's bin width, as ROOT
    has it.
    """
    nbuf = int(n * fraction / 2 + 0.5)
    n2 = n + 2 * nbuf
    if extend:
        width = (high - low) / n
        low, high = low - nbuf * width, high + nbuf * width
    step = (high - low) / n2
    bins = n2 if extend else n
    if high >= 0 and low <= 0:
        zero = min(max(int((0.0 - low) / ((high - low) / bins)), 0), bins - 1)
    elif low > 0:
        zero = int(-low / step)
    else:
        zero = int(-high / step)
    zero += int(n2 * shift / (high - low))
    return nbuf, n2, zero % n2


def buffered(values: np.ndarray[Any, Any], nbuf: int, strategy: int) -> np.ndarray[Any, Any]:
    """The samples of the observable's own range with the buffers either side filled in.

    ``Flat`` repeats the value at each end, ``Mirror`` reflects the range
    about each end - its first bin not repeated below, its last repeated
    above - as ``RooFFTConvPdf::scanPdf`` fills them; ``Extend`` samples the buffers
    themselves, so ``values`` already has them.
    """
    if strategy == MIRROR:
        low = values[..., 1 : nbuf + 1][..., ::-1]
        high = values[..., values.shape[-1] - nbuf :][..., ::-1]
    elif strategy == FLAT:
        low = np.repeat(values[..., :1], nbuf, axis=-1)
        high = np.repeat(values[..., -1:], nbuf, axis=-1)
    else:
        return values
    return np.concatenate([low, values, high], axis=-1)


def sums_over(observables: list[Any], names: frozenset[str]) -> bool:
    """``RooHistPdf::getAnalyticalIntegral``: whether the integral over ``names`` is a sum.

    It is when each observable ``names`` touch is the histogram's own
    variable - not a function of one - and the histogram holds every one.
    """
    touched = [obs for obs in observables if obs.dependents() & names]
    held = {obs.GetName() for obs in observables}
    return (
        bool(touched)
        and all(obs.isFundamental() and obs.GetName() in names for obs in touched)
        and names <= held
    )


@contextmanager
def _range_moved(var: Any, low: float, high: float) -> Iterator[None]:
    """``var``'s own range set to ``[low, high]`` for a while, its value untouched."""
    binning = var.getBinning()
    saved = binning.lowBound(), binning.highBound()
    binning.setRange(low, high)
    try:
        yield
    finally:
        binning.setRange(*saved)


class FFTCache:
    """The convolution of the two densities over the cache's observables, as a histogram."""

    def __init__(self, conv: Any, hist_obs: list[Any], nset: frozenset[str]) -> None:
        self.conv = conv
        self.hist_obs = hist_obs
        self.names = [one.GetName() for one in hist_obs]
        self.binnings = [cache_binning(one) for one in hist_obs]
        self.axes = [Axis.of(b) for b in self.binnings]
        self.axis = self.names.index(conv.x.GetName())
        x = conv.x
        x.setRange(f"refrange_fft_{conv.GetName()}", self.binnings[self.axis].lowBound(),
                   self.binnings[self.axis].highBound())  # fmt: skip
        self.norms = (self._norm_of(conv.pdf1, 0.0), self._norm_of(conv.pdf2, conv._shift2))
        count = self.binnings[self.axis].numBins()
        if count < FEW_BINS:
            log(conv, WARNING, "Eval", f"The FFT convolution '{conv.GetName()}' will run with "
                f"{count} bins. A decent accuracy for difficult convolutions is typically only "
                "reached with n >= 1000. Suggest to increase the number of bins of the observable "
                f"'{x.GetName()}'.")  # fmt: skip
        self._key: Any = None
        self._grid: Any = None

    # -- making it ----------------------------------------------------------------

    def _norm_of(self, pdf: Any, shift: float) -> float:
        """``getNorm(hist()->get())``: the density's integral over the cache's observables, now.

        The second density sees the convolution variable shifted, so its
        integral is over the range moved by the shift, as RooFit's linear
        transform of the variable makes it.
        """
        names = frozenset(self.names) & pdf.dependents()
        if not names:
            return 1.0  # RooFit's unit normalisation for a density of none of them
        announce(pdf, names, label=f"{pdf.GetName()}_Int[{','.join(sorted(names))}]")
        binning = self.binnings[self.axis]
        with _range_moved(self.conv.x, binning.lowBound() - shift, binning.highBound() - shift):
            return float(np.asarray(pdf.norm({}, names)))

    def _slices(self) -> list[tuple[str, np.ndarray[Any, Any]]]:
        """The other observables' values at every slice: each bin centre of each, all combined."""
        others = [(i, name) for i, name in enumerate(self.names) if i != self.axis]
        centres = [self.axes[i].centres(np.arange(self.axes[i].count)) for i, _ in others]
        grid = np.meshgrid(*centres, indexing="ij") if centres else []
        return [(name, values.reshape(-1)) for (_, name), values in zip(others, grid)]

    def _sampled(
        self, pdf: Any, norm: float, shift: float, ctx: Context
    ) -> tuple[np.ndarray[Any, Any], int]:
        """``scanPdf``: the density at the bins' centres, normalised, buffered and rotated."""
        binning = self.binnings[self.axis]
        n, low, high = binning.numBins(), binning.lowBound(), binning.highBound()
        strategy = self.conv.bufferStrategy()
        nbuf, n2, zero = scan_layout(
            n, low, high, self.conv.bufferFraction(), shift, strategy == EXTEND
        )
        width = (high - low) / n
        if strategy == EXTEND:
            start = low - nbuf * width
            count, step = n2, (high + nbuf * width - start) / n2
        else:
            count, start, step = n, low, width
        at = dict(ctx)
        slices = self._slices()
        at.update({name: values[:, None] for name, values in slices})
        at[self.conv.x.GetName()] = start + (np.arange(count) + 0.5) * step - shift
        values = normalized(np.asarray(pdf.compute(at), dtype=np.float64), norm)
        rows = max((len(v) for _, v in slices), default=1)
        values = buffered(np.broadcast_to(values, (rows, count)), nbuf, strategy)
        return np.roll(values, zero, axis=1), zero

    def _filled(self, ctx: Context) -> np.ndarray[Any, Any]:
        """``fillCacheObject``: the histogram's weights, one axis per observable."""
        first, zero = self._sampled(self.conv.pdf1, self.norms[0], 0.0, ctx)
        second, _ = self._sampled(self.conv.pdf2, self.norms[1], self.conv._shift2, ctx)
        n2 = first.shape[1]
        product = np.fft.rfft(first, axis=1) * np.fft.rfft(second, axis=1)
        output = np.fft.irfft(product, n=n2, axis=1) * n2
        n = self.binnings[self.axis].numBins()
        index = (np.arange(n) + zero + (n2 - n) // 2) % n2
        weights = output[:, index]
        shape = [a.count for a in self.axes]
        others = [s for i, s in enumerate(shape) if i != self.axis]
        return np.moveaxis(weights.reshape(*others, n), -1, self.axis)

    # -- reading it ---------------------------------------------------------------

    def _parameters(self) -> list[Any]:
        """What RooFit's change tracker watches: the variables, less the observables of a cache
        of the density's observables - so a cache observable of a cache that does not hold it
        is not watched, and the cache is not refilled when it changes, as in RooFit."""
        final = self.conv.pdf_observable_names(self.names)
        mine = final | {one.GetName() for one in self.conv.actual_observables(frozenset(final))}
        return [one for one in self.conv.leaves() if one.GetName() not in mine]

    def weights(self, ctx: Context) -> np.ndarray[Any, Any]:
        """The weights for the parameters' values in ``ctx`` - refilled only when they change."""
        params = self._parameters()
        key = tuple(float(np.asarray(one.compute(ctx)).reshape(-1)[0]) for one in params)
        if key != self._key:
            self._key = key
            self._grid = self._filled({one.GetName(): v for one, v in zip(params, key)})
        return self._grid

    def _groups(self, ctx: Context) -> Iterator[tuple[Any, Context]]:
        """The events of ``ctx`` by the parameters' values: one histogram for each set of values."""
        params = [one for one in self._parameters() if np.ndim(ctx.get(one.GetName(), 0.0))]
        if not params:
            yield None, ctx
            return
        flat = _flattened(ctx)
        table = np.stack([flat[one.GetName()] for one in params], axis=1)
        values, inverse = np.unique(table, axis=0, return_inverse=True)
        for row, found in enumerate(values):
            keep = inverse.reshape(-1) == row
            one = {k: v[keep] if np.ndim(v) else v for k, v in flat.items()}
            one.update({p.GetName(): float(v) for p, v in zip(params, found)})
            yield keep, one

    def per_event(self, ctx: Context, read: Any) -> Any:
        """``read(weights, ctx)`` for every event, each with the histogram of its parameters."""
        groups = list(self._groups(ctx))
        if len(groups) == 1 and groups[0][0] is None:
            return read(self.weights(ctx), ctx)
        shape = np.broadcast_shapes(*(np.shape(v) for v in ctx.values()))
        found = np.empty(int(np.prod(shape)))
        for keep, one in groups:
            found[keep] = np.broadcast_to(
                read(self.weights(one), one), (int(np.count_nonzero(keep)),)
            )
        return found.reshape(shape)

    def raw(self, ctx: Context) -> Any:
        """``RooHistPdf::evaluate``: the weight over the bin volume at the observables' values."""
        return self.per_event(ctx, self._read)

    def _points(self, ctx: Context) -> tuple[list[np.ndarray[Any, Any]], tuple[int, ...]]:
        """The density's observables at ``ctx``, as flat arrays of one shape - and that shape."""
        observables = self.conv.pdf_observables(self.hist_obs)
        points = [np.asarray(obs.compute(ctx), dtype=np.float64) for obs in observables]
        shape = np.broadcast_shapes(*(p.shape for p in points))
        return [np.broadcast_to(p, shape).reshape(-1) for p in points], shape

    def _read(self, weights: np.ndarray[Any, Any], ctx: Context) -> Any:
        points, shape = self._points(ctx)
        widths = np.meshgrid(*(a.widths() for a in self.axes), indexing="ij")
        found = self._looked_up(weights / np.prod(widths, axis=0), points)
        inside = np.all([(p >= a.low) & (p <= a.high) for a, p in zip(self.axes, points)], axis=0)
        found = np.maximum(np.where(inside, found, 0.0), 0.0).reshape(shape)
        return found if found.ndim else float(found)

    def _looked_up(self, density: np.ndarray[Any, Any], points: list[Any]) -> Any:
        """``weightFast``: the bin's density, or - with an order - the interpolation of it."""
        order = self.conv.getInterpolationOrder()
        if order > 0 and len(points) <= 2:
            rows = clone_rows(self.hist_obs[-1], self.axes[-1].low, self.axes[-1].high)
            return weights_interpolated(self.axes, density, points, order, rows=rows)
        return density[tuple(a.numbers(p) for a, p in zip(self.axes, points))]

    def analytic_over(self, names: frozenset[str]) -> bool:
        return sums_over(self.conv.pdf_observables(self.hist_obs), names)

    def summed(self, names: frozenset[str], ctx: Context) -> Any:
        """``RooDataHist::sum``: the weights summed over ``names``, in the rest's bins."""
        return self.per_event(ctx, lambda weights, one: self._sum(weights, names, one))

    def _sum(self, weights: np.ndarray[Any, Any], names: frozenset[str], ctx: Context) -> Any:
        summed = tuple(i for i, name in enumerate(self.names) if name in names)
        kept = [i for i in range(len(self.names)) if i not in summed]
        total = np.sum(weights, axis=summed)
        if not kept:
            return float(total)
        widths = np.meshgrid(*(self.axes[i].widths() for i in kept), indexing="ij")
        points, shape = self._points(ctx)
        index = tuple(self.axes[i].numbers(points[i]) for i in kept)
        return (total / np.prod(widths, axis=0))[index].reshape(shape)


def _flattened(ctx: Context) -> Context:
    """``ctx`` with every array broadcast to one flat length - and its numbers as they are."""
    shape = np.broadcast_shapes(*(np.shape(v) for v in ctx.values()))
    return {k: np.broadcast_to(v, shape).reshape(-1) if np.ndim(v) else v for k, v in ctx.items()}
