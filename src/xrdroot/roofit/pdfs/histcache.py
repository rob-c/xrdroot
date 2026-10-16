"""Small fixed histogram lookups recur at every likelihood evaluation.

Keys include the coordinates, weights and binning contents, so changing a
parameter, histogram or axis cannot reuse a previous value. Custom binning
and data classes retain their own evaluation behavior.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..binning import RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning
from ..data.datahist import RooDataHist
from ..variables import RooRealVar

#: Match the existing lookup cache's bound; large event columns are computed directly.
LIMIT = 4096
#: These binnings' arrays depend only on the numeric fields recorded in a key.
_BINNINGS = (RooAbsBinning, RooBinning, RooRangeBinning, RooUniformBinning)


def _array(value: Any) -> tuple[Any, ...]:
    return value.shape, value.dtype.str, value.tobytes()


def points(values: Iterable[Any]) -> tuple[Any, ...] | None:
    arrays = list(values)
    if any(value.size > LIMIT for value in arrays):
        return None
    return tuple(_array(value) for value in arrays)


def _axis(var: Any) -> tuple[Any, ...] | None:
    if type(var) is not RooRealVar:
        return None
    binning = var.getBinning()
    if type(binning) not in _BINNINGS:
        return None
    # Snapshot the immutable numbers, including interior edges, without sorting an axis again.
    return (
        var.GetName(),
        binning._low,
        binning._high,
        binning._bins,
        binning._uniform,
        tuple(binning._edges),
    )


def geometry(data: Any) -> tuple[Any, ...] | None:
    if type(data) is not RooDataHist:
        return None
    axes = [_axis(var) for var in data.get()]
    return None if any(axis is None for axis in axes) else tuple(axes)


def located_key(data: Any, columns: dict[str, Any]) -> tuple[Any, ...] | None:
    coords, axes = points(columns.values()), geometry(data)
    return None if coords is None or axes is None else (tuple(columns), coords, axes)


def weights_key(histogram: Any, columns: dict[str, Any]) -> tuple[Any, ...] | None:
    from .histpdf import RooHistFunc, RooHistPdf

    if histogram.order != 0 or type(histogram) not in (RooHistFunc, RooHistPdf):
        return None
    if type(histogram.data) is not RooDataHist:
        return None
    weights = histogram.data.weights()
    if weights.size > LIMIT:
        return None
    location = located_key(histogram.data, columns)
    if location is None:
        return None
    return location, histogram.density, _array(weights)
