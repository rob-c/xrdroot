"""``RooNDKeysPdf``: the same adaptive kernel estimate as ``RooKeysPdf``, in N dimensions.

Every event carries one Gaussian per observable, and the estimate at a point is
the mean of those kernels' products over the events - each observable's kernel as
wide as the density near that event is thin, which is Cranmer's adaptive estimate
again. ``options`` carries ROOT's letters: ``"a"`` for the adaptive widths, ``"m"``
to mirror the events at both ends of every range so the estimate does not fall off
there, and the empty string for a fixed-width estimate. ``rho`` scales every width.

Unlike the one-dimensional estimator this does not tabulate the kernels on a grid:
ROOT's table is a speed-up for a shape that changes little between neighbouring
points, and in several dimensions it costs more memory than it saves, so the
product is summed directly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...random import libm
from ..collections import as_list
from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooNDKeysPdf"]


def _spread(values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Each observable's weighted standard deviation, as ROOT accumulates it."""
    total = float(np.sum(weights))
    mean = np.sum(weights[:, None] * values, axis=0) / total
    var = np.sum(weights[:, None] * (values - mean) ** 2, axis=0) / total
    return np.sqrt(np.maximum(var, 1e-30))


class RooNDKeysPdf(RooAbsPdf):
    """The adaptive kernel estimate of several observables' joint density in a dataset."""

    #: ``RooNDKeysPdf``'s option letters.
    Adaptive, Mirror = "a", "m"

    def __init__(
        self,
        name: Any,
        title: Any,
        observables: Any,
        data: Any,
        options: str = "",
        rho: float = 1.0,
    ) -> None:
        super().__init__(name, title)
        self._vars = [
            self._proxy(f"x{index}", one) for index, one in enumerate(as_list(observables))
        ]
        if not self._vars:
            raise ValueError(
                "RooNDKeysPdf needs at least one observable to estimate a density in; "
                "it was given none."
            )
        self._options = str(options)
        self._rho = float(rho)
        self.LoadDataSet(data)

    # -- what it is made of -----------------------------------------------------

    def observables(self) -> list[Any]:
        """The observables the estimate is a density in."""
        return list(self._vars)

    def LoadDataSet(self, data: Any) -> None:
        """Take the events - mirrored as asked - and choose every kernel's width."""
        names = [one.GetName() for one in self._vars]
        values = np.column_stack(
            [np.asarray(data.column(name), dtype=np.float64) for name in names]
        )
        weights = np.asarray(data.weights(), dtype=np.float64)
        total = float(np.sum(weights))
        self._points, self._point_weights = self._mirrored(values, weights)
        self._sum_weights = total
        sphere = _spread(values, weights) / max(total, 1e-30) ** 0.0
        dimension = values.shape[1]
        # Silverman's widths for a Gaussian kernel in `dimension` dimensions, scaled by rho
        scale = (4.0 / (dimension + 2.0)) ** (1.0 / (dimension + 4.0)) * total ** (
            -1.0 / (dimension + 4.0)
        )
        self._widths = np.maximum(sphere * scale * self._rho, 1e-12)
        if self.Adaptive in self._options:
            self._widths = self._adaptive_widths(dimension, total)

    def _mirrored(
        self, values: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The events, and their reflections about every range's ends when mirroring is asked."""
        if self.Mirror not in self._options:
            return values, weights
        copies, copy_weights = [values], [weights]
        for index, one in enumerate(self._vars):
            for edge in (float(one.getMin()), float(one.getMax())):
                reflected = values.copy()
                reflected[:, index] = 2.0 * edge - values[:, index]
                copies.append(reflected)
                copy_weights.append(weights)
        return np.vstack(copies), np.concatenate(copy_weights)

    def _adaptive_widths(self, dimension: int, total: float) -> np.ndarray[Any, Any]:
        """Widen each event's kernel where a fixed-width estimate around it is thin.

        The estimate at an event under the fixed widths says how crowded that part of
        the data is; the width there goes as one over its square root, relative to the
        typical density, which is the shape of ROOT's adaptive rule. The factor is kept
        between a half and two so one sparse event cannot double the estimate's tails.
        """
        fixed = self._widths
        density = self._kernels(self._points, self._points, fixed) @ self._point_weights
        typical = float(np.median(density))
        factor = np.sqrt(np.maximum(typical / np.maximum(density, 1e-300), 1e-12))
        factor = np.clip(factor, 0.5, 2.0)
        floor = fixed * math.sqrt(2.0) / 10.0
        widened = fixed[None, :] * factor[:, None]
        return np.maximum(widened, floor[None, :])

    def _kernels(
        self, at: np.ndarray[Any, Any], points: np.ndarray[Any, Any], widths: Any
    ) -> np.ndarray[Any, Any]:
        """Every kernel's product at each point of ``at``: ``(len(at), len(points))``."""
        width = np.asarray(widths, dtype=np.float64)
        if width.ndim == 1:  # one width per dimension, the same for every event
            width = np.broadcast_to(width, (len(points), len(width)))
        difference = (at[:, None, :] - points[None, :, :]) / width[None, :, :]
        exponential = libm.exp(-0.5 * np.sum(difference * difference, axis=2))
        normal = np.prod(width * math.sqrt(2.0 * math.pi), axis=1)[None, :]
        return exponential / normal

    # -- its value --------------------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        """The mean over the events of each one's kernel product at the point asked about."""
        columns = [np.atleast_1d(one.compute(ctx)) for one in self._vars]
        shape = np.broadcast(*columns).shape
        at = np.column_stack([np.broadcast_to(one, shape) for one in columns])
        kernels = self._kernels(at, self._points, self._widths)
        return kernels @ self._point_weights / max(self._sum_weights, 1e-30)

    def __repr__(self) -> str:
        return (
            f"<RooNDKeysPdf {self.GetName()!r} in {len(self._vars)} dimension(s) "
            f"from {len(self._points)} event(s)>"
        )
