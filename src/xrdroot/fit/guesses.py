"""Where a fit of a built-in shape starts: ``InitGaus``, ``Init2DGaus`` and ``InitExpo``.

``h->Fit("gaus")`` works without being told where the peak is because
``HFit::Fit`` looks at the points first: for a Gaussian - or a Landau, which
it treats the same way - the mean and RMS of the points weighted by their
values, and a height halfway between the largest value and what a Gaussian
of that RMS and area would have; for an exponential, the line through the
logarithms of the values at the two ends. It also bounds the Gaussian's
width to between zero and ten RMS, and that bound stays on the function
afterwards. Option ``B`` skips all of this and starts from the parameters
and limits the function already has.

These are ROOT's routines, statement for statement, over the fit's points
in the order ``FillData`` made them: the smallest gap between successive
points is the "bin width", as ROOT takes it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .data import FitData

__all__ = ["initial"]

#: ``sqrt(2 pi)`` as ``InitGaus`` writes it, to seven figures.
SQRT_PI = 2.506628
#: ``TFormula::fNumber`` of the shapes that are started from the data.
GAUSSIAN, GAUSSIAN_2D, BIGAUSSIAN, EXPONENTIAL, LANDAU, LANDAU_2D = 100, 110, 112, 200, 400, 410


def _moments(x: Any, values: Any) -> tuple[float, float, float, float] | None:
    """The weighted sums ``InitGaus`` makes: the total, the mean, the RMS and the finest step."""
    n = len(x)
    rangex = float(x[-1] - x[0])
    width = rangex if rangex > 0 else 1.0
    sumx = sumx2 = allcha = 0.0
    for i in range(n):
        val, xi = float(values[i]), float(x[i])
        sumx += val * xi
        sumx2 += val * xi * xi
        allcha += val
        if i > 0 and xi - float(x[i - 1]) < width:
            width = xi - float(x[i - 1])
    if allcha <= 0:
        return None
    mean = sumx / allcha
    rms = sumx2 / allcha - mean * mean
    rms = math.sqrt(rms) if rms > 0 else width * n / 4
    return allcha, mean, rms, width


def init_gaus(data: FitData, function: Any) -> None:
    """``InitGaus``: a Gaussian's height, mean and width from the points' moments."""
    if not data.size:
        return
    found = _moments(data.x[:, 0], data.y)
    if found is None:
        return
    allcha, mean, rms, width = found
    valmax = max(0.0, float(np.max(data.y)))
    constant = 0.5 * (valmax + width * allcha / (SQRT_PI * rms))
    function.set_parameters(constant, mean, rms)
    function.set_limits(2, 0.0, 10 * rms)


def init_2d_gaus(data: FitData, function: Any) -> None:
    """``Init2DGaus``: the same along x and y, the height the product of the two."""
    if not data.size:
        return
    along_x = _moments(data.x[:, 0], data.y)
    along_y = _moments(data.x[:, 1], data.y)
    if along_x is None or along_y is None:
        return
    allcha, meanx, rmsx, widthx = along_x
    _, meany, rmsy, widthy = along_y
    valmax = max(0.0, float(np.max(data.y)))
    constant = (
        0.5
        * (valmax + widthx * allcha / (SQRT_PI * rmsx))
        * (valmax + widthy * allcha / (SQRT_PI * rmsy))
    )
    function.set_parameters(constant, meanx, rmsx, meany, rmsy)
    function.set_limits(2, 0.0, 10 * rmsx)
    function.set_limits(4, 0.0, 10 * rmsy)


def init_expo(data: FitData, function: Any) -> None:
    """``InitExpo``: the exponential through the values at the lowest and highest x."""
    if not data.size:
        return
    x, values = data.x[:, 0], data.y
    low = high = 0
    for i in range(1, data.size):
        if x[i] < x[low]:
            low = i
        elif x[i] > x[high]:
            high = i
    xmin, xmax = float(x[low]), float(x[high])
    vmin, vmax = float(values[low]), float(values[high])
    if vmin <= 0 < vmax:
        vmin = vmax
    elif vmax <= 0 < vmin:
        vmax = vmin
    elif vmin <= 0 and vmax <= 0:
        vmin = vmax = 1.0
    slope = math.log(vmax / vmin) / (xmax - xmin)
    function.set_parameters(math.log(vmin) - slope * xmin, slope)


#: The starting guess for each shape ROOT starts from the data.
INITIALISERS = {
    GAUSSIAN: init_gaus,
    LANDAU: init_gaus,
    GAUSSIAN_2D: init_2d_gaus,
    BIGAUSSIAN: init_2d_gaus,
    LANDAU_2D: init_2d_gaus,
    EXPONENTIAL: init_expo,
}


def initial(data: FitData, function: Any, number: int) -> None:
    """Start a built-in shape where ``HFit::Fit`` starts it, or leave it as it is."""
    initialiser = INITIALISERS.get(number)
    if initialiser is not None:
        initialiser(data, function)
