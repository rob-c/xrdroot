"""``TSpectrum2Fit``: 2-D peaks, each with a ridge in x and in y, fitted to a spectrum.

Each peak has seven parameters - its amplitude and position, then its x and
y ridges' amplitudes and positions - and all share sigma in x and y, the
correlation ``ro``, a background ``a0 + ax*x + ay*y``, and the tails' and
steps' parameters, in ROOT's order. The channels are taken x by x, y within
each, as ROOT's loops take them. Two of ROOT's slips are kept: the x ridges'
step starts from the 2-D step's value, and the volumes' errors read the
sums at the places the shared parameters have among all the parameters,
rather than among the fitted ones (0 where that is past them - memory ROOT
never wrote).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import fit2ders as ders
from . import fit2peaks as peaks2
from .fit import _slope_held
from .fitengine import FitSettings

__all__ = ["PEAK", "SHARED2", "Peak2Model", "Peak2Setup"]

Array = Any

#: A peak's seven parameters, in ROOT's order.
PEAK = ("amp", "x", "y", "ampx", "ampy", "x1", "y1")

#: The parameters the peaks share, in ROOT's order.
SHARED2 = ("sigmax", "sigmay", "ro", "a0", "ax", "ay", "txy", "sxy", "tx", "ty", "sx", "sy",
           "bx", "by")  # fmt: skip

#: Where ROOT's constructor starts the shared parameters: sigmas 2, slopes 1, the rest 0.
START2 = {name: 2.0 if name.startswith("sigma") else 1.0 if name in ("bx", "by") else 0.0
          for name in SHARED2}  # fmt: skip


@dataclass
class Peak2Setup:
    """What ``TSpectrum2Fit``'s setters set: each peak's seven starts and fixes, and the rest's.

    ROOT fits the sigmas and fixes everything else it shares unless told.
    """

    peaks: dict[str, list[float]]
    fix: dict[str, list[bool]]
    init: dict[str, float] = field(default_factory=lambda: dict(START2))
    fixed: dict[str, bool] = field(
        default_factory=lambda: {n: not n.startswith("sigma") for n in SHARED2}
    )


@dataclass
class FitSettings2(FitSettings):
    """``SetFitParameters``' settings, with the y channels too."""

    ymin: int = 0
    ymax: int = 100


def _free(setup: Peak2Setup) -> list[tuple[int, str]]:
    """Each fitted parameter's place among all of them, and what it is."""
    count = len(setup.peaks["amp"])
    free = [(7 * k + n, kind) for k in range(count) for n, kind in enumerate(PEAK)
            if not setup.fix[kind][k]]  # fmt: skip
    return free + [(7 * count + n, name) for n, name in enumerate(SHARED2)
                   if not setup.fixed[name]]  # fmt: skip


def _held(kind: str, value: float, window: tuple[int, int, int, int]) -> float:
    """``value`` kept where ROOT keeps a parameter of this kind as the fit steps."""
    xmin, xmax, ymin, ymax = window
    bounds = {"x": (xmin, xmax), "x1": (xmin, xmax), "y": (ymin, ymax), "y1": (ymin, ymax),
              "ro": (-1, 1)}  # fmt: skip
    if kind in bounds:
        low, high = bounds[kind]
        value = float(low) if value < low else value
        return float(high) if value > high else value
    hold = _HOLDS2.get(kind)
    return hold(value) if hold is not None else value


def _at_least(floor: float) -> Callable[[float], float]:
    return lambda v: floor if v < floor else v


#: How amplitudes, sigmas and slopes are kept in bounds; the rest go free.
_HOLDS2: dict[str, Callable[[float], float]] = {
    "amp": _at_least(0.0), "ampx": _at_least(0.0), "ampy": _at_least(0.0),
    "sigmax": _at_least(0.001), "sigmay": _at_least(0.001),
    "bx": _slope_held, "by": _slope_held,
}  # fmt: skip


class Peak2Model:
    """``TSpectrum2Fit``'s spectrum: its peaks and ridges, their tails and steps, the background."""

    #: The least weight AWMI's likelihood gives a channel.
    likelihood_floor = 0.00001

    def __init__(self, setup: Peak2Setup, settings: FitSettings2) -> None:
        self.count = count = len(setup.peaks["amp"])
        self.param = [0.0] * (7 * count + len(SHARED2))
        for n, kind in enumerate(PEAK):
            self.param[n : 7 * count : 7] = [float(v) for v in setup.peaks[kind]]
        self.param[7 * count :] = [float(setup.init[name]) for name in SHARED2]
        self.param[7 * count + SHARED2.index("sx")] = float(setup.init["sxy"])
        self.free = _free(setup)
        self.start = [float(setup.init[kind]) if index >= 7 * count else self.param[index]
                      for index, kind in self.free]  # fmt: skip
        self.window = (settings.xmin, settings.xmax, settings.ymin, settings.ymax)
        xs = np.arange(settings.xmin, settings.xmax + 1, dtype=np.float64)
        ys = np.arange(settings.ymin, settings.ymax + 1, dtype=np.float64)
        self.x, self.y = np.repeat(xs, len(ys)), np.tile(ys, len(xs))

    def shared(self) -> dict[str, float]:
        """The parameters the peaks share, by name."""
        return dict(zip(SHARED2, self.param[7 * self.count :], strict=False))

    def peaks(self) -> peaks2.Peaks2:
        """Every peak's seven parameters."""
        return peaks2.Peaks2(*(np.array(self.param[n : 7 * self.count : 7]) for n in range(7)))

    def shape(self) -> Array:
        return peaks2.shape2(self.x, self.y, self.peaks(), self.shared())

    def derivatives(self) -> Array:
        found = {kind: _DERIVATIVES2[kind](self) for kind in {kind for _, kind in self.free}}
        rows = [_pick(found[kind], index, kind) for index, kind in self.free]
        return np.array(rows).reshape(len(rows), len(self.x))

    def second(self) -> Array:
        found = {kind: _SECONDS[kind](self) for kind in {k for _, k in self.free} & set(_SECONDS)}
        zero = np.zeros_like(self.x)
        rows = [_pick(found[kind], index, kind) if kind in found else zero
                for index, kind in self.free]  # fmt: skip
        return np.array(rows).reshape(len(rows), len(self.x))

    def stale(self) -> Array:
        return np.array([kind == "ro" for _, kind in self.free], dtype=bool)

    def taylored(self) -> Array:
        wanted = ("x", "y", "x1", "y1", "sigmax", "sigmay", "ro")
        return np.array([kind in wanted for _, kind in self.free], dtype=bool)

    def apply(self, xk: list[float]) -> None:
        for n, (index, kind) in enumerate(self.free):
            xk[n] = _held(kind, xk[n], self.window)
            self.param[index] = xk[n]


def _pick(rows: Array, index: int, kind: str) -> Array:
    """A peak's own row of a per-peak derivative; a shared parameter's is the whole."""
    return rows[index // 7] if kind in PEAK else rows


def _ridge_amp(m: Peak2Model, by_x: bool) -> Array:
    """``Derampx`` along x, or along y."""
    q, p = m.shared(), m.peaks()
    if by_x:
        return peaks2.derampx(m.x, p.x1, q["sigmax"], q["tx"], q["sx"], q["bx"])
    return peaks2.derampx(m.y, p.y1, q["sigmay"], q["ty"], q["sy"], q["by"])


def _ridge_pos(m: Peak2Model, by_x: bool) -> Array:
    """``Deri01`` along x, or along y."""
    q, p = m.shared(), m.peaks()
    if by_x:
        return peaks2.deri01(m.x, p.ampx, p.x1, q["sigmax"], q["tx"], q["sx"], q["bx"])
    return peaks2.deri01(m.y, p.ampy, p.y1, q["sigmay"], q["ty"], q["sy"], q["by"])


def _ridge_pos2(m: Peak2Model, by_x: bool) -> Array:
    """``Derderi01`` along x, or along y."""
    q, p = m.shared(), m.peaks()
    if by_x:
        return peaks2.derderi01(m.x, p.ampx, p.x1, q["sigmax"])
    return peaks2.derderi01(m.y, p.ampy, p.y1, q["sigmay"])


def _ridge_tail(m: Peak2Model, by_x: bool) -> Array:
    """``Dertx`` or ``Derty``."""
    q, p = m.shared(), m.peaks()
    if by_x:
        return ders.dertx(m.x, p.ampx, p.x1, q["sigmax"], q["bx"])
    return ders.dertx(m.y, p.ampy, p.y1, q["sigmay"], q["by"])


def _ridge_step(m: Peak2Model, by_x: bool) -> Array:
    """``Dersx`` or ``Dersy``."""
    p = m.peaks()
    if by_x:
        return ders.dersx(m.x, p.ampx, p.x1, m.shared()["sigmax"])
    return ders.dersx(m.y, p.ampy, p.y1, m.shared()["sigmay"])


def _whole(function: Callable[..., Array], **options: bool) -> Callable[[Peak2Model], Array]:
    """A derivative of all the channels and peaks at once, from the model's parameters."""
    return lambda m: function(m.x, m.y, m.peaks(), m.shared(), **options)


#: Each kind of parameter's derivative, ``TSpectrum2Fit``'s ``Deramp2`` to ``Derby``.
_DERIVATIVES2: dict[str, Callable[[Peak2Model], Array]] = {
    "amp": _whole(peaks2.deramp2),
    "x": _whole(peaks2.deri02, by_x=True),
    "y": _whole(peaks2.deri02, by_x=False),
    "ampx": lambda m: _ridge_amp(m, True),
    "ampy": lambda m: _ridge_amp(m, False),
    "x1": lambda m: _ridge_pos(m, True),
    "y1": lambda m: _ridge_pos(m, False),
    "sigmax": _whole(ders.dersigmax, by_x=True),
    "sigmay": _whole(ders.dersigmax, by_x=False),
    "ro": _whole(ders.derro),
    "a0": lambda m: np.ones_like(m.x),
    "ax": lambda m: m.x,
    "ay": lambda m: m.y,
    "txy": _whole(ders.dertxy),
    "sxy": _whole(ders.dersxy),
    "tx": lambda m: _ridge_tail(m, True),
    "ty": lambda m: _ridge_tail(m, False),
    "sx": lambda m: _ridge_step(m, True),
    "sy": lambda m: _ridge_step(m, False),
    "bx": _whole(ders.derbx, by_x=True),
    "by": _whole(ders.derbx, by_x=False),
}

#: The second derivatives a Taylor step takes, where ROOT has them.
_SECONDS: dict[str, Callable[[Peak2Model], Array]] = {
    "x": _whole(peaks2.derderi02, by_x=True),
    "y": _whole(peaks2.derderi02, by_x=False),
    "x1": lambda m: _ridge_pos2(m, True),
    "y1": lambda m: _ridge_pos2(m, False),
    "sigmax": _whole(ders.derdersigmax, by_x=True),
    "sigmay": _whole(ders.derdersigmax, by_x=False),
}
