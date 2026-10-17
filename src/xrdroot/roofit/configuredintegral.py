"""Dispatch one-dimensional numerical integration through RooFit's configuration."""

from __future__ import annotations

from functools import partial
from math import sqrt
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .integratorconfig import configuration, method, romberg_settings


def integrate(
    func: Any,
    along: Any,
    low: float,
    high: float,
    selected: str | None = None,
) -> Any:
    from .integration import integrate_1d, romberg

    config = configuration(func)
    opened = bool(np.any(np.isinf(low)) or np.any(np.isinf(high)))
    selected = selected or method(func, 1, opened)
    if selected == "RooAdaptiveGaussKronrodIntegrator1D":
        return _kronrod(along, low, high, config)
    if selected not in (
        "RooIntegrator1D",
        "RooImproperIntegrator1D",
        "RooIntegrator2D",
        "RooSegmentedIntegrator1D",
        "RooSegmentedIntegrator2D",
    ):
        raise UnsupportedFeatureError(
            f"Numerical integration with {selected} is not implemented. "
            "Select RooIntegrator1D for finite one-dimensional integrals."
        )
    settings = romberg_settings(config)
    section = config.getConfigSection("RooIntegrator1D")
    run = partial(
        romberg,
        func=along,
        trapezoid=section.getCatIndex("sumRule") == 0,
        eps_abs=_precision(config.epsAbs()),
        eps_rel=_precision(config.epsRel()),
        name=func.GetName(),
        settings=settings,
    )
    if selected in ("RooSegmentedIntegrator1D", "RooSegmentedIntegrator2D"):
        return _segmented(run, low, high, config)
    return integrate_1d(
        along,
        low,
        high,
        func.GetName(),
        trapezoid=section.getCatIndex("sumRule") == 0,
        eps_abs=_precision(config.epsAbs()),
        eps_rel=_precision(config.epsRel()),
        settings=settings,
    )


def _precision(value: float) -> float:
    return 1e-6 if value <= 0 else value


def _segmented(run: Any, low: float, high: float, config: Any) -> Any:
    section = config.getConfigSection("RooSegmentedIntegrator1D")
    count = int(section.getRealValue("numSeg"))
    width = (high - low) / count
    scale = float(sqrt(count))
    total: Any = 0.0
    for index in range(count):
        total += run(
            low=low + index * width,
            high=low + (index + 1) * width,
            eps_abs=_precision(config.epsAbs() / scale),
            eps_rel=_precision(config.epsRel() / scale),
        )
    return total


def _kronrod(along: Any, low: float, high: float, config: Any) -> float:
    from .kronrod import integrate as adaptive

    section = config.getConfigSection("RooAdaptiveGaussKronrodIntegrator1D")
    key = section.getCatIndex("method")
    if key == 0 or np.isinf(low) or np.isinf(high):
        raise UnsupportedFeatureError(
            "Wynn-epsilon Gauss-Kronrod integration is not implemented. "
            "Select a points rule on a finite interval or RooImproperIntegrator1D for open ends."
        )
    points = (15, 21, 31, 41, 51, 61)[key - 1]
    # RooAdaptiveGaussKronrodIntegrator1D's constructor exchanges the two
    # tolerance fields; retain that ordering when the requested values differ.
    return adaptive(
        along,
        low,
        high,
        config.epsRel(),
        config.epsAbs(),
        int(section.getRealValue("maxSeg")),
        points,
    ).value
