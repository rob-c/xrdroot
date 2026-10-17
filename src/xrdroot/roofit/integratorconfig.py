"""An integral can override its function's settings without changing the function.

Overrides belong to the integrated function, so normalisation integrals of its
children keep their own configurations. Context-local state also keeps nested
evaluations and concurrent callers independent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from .messages import ERROR, log
from .numintconfig import RooNumIntConfig

#: Only explicit integral configurations live here; ordinary functions use their own.
OVERRIDES: ContextVar[dict[int, RooNumIntConfig] | None] = ContextVar(
    "numint_overrides", default=None
)


def configuration(func: Any) -> RooNumIntConfig:
    override = (OVERRIDES.get() or {}).get(id(func))
    return override if override is not None else func.getIntegratorConfig()


@contextmanager
def configured(func: Any, config: RooNumIntConfig) -> Iterator[None]:
    token = OVERRIDES.set({**(OVERRIDES.get() or {}), id(func): config})
    try:
        yield
    finally:
        OVERRIDES.reset(token)


def method(func: Any, dimension: int, opened: bool) -> str:
    config = configuration(func)
    suffix = "Open" if opened else ""
    dimension_name = f"{dimension}D" if dimension < 3 else "ND"
    return str(getattr(config, f"method{dimension_name}{suffix}")().getLabel())


@dataclass(frozen=True)
class RombergSettings:
    """RooIntegrator1D's refinement choices, including its fixed-step mode."""

    max_steps: int = 20
    min_steps: int = 999
    fix_steps: int = 0
    extrapolate: bool = True


#: The refinement settings used before any configuration is supplied.
DEFAULT_ROMBERG = RombergSettings()


def romberg_settings(config: RooNumIntConfig) -> RombergSettings:
    section = config.getConfigSection("RooIntegrator1D")
    maximum = int(section.getRealValue("maxSteps"))
    fixed = int(section.getRealValue("fixSteps"))
    if fixed > maximum:
        log(
            None,
            ERROR,
            "Integration",
            "RooRombergIntegrator::ctor() ERROR: fixSteps>maxSteps, fixSteps set to maxSteps",
        )
        fixed = maximum
    if maximum <= 0:
        maximum = 20 if section.getCatIndex("sumRule") == 0 else 14
    return RombergSettings(
        maximum,
        int(section.getRealValue("minSteps")),
        fixed,
        bool(section.getCatIndex("extrapolation")),
    )
