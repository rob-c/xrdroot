"""Sampler choices are read after the numeric-generator factory initializes defaults."""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError


def configuration(pdf: Any) -> Any:
    from ..numgenconfig import initialize_factory

    initialize_factory()
    return pdf.getGeneratorConfig()


def method(pdf: Any, names: frozenset[str], conditional: bool) -> str:
    config = configuration(pdf)
    variables = [one for one in pdf.leaves() if one.GetName() in names]
    real = sum(not hasattr(one, "lookupIndex") for one in variables)
    categories = any(hasattr(one, "lookupIndex") for one in variables)
    dimension = "1D" if real <= 1 else "2D" if real == 2 else "ND"
    return str(getattr(config, "method" + dimension)(conditional, categories).getLabel())


def foam_options(pdf: Any, dimension: int) -> dict[str, int]:
    section = configuration(pdf).getConfigSection("RooFoamGenerator")
    key = f"nCell{dimension}D" if dimension <= 3 else "nCellND"
    return {
        "n_cells": int(section.getRealValue(key)),
        "n_sample": int(section.getRealValue("nSample")),
        "chat_level": int(section.getRealValue("chatLevel")),
    }


def trial_counts(pdf: Any) -> tuple[int, ...]:
    section = configuration(pdf).getConfigSection("RooAcceptReject")
    return tuple(int(section.getRealValue(f"nTrial{dimension}D")) for dimension in range(4))


class NoObservables:
    """A component with nothing to draw for: it contributes no values, only a count.

    ROOT's Foam needs a dimension to explore, and a component whose observables are
    all conditional on the rest of the model has none - so it is drawn as the empty
    event it is, rather than sent to a generator that could only refuse it.
    """

    #: Nothing is rejected when there is nothing to draw.
    resample_ratio = 1.0

    def generate(self, count: int) -> Any:
        import numpy as np

        return np.zeros((int(count), 0), dtype=np.float64)

    def __repr__(self) -> str:
        return "<no observables to draw>"


def sampler(pdf: Any, names: frozenset[str], density: Any, variables: list[Any], rng: Any) -> Any:
    from .empirical import EmpiricalSampler
    from .foam import FoamGenerator

    if not variables:
        return NoObservables()
    selected = method(pdf, names, False)
    if selected == "RooFoamGenerator":
        ranges = [(one.getMin(), one.getMax()) for one in variables]
        return FoamGenerator(
            density,
            ranges,
            rng,
            vectorized=True,
            **foam_options(pdf, len(variables)),
        )
    if selected == "RooAcceptReject":
        from .acceptreject import minimum_trials

        count = minimum_trials(variables, trial_counts(pdf), f"{pdf.GetName()}_AccRej")
        return EmpiricalSampler(density, variables, rng, count)
    if selected == "N/A":
        raise ValueError(
            "No numerical event generator was selected in this configuration. "
            "Copy the initialized default configuration and select a sampler."
        )
    raise UnsupportedFeatureError(
        f"Numerical event generation with {selected} is not implemented. "
        "Select RooFoamGenerator or RooAcceptReject in the generation configuration."
    )
