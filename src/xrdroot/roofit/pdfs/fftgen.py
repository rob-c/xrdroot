"""How a ``RooFFTConvPdf`` draws its events: RooFit's two generator contexts for it.

When both densities draw the convolution variable themselves,
``RooConvGenContext`` draws an event of each - the second first - with the
variable's range opened for them, adds the two, and keeps the sum if it is
inside the range, drawing again if not. Otherwise ``RooGenContext`` samples
the convolution numerically, from a copy of it made to generate with - two
copies are made, as RooFit makes them, each announcing its cache.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from .. import copies
from ..generation.contexts import Context, context_for
from ..rng import generator

__all__ = ["ConvolutionContext", "SampledContext"]


@contextmanager
def _opened(var: Any) -> Iterator[None]:
    """``removeMin()`` and ``removeMax()`` on the generator's copy of ``var``: its range opened."""
    binning = var.getBinning()
    saved = binning.lowBound(), binning.highBound()
    binning.setRange(-np.inf, np.inf)
    try:
        yield
    finally:
        binning.setRange(*saved)


class ConvolutionContext(Context):
    """``RooConvGenContext``: an event of each density, added."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        super().__init__(pdf, names)
        with _opened(pdf.x):
            self.first = context_for(pdf.pdf1, names)
            self.second = context_for(pdf.pdf2, names)

    def event(self, remaining: int) -> dict[str, float]:
        x = self.pdf.x
        name = x.GetName()
        while True:
            with _opened(x):
                smear = self.second.event(remaining)[name]
                value = self.first.event(remaining)[name]
            if x.can_hold(value + smear):
                return {name: value + smear}


class SampledContext(Context):
    """``RooGenContext`` with TFoam: the normalised convolution sampled over its observables."""

    def __init__(self, pdf: Any, names: frozenset[str]) -> None:
        from ..generation.foam import FoamGenerator

        super().__init__(pdf, names)
        self.states = [copies.copies_of(pdf, "generate", names) for _ in range(2)][-1]
        self.order = [one for one in pdf.leaves() if one.GetName() in names]
        ranges = [(one.getMin(), one.getMax()) for one in self.order]
        keys = [one.GetName() for one in self.order]

        def density(points: Any) -> Any:
            ctx = {key: points[:, i] for i, key in enumerate(keys)}
            return np.broadcast_to(pdf.value(ctx, names), (len(points),))

        with copies.within(self.states):
            self.sampler = FoamGenerator(density, ranges, generator(), vectorized=True)

    def event(self, remaining: int) -> dict[str, float]:
        with copies.within(self.states):
            point = self.sampler.generate()
        return {one.GetName(): float(v) for one, v in zip(self.order, point, strict=False)}
