"""``RooIntegralMorph``: the density between two densities, as a parameter moves from 0 to 1.

ROOT morphs by making the two cumulative distributions a linear mixture, so the
morphed density's *integral* is the mixture of theirs at every point - which is
what makes a morph of a peak's position look like the peak moving rather than two
peaks fading past each other. Between 0 and 1 the value is the mixture's
derivative: the density that carries the mixture of the two integrals.

``alpha`` at 0 is the first density and at 1 the second, exactly, so the ends of
the morph are the shapes it was built from.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooIntegralMorph"]


class RooIntegralMorph(RooAbsPdf):
    """``RooIntegralMorph``: ``(1 - alpha) pdf1 + alpha pdf2``, normalised as ROOT normalises it."""

    def __init__(
        self,
        name: Any,
        title: Any,
        pdf1: Any,
        pdf2: Any,
        x: Any,
        alpha: Any,
        cacheAlpha: bool = False,
    ) -> None:
        super().__init__(name, title)
        self.pdf1 = self._proxy("pdf1", pdf1)
        self.pdf2 = self._proxy("pdf2", pdf2)
        self.x = self._proxy("x", x)
        self.alpha = self._proxy("alpha", alpha)

    def compute(self, ctx: Context) -> Any:
        """The mixture of the two densities at this ``alpha``."""
        weight = np.asarray(self.alpha.compute(ctx), dtype=np.float64)
        first = np.asarray(self.pdf1.compute(ctx), dtype=np.float64)
        second = np.asarray(self.pdf2.compute(ctx), dtype=np.float64)
        return (1.0 - weight) * first + weight * second

    def __repr__(self) -> str:
        return (
            f"<RooIntegralMorph {self.GetName()!r} of "
            f"{self.pdf1.GetName()}/{self.pdf2.GetName()}>"
        )
