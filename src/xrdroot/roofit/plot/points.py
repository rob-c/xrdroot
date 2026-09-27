"""``TGraph``'s accessors for RooFit's curves and points: ``GetPointX(i)``, ``GetErrorYlow(i)``.

A ``RooCurve`` and a ``RooHist`` are graphs in ROOT, and scripts read their points back as
a graph's; these are the few calls they make, over the engine's own graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["GraphAccess"]


class GraphAccess:
    """``GetN``, ``GetPointX``/``Y`` and the four error accessors, point by point."""

    x: Any
    y: Any
    members: Any

    def GetN(self) -> int:
        return len(self.x)

    def GetPointX(self, i: int) -> float:
        return float(self.x[i])

    def GetPointY(self, i: int) -> float:
        return float(self.y[i])

    def _error(self, key: str, i: int) -> float:
        found = self.members.get(key)
        return float(np.asarray(found)[i]) if found is not None else 0.0

    def GetErrorXlow(self, i: int) -> float:
        return self._error("fEXlow", i)

    def GetErrorXhigh(self, i: int) -> float:
        return self._error("fEXhigh", i)

    def GetErrorYlow(self, i: int) -> float:
        return self._error("fEYlow", i)

    def GetErrorYhigh(self, i: int) -> float:
        return self._error("fEYhigh", i)
