"""``TGraphTime``: a graph that keeps snapshots and draws them one after another.

ROOT's ``TGraphTime`` (``TGraphTime.h``) is built with a number of steps and a frame -
``TGraphTime(nsteps, xmin, ymin, xmax, ymax)`` - and collects points per step with
``AddPoint``, ``Paint`` drawing each step in turn. It is a drawing object with no
streamer of its own worth writing here, so it keeps its snapshot list in Python.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar

from .drawn import Drawn

__all__ = ["TGraphTime"]


class TGraphTime(Drawn):
    """Points added a step at a time, drawn step by step."""

    classname = "TGraphTime"
    groups: ClassVar[tuple[str, ...]] = ("line", "marker")
    fields: ClassVar[dict[str, type]] = {"Nsteps": int}

    def __init__(
        self,
        nsteps: int = 0,
        xmin: float = 0.0,
        ymin: float = 0.0,
        xmax: float = 1.0,
        ymax: float = 1.0,
    ) -> None:
        super().__init__()
        self.members.update(fNsteps=int(nsteps))
        self._frame = (float(xmin), float(ymin), float(xmax), float(ymax))
        self._steps: list[list[tuple[float, float]]] = [[] for _ in range(int(nsteps))]
        self._current = -1

    def AddPoint(self, x: float, y: float, option: str = "") -> None:
        """A point in the step now being filled; ``option`` names the step when it moves on."""
        if option:
            self._current += 1
        if not self._steps:
            self._steps.append([])
        if self._current < 0:
            self._current = 0
        while self._current >= len(self._steps):
            self._steps.append([])
        self._steps[self._current].append((float(x), float(y)))

    def TakeTime(self, x: float, y: float, option: str = "") -> None:
        """``TakeTime`` is ``AddPoint`` with ROOT's name for moving to the next step."""
        self.AddPoint(x, y, option)

    def GetNsteps(self) -> int:
        return int(self.members["fNsteps"])

    def GetStep(self, step: int) -> list[tuple[float, float]]:
        """The points of one step, which is what ``Paint`` draws."""
        return list(self._steps[step]) if 0 <= step < len(self._steps) else []

    def GetFrame(self) -> tuple[float, float, float, float]:
        return self._frame

    def Paint(self, option: str = "") -> Any:
        """Each step's points in turn, as ROOT's ``Paint`` walks them."""
        from .shapes import TLine

        for step in self._steps:
            for (x1, y1), (x2, y2) in itertools.pairwise(step):
                TLine(x1, y1, x2, y2).Draw(option)
        return self
