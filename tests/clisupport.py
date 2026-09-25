"""A stand-in matplotlib for the command-line tests, which CI runs without the real one.

``print`` and ``draw`` ask an object to ``.plot()`` itself and save the figure
its axes are on; what is under test is that they ask, and save where told,
not matplotlib's drawing - which ``tests/test_root_draw.py`` looks at.
"""

from __future__ import annotations

import sys
import types
from typing import Any


class Figure:
    """Enough of a figure to be saved and closed."""

    def __init__(self, saved: list[Any], style: dict[str, Any]) -> None:
        self.saved = saved
        self.style = style

    def savefig(self, path: str) -> None:
        self.saved.append((path, self.style))


class FakeMatplotlib:
    """``matplotlib`` and ``matplotlib.pyplot``, as far as the command line uses them."""

    def __init__(self) -> None:
        self.backends: list[str] = []
        self.closed: list[Any] = []
        self.module = types.ModuleType("matplotlib")
        self.module.use = self.backends.append  # type: ignore[attr-defined]
        self.pyplot = types.ModuleType("matplotlib.pyplot")
        self.pyplot.close = self.closed.append  # type: ignore[attr-defined]
        self.module.pyplot = self.pyplot  # type: ignore[attr-defined]

    def install(self, monkeypatch: Any) -> FakeMatplotlib:
        monkeypatch.setitem(sys.modules, "matplotlib", self.module)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", self.pyplot)
        return self

    @staticmethod
    def axes(saved: list[Any], style: dict[str, Any]) -> Any:
        """Axes whose figure writes down where it was saved, and how it was drawn."""
        return types.SimpleNamespace(figure=Figure(saved, style))
