"""What the tests of ``xrdroot.pyroot.graphics`` share: a clean session, and wrapped data.

Each test starts as a fresh ROOT session would: no canvas, ``gStyle`` the
``Modern`` style, no colour made and no PDF open. The core module's
wrappers are not on this branch's path, so :class:`Wrapped` stands in for
one - anything with an ``_xrd`` is drawn as the object it wraps.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from xrdroot import Graph, Histogram
from xrdroot.pyroot.core import draw_hook


class Wrapped:
    """A core wrapper as far as drawing needs one: ``_xrd``, its name and title, ``Draw``."""

    def __init__(self, xrd: Any) -> None:
        self._xrd = xrd

    def GetName(self) -> str:
        return str(self._xrd.name)

    def GetTitle(self) -> str:
        return str(self._xrd.title)

    def Draw(self, option: str = "") -> None:
        draw_hook(self, option)


def gaussian(name: str = "h", title: str = "A Gaussian", n: int = 1000) -> Wrapped:
    h = Histogram.book(name, (40, -4.0, 4.0), title=title)
    h.fill(np.random.default_rng(7).normal(size=n))
    return Wrapped(h)


def graph(name: str = "g", title: str = "A graph") -> Wrapped:
    return Wrapped(Graph.new(name, [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 3.0, 5.0], title=title))


@pytest.fixture(autouse=True)
def fresh_session():
    """No canvas, the Modern style, no colours made, no book open; warnings are errors."""
    from xrdroot.pyroot.graphics import colors, output, pads, style

    pads.CANVASES.clear()
    pads.set_current(None)
    style._MADE.clear()
    style._CURRENT[0] = style.TStyle("Modern")
    colors.MADE.clear()
    colors.ALPHA.clear()
    colors.OBJECTS.clear()
    colors.LAID.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
    output.close_books()
