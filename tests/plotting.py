"""What the plotting tests share: headless matplotlib, objects to draw, and a clean slate.

Every backend remembers what it last drew on, for ``SAME``, and one of them
is the backend in use; both are put back after each test, and every figure
pyplot opened is closed, so no test draws on another's axes.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pytest
from matplotlib import pyplot

from xrdroot import Function, Graph, Histogram, open_root
from xrdroot.plot import backends

DATA = __file__.rsplit("/", 1)[0] + "/data"


@pytest.fixture(autouse=True)
def tidy():
    """Close what was drawn and forget what was drawn on, after every test."""
    yield
    pyplot.close("all")
    backends._state["backend"] = "matplotlib"
    backends._state["current"].clear()


def gauss(name: str = "h", seed: int = 1, entries: int = 1000) -> Histogram:
    """A histogram of a thousand normal numbers, titled with its axes as ROOT spells them."""
    made = Histogram.book(name, (20, -3, 3), title=f"{name};x [GeV]")
    made.fill(np.random.default_rng(seed).normal(size=entries))
    return made


def grid() -> Histogram:
    """The 3 by 3 histogram of ``gauss-h2.root``."""
    with open_root(f"{DATA}/gauss-h2.root") as root:
        return root["h2d"]


def cube() -> Histogram:
    """A 3-D histogram with one bin filled."""
    made = Histogram.book("cube", (2, 0, 2), (2, 0, 2), (2, 0, 2))
    made.fill([0.5], [1.5], [0.5])
    return made


def points() -> Graph:
    """A graph with bars of different lengths each side of its points."""
    return Graph.new(
        "g", [1, 2, 3], [2, 3, 1], title="scan", yerr=([0.1, 0.2, 0.1], [0.3, 0.2, 0.1])
    )


def curve() -> Function:
    """A Gaussian over (-3, 3), as ROOT's ``gaus``."""
    return Function("f", "gaus", range=(-3, 3), parameters=[1, 0, 1])


def surface() -> Function:
    """A 2-D Gaussian: a ``TF2``."""
    return Function("f2", "exp(-x*x-y*y)", range=[(-1, 1), (-1, 1)])
