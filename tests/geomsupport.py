"""What the tests of ``xrdroot.pyroot.geom`` share: a session with no geometry, and pictures.

Each test starts as a fresh ROOT would: no ``gGeoManager``, no ``gGeometry``,
no matrix registered, no canvas, in a directory of its own. A picture is
compared with ROOT's own, saved beside the test data, pixel for pixel: the
wireframes are drawn in whole pixels by xrdroot's own rasteriser, so they
are the same on every machine.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh

#: Where the pictures and files ROOT made are.
DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def geometry_session(tmp_path: Any) -> Iterator[None]:
    """No geometry, old or new, no registered matrix, no canvas, a directory of its own."""
    from xrdroot.pyroot.geom import legacy, manager, matrices
    from xrdroot.pyroot.graphics import pads

    manager._CURRENT.clear()
    manager._SETTINGS.update(verbose=1, precision=17)
    legacy._CURRENT.clear()
    matrices.REGISTERED.clear()
    pads.CANVASES.clear()
    pads.set_current(None)
    ROOT.gROOT.GetListOfGeometries().Clear()
    from xrdroot.pyroot.graphics import hook

    session = fresh(tmp_path)
    next(session)
    hook.install()  # drawing puts things in pads, as the graphics make it
    yield
    next(session, None)
    ROOT.gROOT.GetListOfGeometries().Clear()


def pixels(path: Any) -> np.ndarray[Any, Any]:
    from matplotlib.image import imread

    return np.asarray(imread(str(path)))[..., :3]


def differing(ours: Any, roots: str) -> float:
    """The fraction of pixels in ``ours`` that are not ROOT's picture's ``roots``."""
    mine, theirs = pixels(ours), pixels(DATA / roots)
    assert mine.shape == theirs.shape
    return float(np.mean(np.any(mine != theirs, axis=-1)))
