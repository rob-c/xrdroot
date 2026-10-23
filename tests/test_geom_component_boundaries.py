"""Geometry keeps shared arithmetic and compatibility paths without importing renderers."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from xrdroot import colors
from xrdroot.canvas import colors as canvas_colors
from xrdroot.canvas import geompaint, view3d
from xrdroot.geom import backends, paint, projection, scene, view
from xrdroot.plot import colors as plot_colors


def test_legacy_painter_names_are_the_exact_graphics_objects():
    assert paint.points_of is geompaint.points_of is scene.points_of
    assert paint.extent is scene.extent and paint.make_view is view.make_view
    for name in paint.__all__:
        assert getattr(paint, name) is getattr(geompaint, name)
    assert paint._view is geompaint._view and paint._pixels is geompaint._pixels
    assert set(paint.__all__) <= set(dir(paint))
    with pytest.raises(AttributeError, match="has no attribute 'unknown'"):
        paint.__getattr__("unknown")


def test_canvas_projection_names_reexport_the_single_geometry_implementation():
    for name in ("View3D", "Corners", "Angles", "RAD", "CORNER_ORDER", "AXIS_ENDS",
                 "_translation", "_rotation", "_direction"):
        assert getattr(view3d, name) is getattr(projection, name)
    assert view3d.__all__ == ["View3D"]


def test_plot_colours_reexport_the_shared_tables_and_functions():
    for name in plot_colors.__all__:
        assert getattr(plot_colors, name) is getattr(colors, name)
    assert canvas_colors._rgb is colors._rgb
    assert canvas_colors.RGB is colors.RGB


def test_geometry_and_canvas_see_the_same_modified_colour_table(monkeypatch):
    monkeypatch.setitem(colors.COLORS, 10000, "#123456")
    expected = (0x12 / 255, 0x34 / 255, 0x56 / 255)
    assert backends._rgb(10000) == canvas_colors.Colors().rgb(10000) == expected
    assert plot_colors.color(10000) == "#123456"
    assert colors.rgb(-1) == (0.0, 0.0, 0.0)


def test_geometry_models_and_shared_arithmetic_import_without_renderers():
    script = """
import importlib.abc
import sys

class BlockRenderers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('xrdroot.canvas', 'xrdroot.plot', 'matplotlib', 'plotly', 'pyvista')
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError('Renderer import blocked: ' + fullname)
        return None

sys.meta_path.insert(0, BlockRenderers())
from xrdroot.colors import rgb
from xrdroot.geom import Solid
from xrdroot.geom import backends, shapes
from xrdroot.geom.scene import extent
from xrdroot.geom.view import ParallelView
solid = Solid(shapes.box(1, 2, 3))
low, high = extent([solid])
assert low.tolist() == [-1.0, -2.0, -3.0]
assert ParallelView(low, high).project([[0, 0, 0]]).tolist() == [[0.0, 0.0]]
assert backends._rgb(2) == rgb(2) == (1.0, 0.0, 0.0)
print('geometry independent')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "geometry independent"
