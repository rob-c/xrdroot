"""The rest of ``xrdroot.geom``: views, Boolean expressions, ``.obj`` files, 3-D libraries.

What a view does to a point is checked against ``TView3D``'s own
arithmetic worked by hand; the pictures it makes are checked, pixel for
pixel, against ROOT's in ``test_pyroot_geom``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.geom import Solid, backends, composite, wavefront
from xrdroot.geom import shapes as build
from xrdroot.geom.scene import extent
from xrdroot.geom.view import ParallelView, PerspectiveView, make_view


def test_a_perspective_view_puts_the_middle_of_its_range_in_the_middle_of_the_pad() -> None:
    view = PerspectiveView([-1, -1, -1], [1, 1, 1])
    assert view.project([[0.0, 0.0, 0.0]])[0] == pytest.approx([0.0, 0.0], abs=1e-12)


def across_and_deep(tnorm: Any) -> tuple[Any, Any]:
    """A direction across the screen, and the one straight into it, from a view's matrix."""
    across = np.asarray(tnorm[0][:3], dtype=float)
    deep = np.cross(tnorm[0][:3], tnorm[1][:3])
    return across / np.linalg.norm(across), deep / np.linalg.norm(deep)


def test_a_perspective_view_shrinks_what_is_further_away() -> None:
    view = PerspectiveView([-1, -1, -1], [1, 1, 1])
    across, deep = across_and_deep(view.tnorm)
    near, far = view.project([0.3 * across - 0.5 * deep, 0.3 * across + 0.5 * deep])
    assert abs(near[0]) != pytest.approx(abs(far[0]))


def test_a_point_behind_the_eye_is_thrown_far_off_the_pad() -> None:
    view = PerspectiveView([-1, -1, -1], [1, 1, 1], longitude=0.0, latitude=0.0)
    behind = view.project([[0.0, 100.0, 0.0]])[0]
    assert abs(behind).max() > 10


def test_a_pad_s_angles_are_the_view_s_longitude_and_latitude() -> None:
    view = PerspectiveView.from_pad([0, 0, 0], [1, 1, 1], theta=30.0, phi=30.0, aspect=0.5)
    assert (view.longitude, view.latitude, view.window[1] / view.window[0]) == (-120.0, 60.0, 0.5)


def test_a_view_of_nothing_still_has_a_size() -> None:
    assert PerspectiveView([0, 0, 0], [0, 0, 0]).dview == 3.0


def test_a_parallel_view_keeps_what_is_further_away_the_same_size() -> None:
    view = ParallelView([-1, -1, -1], [1, 1, 1])
    across, deep = across_and_deep(view.tnorm)
    near, far = view.project([0.3 * across - 0.5 * deep, 0.3 * across + 0.5 * deep])
    assert near == pytest.approx(far)


def test_a_view_is_chosen_and_ranged_by_its_settings() -> None:
    auto = make_view({}, [0, 0, 0], [2, 2, 2], 1.0)
    fixed = make_view({"perspective": False, "rmin": [0, 0, 0], "rmax": [4, 4, 4]}, None, None, 1.0)
    assert isinstance(auto, PerspectiveView) and list(auto.rmax) == [2.0, 2.0, 2.0]
    assert isinstance(fixed, ParallelView)


def test_the_range_of_nothing_painted_is_the_unit_box_either_side() -> None:
    assert [list(side) for side in extent([])] == [[-1.0] * 3, [1.0] * 3]
    low, high = extent([Solid(build.box(1, 2, 3))])
    assert (list(low), list(high)) == ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])


def test_a_composite_expression_reads_as_root_reads_it() -> None:
    tree = composite.parse("(A:t1+B:t2)-C*D")
    assert tree == ("-", ("+", ("A", "t1"), ("B", "t2")), ("*", ("C", ""), ("D", "")))
    assert composite.leaves(tree) == [("A", "t1"), ("B", "t2"), ("C", ""), ("D", "")]


@pytest.mark.parametrize(
    ("expression", "said"),
    [("A+", "ends too soon"), ("(A+B C", "unclosed parenthesis"), ("A+-B", "where a shape"),
     ("A B", "left over"), ("A+$", "cannot be read")],
)  # fmt: skip
def test_a_composite_expression_that_does_not_read_is_refused_saying_where(
    expression: str, said: str
) -> None:
    with pytest.raises(ValueError, match=said):
        composite.parse(expression)


def obj(tmp_path: Any, text: str) -> str:
    path = tmp_path / "shape.obj"
    path.write_text(text)
    return str(path)


def test_an_obj_file_is_read_for_its_vertices_and_faces_alone(tmp_path: Any) -> None:
    text = ("# a pyramid\nv 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1 2\nvn 0 0 1\nvt 0 0\n"
            "f 1/1/1 2/1/1 3/1/1\nf 1 2 4\nf 1 3 4\nf 2 3 4 1\n")  # fmt: skip
    vertices, faces = wavefront.read_obj(obj(tmp_path, text))
    assert vertices[3] == (0.0, 0.0, 2.0)
    assert faces == [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3, 0)]


@pytest.mark.parametrize(
    ("text", "said"),
    [("v 0 0 0\nf 1 2\n", "unsupported 2 vertices"), ("v 0 0 0\nf -1 -2 -3\n", "relative"),
     ("v 0 0 0\nf 1 1 1\n", "Not enough faces")],
)  # fmt: skip
def test_an_obj_file_root_would_not_import_is_refused_in_root_s_words(
    tmp_path: Any, text: str, said: str
) -> None:
    with pytest.raises(wavefront.ObjError, match=said):
        wavefront.read_obj(obj(tmp_path, text))


def test_an_obj_file_that_is_not_there_is_refused(tmp_path: Any) -> None:
    with pytest.raises(wavefront.ObjError, match="Unable to open"):
        wavefront.read_obj(tmp_path / "none.obj")


def solids() -> list[Solid]:
    return [Solid(build.box(1, 1, 1), line_color=2, name="box"),
            Solid(build.tube(0, 1, 1, steps=8), line_color=4, transparency=50)]  # fmt: skip


def test_plotly_gets_a_shaded_mesh_per_solid_in_its_colour(tmp_path: Any) -> None:
    figure = backends.to_plotly(solids())
    assert [trace.color for trace in figure.data] == ["#ff0000", "#0000ff"]
    assert figure.data[1].opacity == 0.5
    backends.save(solids(), str(tmp_path / "scene.html"), backend="plotly")
    assert "<html>" in (tmp_path / "scene.html").read_text()


class Plotter:
    """pyvista's ``Plotter`` as far as a geometry uses it: surfaces added, a screenshot taken."""

    def __init__(self, off_screen: bool, window_size: list[int]) -> None:
        self.off_screen, self.window_size = off_screen, window_size
        self.meshes: list[Any] = []
        self.closed = False

    def add_mesh(self, surface: Any, color: Any, opacity: float) -> None:
        self.meshes.append((surface, color, opacity))

    def screenshot(self, path: str) -> None:
        with open(path, "w") as out:
            out.write(f"{len(self.meshes)} surfaces")

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def pyvista(monkeypatch: Any) -> Any:
    """A stand-in for pyvista, which the suite does not install: what it is handed is kept."""
    module = types.ModuleType("pyvista")
    module.Plotter = Plotter  # type: ignore[attr-defined]
    module.PolyData = lambda points, faces: (points, faces)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvista", module)
    return module


def test_pyvista_gets_a_surface_per_solid_off_screen(pyvista: Any, tmp_path: Any) -> None:
    plotter = backends.to_pyvista(solids(), size=(300, 200))
    assert (plotter.off_screen, plotter.window_size) == (True, [300, 200])
    (points, faces), color, opacity = plotter.meshes[0]
    assert (len(points), faces[:5].tolist()) == (8, [4, 0, 1, 2, 3])
    assert (color, opacity) == ((1.0, 0.0, 0.0), 1.0)
    backends.save(solids(), str(tmp_path / "scene.png"))
    assert (tmp_path / "scene.png").read_text() == "2 surfaces"


def test_a_3d_library_that_is_not_installed_is_refused_with_how_to_get_it(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "pyvista", None)
    with pytest.raises(UnsupportedFeatureError, match="pip install pyvista"):
        backends.to_pyvista(solids())


def test_a_3d_library_that_is_neither_is_refused() -> None:
    with pytest.raises(UnsupportedFeatureError, match="plotly or pyvista"):
        backends.save(solids(), "x.png", backend="vtk")
