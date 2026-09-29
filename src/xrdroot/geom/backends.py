"""A geometry as a 3-D scene you can turn: plotly's figure, or pyvista's off-screen render.

ROOT shows a geometry in its OpenGL viewer, a window of its own. Here the
same solids go to one of the two scientific-Python 3-D libraries instead:
:func:`to_plotly` makes a ``plotly.graph_objects.Figure`` of shaded meshes,
one per volume in its colour and transparency, to show in a notebook or
save as HTML; :func:`to_pyvista` makes a ``pyvista.Plotter`` rendering
off-screen, whose :meth:`screenshot` is a PNG with hidden surfaces hidden.
Neither library is a dependency - each is the ``geom`` extra - and asking
for one that is not installed is refused with the ``pip install`` that
fixes it. A picture of a pad, ``SaveAs("x.png")``, needs neither: it is the
wireframe :mod:`.paint` draws with matplotlib.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from .scene import Solid

__all__ = ["to_plotly", "to_pyvista", "save", "library"]

#: The package each 3-D library is, for the refusal that says how to get it.
PACKAGES = {"plotly": "plotly", "pyvista": "pyvista"}


def library(name: str) -> Any:
    """Import one of the 3-D libraries, or refuse with the command that installs it."""
    try:
        return importlib.import_module("plotly.graph_objects" if name == "plotly" else name)
    except ImportError:
        raise UnsupportedFeatureError(
            f"drawing a geometry with {name} needs {PACKAGES[name]}, which is not installed: "
            f"pip install {PACKAGES[name]} (or pip install xrdroot[geom]) - or save the pad "
            f"as a PNG, which draws its wireframe with matplotlib"
        ) from None


def _rgb(index: int) -> tuple[float, float, float]:
    from ..canvas.colors import Colors

    return Colors().rgb(index)


def _hex(index: int) -> str:
    return "#" + "".join(f"{round(255 * channel):02x}" for channel in _rgb(index))


def to_plotly(solids: Sequence[Solid]) -> Any:
    """A plotly figure of each solid as a shaded mesh, in its volume's line colour."""
    go = library("plotly")
    figure = go.Figure()
    for solid in solids:
        x, y, z = solid.mesh.points.T
        i, j, k = solid.mesh.triangles().T
        figure.add_trace(go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, name=solid.name,
                                   color=_hex(solid.line_color),
                                   opacity=1.0 - solid.transparency / 100.0))  # fmt: skip
    figure.update_layout(scene={"aspectmode": "data"}, showlegend=False)
    return figure


def to_pyvista(solids: Sequence[Solid], size: tuple[int, int] = (700, 500)) -> Any:
    """A pyvista plotter, off-screen, with each solid as a surface in its volume's colour."""
    pv = library("pyvista")
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    for solid in solids:
        faces = np.concatenate([[len(face), *face] for face in solid.mesh.faces] or [[]])
        surface = pv.PolyData(solid.mesh.points, faces.astype(np.int64))
        plotter.add_mesh(surface, color=_rgb(solid.line_color),
                         opacity=1.0 - solid.transparency / 100.0)  # fmt: skip
    return plotter


def save(solids: Sequence[Solid], path: str, backend: str = "pyvista") -> None:
    """The solids as a picture: a PNG screenshot through pyvista, or plotly's HTML page."""
    if backend == "plotly":
        to_plotly(solids).write_html(str(path))
        return
    if backend != "pyvista":
        raise UnsupportedFeatureError(
            f"a geometry is drawn in 3-D with plotly or pyvista, and {backend!r} is neither; "
            f"a pad saved as a picture draws its wireframe with matplotlib"
        )
    plotter = to_pyvista(solids)
    plotter.screenshot(str(path))
    plotter.close()
