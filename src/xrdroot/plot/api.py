"""``plot()``: any object this library reads, drawn by the backend asked for.

    >>> ax = f["h_pt"].plot()                                  # doctest: +SKIP
    >>> f["h_eta"].plot(option="E1 SAME", color="kRed+1")      # doctest: +SKIP
    >>> fig = f["h2"].plot(backend="plotly", option="LEGO2Z")  # doctest: +SKIP

One call for everything: the object is made a picture (:mod:`~.build`), and
the backend draws the picture and hands back its own object - matplotlib
axes, a plotly figure, a bokeh figure, a string - to keep styling.
"""

from __future__ import annotations

from typing import Any

from . import backends, build, styles
from .model import Picture

__all__ = ["draw_picture", "plot"]


def draw_picture(
    picture: Picture, ax: Any = None, backend: str | None = None, style: str | None = None
) -> Any:
    """A :class:`~.model.Picture` drawn by a backend, onto ``ax`` or something new."""
    name, module = backends.backend(backend)
    with styles.applied(style, name):
        drawn = module.render(picture, ax, backends.current(name))
    return backends.remember(name, styles.dressed(drawn, style, name))


def plot(
    obj: Any, ax: Any = None, backend: str | None = None, option: str = "", **style: Any
) -> Any:
    """Draw ``obj`` with ROOT's ``option`` and hand back what it was drawn on.

    ``obj`` is a histogram, profile, efficiency, graph, multigraph, stack
    or function. ``option`` is ROOT's - ``"E1"``, ``"HIST"``, ``"COLZ"``,
    ``"AP"``, ``"SAME"`` - and an option this does not draw is refused by
    name. ``ax`` is what to draw on: matplotlib axes, a plotly figure, a
    bokeh figure, or the text drawn before.

    The keywords restyle it: ``color``, ``linewidth``, ``linestyle``,
    ``fill``, ``alpha``, ``hatch``, ``marker``, ``markersize``,
    ``markercolor`` and ``label`` its marks - where a colour may be ROOT's,
    ``2`` or ``"kRed+1"`` - and ``title``, ``xlabel``, ``ylabel``,
    ``zlabel``, ``logx``, ``logy``, ``logz``, ``xlim``, ``ylim``,
    ``legend`` and ``grid`` its frame. ``palette`` shades a grid
    (``"bird"``, ROOT's default, or ``"viridis"``, or a colour map the
    backend knows), ``levels`` is how many contours, ``labels`` names the
    things in a stack or multigraph, and ``style`` is ``"ROOT"`` or one of
    mplhep's. Anything else is the backend's own, handed to its call for
    the first layer as it is.
    """
    hep = style.pop("style", None)
    picture = build.picture(obj, option, style)
    return draw_picture(picture, ax, backend, hep)
