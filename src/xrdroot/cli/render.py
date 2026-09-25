"""Drawing an object into a picture file, for ``print`` and ``draw``.

Whatever draws itself is asked to: a canvas-like object that saves itself
(``save_as``, as ROOT's ``TCanvas::SaveAs``) does, and anything with a
``.plot()`` - a histogram, a graph, a profile, a stack - draws onto fresh
matplotlib axes whose figure is saved in the format the file name's suffix
says. A ``TCanvas`` read as its bare members is handed to
``xrdroot.canvas.render(obj, path, **options)`` when that module is there,
and refused by name when it is not.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

from ..errors import UnsupportedFeatureError

__all__ = ["render", "options", "FORMATS"]

#: The picture formats a file name may end in.
FORMATS = (".png", ".pdf", ".svg", ".eps", ".ps", ".jpg", ".jpeg")
#: The classes whose pictures are a canvas's to draw, not an object's.
CANVASES = ("TCanvas", "TPad")


def options(given: list[str] | None) -> dict[str, Any]:
    """``["color=red", "linewidth=2"]`` as keyword arguments, each value a literal if it is one."""
    found: dict[str, Any] = {}
    for item in given or []:
        name, equals, value = item.partition("=")
        if not equals or not name:
            raise ValueError(f"an option is name=value, and {item!r} is not")
        try:
            found[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            found[name] = value
    return found


def _checked(path: str) -> Path:
    where = Path(path)
    if where.suffix.lower() not in FORMATS:
        raise ValueError(
            f"{path} does not end in a picture format this can write; "
            f"end it in one of {', '.join(FORMATS)}"
        )
    return where


def _canvas(value: Any, path: Path, style: dict[str, Any]) -> None:
    """A ``TCanvas`` read as members, drawn by ``xrdroot.canvas`` if it is installed."""
    try:
        module = importlib.import_module("xrdroot.canvas")
    except ImportError:
        module = None
    draw = getattr(module, "render", None)
    if draw is None:
        raise UnsupportedFeatureError(
            "this is a TCanvas, and drawing a canvas needs xrdroot.canvas, which this "
            "installation does not have; print the histograms and graphs in it one by one"
        )
    draw(value, str(path), **style)


def _matplotlib() -> None:
    """Draw without a window: a command-line tool writes files, and may have no display."""
    try:
        import matplotlib
    except ImportError:
        return  # .plot() itself refuses, naming both ways out
    matplotlib.use("Agg")


def render(value: Any, path: str, style: dict[str, Any] | None = None, classname: str = "") -> Path:
    """Draw ``value`` into the picture file ``path``, and say where it went."""
    where = _checked(path)
    style = style or {}
    save_as = getattr(value, "save_as", None)
    if callable(save_as):
        save_as(str(where), **style)
        return where
    if classname in CANVASES and not hasattr(value, "plot"):
        _canvas(value, where, style)
        return where
    if not callable(getattr(value, "plot", None)):
        raise UnsupportedFeatureError(
            f"a {classname or type(value).__name__} has no picture to draw; "
            f"histograms, graphs, profiles, stacks and canvases do"
        )
    _matplotlib()
    figure = value.plot(**style).figure
    figure.savefig(str(where))
    from matplotlib import pyplot

    pyplot.close(figure)
    return where
