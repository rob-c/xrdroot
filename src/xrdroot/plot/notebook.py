"""What a Jupyter notebook shows for an object left at the end of a cell.

A histogram, graph, profile, efficiency or function shows as its picture: a
small SVG drawn by matplotlib on a figure pyplot never hears of, or plotly's
HTML when plotly is the backend set. Showing must never be why a cell
fails, so anything that goes wrong on the way falls back to the picture in
characters, and failing that to the object's ``repr``.

A tree, a chain, an RNTuple and a directory show as tables of what they
hold - names, types, entries, classes and cycles - made from what was read
when they were opened, without reading a single basket or page.
"""

from __future__ import annotations

from typing import Any

from ..notebook import (
    STYLE as STYLE,
)
from ..notebook import (
    _preformatted,
)
from ..notebook import (
    chain_table as chain_table,
)
from ..notebook import (
    directory_table as directory_table,
)
from ..notebook import (
    rntuple_table as rntuple_table,
)
from ..notebook import (
    table as table,
)
from ..notebook import (
    tree_table as tree_table,
)
from . import backends, build
from .backends import astext

__all__ = ["chain_table", "directory_table", "figure", "rntuple_table", "table", "tree_table"]


def _text(obj: Any) -> str:
    """The object in characters, or its ``repr`` if not even that can be drawn."""
    try:
        return _preformatted(astext.render(build.picture(obj)))
    except Exception:
        return _preformatted(repr(obj))


def _drawn(obj: Any) -> str:
    picture = build.picture(obj)
    name = backends.get_backend()
    if name == "plotly":
        backends.backend(name)
        from .backends.withplotly import html as plotly_html

        return plotly_html(picture)
    if name == "text":
        return _preformatted(astext.render(picture))
    backends.backend("matplotlib")
    from .backends.withmatplotlib import svg

    return svg(picture)


def figure(obj: Any) -> str:
    """``_repr_html_`` for anything drawn: its picture, never an exception."""
    try:
        return _drawn(obj)
    except Exception:
        return _text(obj)
