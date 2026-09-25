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

import html
from collections.abc import Iterable, Sequence
from typing import Any

from . import backends, build
from .backends import astext

__all__ = ["chain_table", "directory_table", "figure", "rntuple_table", "table", "tree_table"]

#: How the tables look: compact, left-aligned, in the notebook's own font.
STYLE = (
    "border-collapse:collapse;font-size:90%;text-align:left",
    "padding:2px 10px;border-bottom:1px solid #ddd;text-align:left",
)


def _preformatted(text: str) -> str:
    return f"<pre>{html.escape(text)}</pre>"


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


def table(caption: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """An HTML table with a caption, every cell escaped."""
    head = "".join(f'<th style="{STYLE[1]}">{html.escape(str(cell))}</th>' for cell in header)
    body = "".join(
        "<tr>" + "".join(f'<td style="{STYLE[1]}">{html.escape(str(cell))}</td>' for cell in row)
        + "</tr>"
        for row in rows
    )  # fmt: skip
    return (
        f'<table style="{STYLE[0]}"><caption style="text-align:left">'
        f"{html.escape(caption)}</caption><tr>{head}</tr>{body}</table>"
    )


def tree_table(tree: Any) -> str:
    """A tree's branches: each name, the type it reads as, and whether it varies in length."""
    rows = [
        (name, branch.typename or f"? ({branch.leaf.classname})",
         "variable" if branch.is_jagged else "", tree.unreadable.get(name, ""))
        for name, branch in tree.branches.items()
    ]  # fmt: skip
    caption = f"TTree {tree.name!r}: {len(tree)} entries, {len(rows)} branches"
    return table(caption, ("branch", "type", "length", "not read because"), rows)


def chain_table(name: str, files: Sequence[str], counts: Sequence[int] | None) -> str:
    """A chain's files, with their entries once something has counted them."""
    rows = [(file, "" if counts is None else counts[at]) for at, file in enumerate(files)]
    return table(f"Chain {name!r} over {len(rows)} files", ("file", "entries"), rows)


def rntuple_table(ntuple: Any) -> str:
    """An RNTuple's fields: each name, its Python type and the C++ it was written from."""
    rows = [
        (name, field.typename, field.cxx_type, ntuple.unreadable.get(name, ""))
        for name, field in ntuple.fields.items()
    ]
    caption = f"RNTuple {ntuple.name!r}: {ntuple.num_entries} entries, {len(rows)} fields"
    return table(caption, ("field", "type", "C++ type", "not read because"), rows)


def directory_table(caption: str, keys: Iterable[Any]) -> str:
    """A directory's keys: every name and cycle, the class it holds and its title."""
    rows = [(key.name, key.cycle, key.classname, key.title) for key in keys]
    return table(caption, ("name", "cycle", "class", "title"), rows)
