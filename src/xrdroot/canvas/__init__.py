"""ROOT's canvases, read from a file and drawn the way ROOT drew them.

    >>> c = f["c1"]                                     # doctest: +SKIP
    >>> [(type(obj).__name__, option) for obj, option in c.primitives]
    [('Histogram', 'hist'), ('Primitive', ''), ('Primitive', '')]
    >>> c.save("c1.png")

A saved ``TCanvas`` is a tree of pads, each a rectangle of the canvas with
margins round a frame and a list of what it drew, every entry with the
option it was drawn with. :class:`Canvas` reads all of it - the pads in
:attr:`~Canvas.pads`, what each drew in :attr:`~Pad.primitives` - and
:meth:`Canvas.plot` draws it with matplotlib, an axes per pad, each thing
in it by its option: histograms and graphs by their own ``plot``, text in
ROOT's ``#`` mathematics, legends, stats boxes and paves where ROOT put
them, in ROOT's colours, line styles, markers and fonts.

Reading the canvas takes nothing but the file; drawing it takes
matplotlib. What it draws and what it leaves out are listed in the
documentation's "Canvases" section, and anything left out of a picture is
named in a warning when it is drawn.
"""

from __future__ import annotations

from typing import Any

from .model import PRIMITIVES, Canvas, Pad, Primitive

__all__ = ["CANVASES", "Canvas", "Pad", "Primitive", "render"]

#: The classes of a canvas that come back as more than a dictionary.
CANVASES: dict[str, type] = {
    "TCanvas": Canvas,
    "TPad": Pad,
    **dict.fromkeys(PRIMITIVES, Primitive),
}


def render(obj: Any, path: Any, **options: Any) -> Any:
    """Save a canvas or a pad as a picture, whatever the suffix of ``path`` says.

    ``obj`` is a :class:`Canvas` or :class:`Pad`, or the members of a
    ``TCanvas`` or ``TPad`` as a dictionary; ``options`` are
    :meth:`matplotlib.figure.Figure.savefig`'s.
    """
    if isinstance(obj, dict):
        obj = Canvas("TCanvas", obj) if "TPad" in obj else Pad("TPad", obj)
    if not isinstance(obj, Pad):
        raise TypeError(
            f"only a canvas or a pad can be rendered as a canvas, and this is a "
            f"{type(obj).__name__}; draw it with its own .plot() instead"
        )
    return obj.save(path, **options)
