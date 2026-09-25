"""What a canvas is made of: pads, and the things drawn in them.

A ``TCanvas`` is a ``TPad`` with a window round it, and a pad is a
rectangle - placed as a fraction of the pad it is in, with margins round
the frame its axes are drawn in - and a list of what to draw in it, each
entry with the option it was drawn with: ``"hist same"``, ``"ap"``,
``"colz"``. Pads nest, as entries of that list.

The objects a pad draws are the classes the rest of the library reads - a
histogram is a :class:`~xrdroot.Histogram` here as anywhere - or one of the
drawing classes, the text, lines, boxes and legends, which come back as a
:class:`Primitive`: the class, and its members looked up by name however
deep in its bases ROOT keeps them.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .colors import Colors

__all__ = ["PRIMITIVES", "Canvas", "Pad", "Primitive", "lookup"]

#: The drawing classes a pad holds that come back as a :class:`Primitive`.
PRIMITIVES = (
    "TFrame",
    "TPave",
    "TPaveText",
    "TPavesText",
    "TPaveLabel",
    "TPaveStats",
    "TPaletteAxis",
    "TLegend",
    "TLegendEntry",
    "TText",
    "TLatex",
    "TLine",
    "TArrow",
    "TBox",
    "TWbox",
    "TEllipse",
    "TMarker",
    "TGaxis",
    "TColor",
)
#: A bit of ``TObject::fBits`` ROOT's own classes lend to more than one
#: meaning; for text, lines and markers, it says their place is in NDC.
NDC_BIT = 1 << 14


def _flatten(members: dict[str, Any]) -> dict[str, Any]:
    """Every member by name, the nearest base first, as a lookup table.

    The members of a class sit beside the bases it derives from, each of
    those a dictionary of its own; walking them breadth first finds the
    member a class itself declares before one a distant base does.
    """
    found: dict[str, Any] = {}
    layer = [members]
    while layer:
        deeper = []
        for row in layer:
            for name, value in row.items():
                found.setdefault(name, value)
                if isinstance(value, dict) and name[:1] == "T":
                    deeper.append(value)
        layer = deeper
    return found


def lookup(obj: Any, name: str, default: Any = None) -> Any:
    """Member ``name`` of anything this library read, wherever its bases keep it."""
    table = getattr(obj, "_table", None)
    if table is None:
        members = getattr(obj, "members", obj)
        table = _flatten(members) if isinstance(members, dict) else {}
    return table.get(name, default)


class Primitive:
    """One of ROOT's drawing classes: text, a line, a box, a legend, a pave.

    >>> text["fTitle"], text.get("fTextSize")       # doctest: +SKIP
    ('#sqrt{s} = 13 TeV', 0.04)
    """

    __slots__ = ("classname", "members", "_table")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        #: The class the file says this is, such as ``TLatex``.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        self._table = _flatten(members)

    def get(self, name: str, default: Any = None) -> Any:
        """Member ``name``, from the class or any of its bases, or ``default``."""
        return self._table.get(name, default)

    def __getitem__(self, name: str) -> Any:
        return self._table[name]

    @property
    def name(self) -> str:
        return str(self.get("fName", ""))

    @property
    def ndc(self) -> bool:
        """Whether its place is in the pad's fractions rather than its axes' units."""
        return bool(int(self.get("fBits", 0)) & NDC_BIT)

    def __repr__(self) -> str:
        return f"<{self.classname} {self.name!r}>" if self.name else f"<{self.classname}>"


def _is_colors(entry: Any) -> bool:
    """Whether a list in a pad is one of the colour tables ROOT saves there."""
    return (
        isinstance(entry, list)
        and bool(entry)
        and all(getattr(one, "classname", "") == "TColor" for one in entry)
    )


class Pad:
    """A ``TPad``: a rectangle of a canvas, and what is drawn in it.

    Its place is a fraction of the pad it is in, and its range is in the
    units of its axes - in powers of ten on an axis drawn logarithmically,
    the way ROOT keeps it.
    """

    __slots__ = ("classname", "members", "_table", "primitives", "colors")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        #: The class the file says this is.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        self._table = _flatten(members)
        held = members.get("fPrimitives") or []
        options = getattr(held, "options", [""] * len(held))
        #: The colour tables the pad was saved with - ROOT's, then its palette.
        self.colors: list[list[Any]] = [entry for entry in held if _is_colors(entry)]
        #: What the pad draws, in order, each with the option it was drawn with.
        self.primitives: list[tuple[Any, str]] = [
            (entry, str(option)) for entry, option in zip(held, options) if not _is_colors(entry)
        ]

    def get(self, name: str, default: Any = None) -> Any:
        """Member ``name``, from the pad or any of its bases, or ``default``."""
        return self._table.get(name, default)

    @property
    def name(self) -> str:
        return str(self.get("fName", ""))

    @property
    def title(self) -> str:
        return str(self.get("fTitle", ""))

    @property
    def pads(self) -> list[Pad]:
        """The pads drawn in this one, in the order they were drawn."""
        return [entry for entry, _option in self.primitives if isinstance(entry, Pad)]

    def walk(self) -> Iterator[Pad]:
        """This pad, then every pad inside it, depth first."""
        yield self
        for pad in self.pads:
            yield from pad.walk()

    @property
    def place(self) -> tuple[float, float, float, float]:
        """Where it is in the pad it is in, as fractions: left, bottom, width, height."""
        return (
            float(self.get("fXlowNDC", 0.0)),
            float(self.get("fYlowNDC", 0.0)),
            float(self.get("fWNDC", 1.0)),
            float(self.get("fHNDC", 1.0)),
        )

    @property
    def margins(self) -> tuple[float, float, float, float]:
        """The margins round its frame, as fractions of it: left, right, bottom, top."""
        return (
            float(self.get("fLeftMargin", 0.1)),
            float(self.get("fRightMargin", 0.1)),
            float(self.get("fBottomMargin", 0.1)),
            float(self.get("fTopMargin", 0.1)),
        )

    @property
    def range(self) -> tuple[float, float, float, float]:
        """Its whole extent in the units of its axes: ``x1, y1, x2, y2``."""
        return tuple(float(self.get(name, 0.0)) for name in ("fX1", "fY1", "fX2", "fY2"))  # type: ignore[return-value]

    @property
    def frame(self) -> tuple[float, float, float, float]:
        """The extent of its frame, as its axes were last drawn: ``xmin, ymin, xmax, ymax``.

        ROOT keeps a logarithmic axis's ends as powers of ten; these are the
        ends themselves.
        """
        ends = [float(self.get(name, 0.0)) for name in ("fUxmin", "fUymin", "fUxmax", "fUymax")]
        if self.logx:
            ends[0], ends[2] = 10 ** ends[0], 10 ** ends[2]
        if self.logy:
            ends[1], ends[3] = 10 ** ends[1], 10 ** ends[3]
        return ends[0], ends[1], ends[2], ends[3]

    @property
    def logx(self) -> bool:
        return bool(self.get("fLogx", 0))

    @property
    def logy(self) -> bool:
        return bool(self.get("fLogy", 0))

    @property
    def logz(self) -> bool:
        return bool(self.get("fLogz", 0))

    @property
    def grid(self) -> tuple[bool, bool]:
        """Whether a grid is drawn along x, and along y."""
        return bool(self.get("fGridx", False)), bool(self.get("fGridy", False))

    @property
    def ticks(self) -> tuple[int, int]:
        """``fTickx`` and ``fTicky``: ticks on the top and right sides as well."""
        return int(self.get("fTickx", 0)), int(self.get("fTicky", 0))

    @property
    def painted(self) -> bool:
        """Whether the pad was drawn before it was saved, and so knows its frame.

        A pad that was drawn holds the ``TFrame`` its axes were drawn in; one
        saved without being drawn has only the range it was made with.
        """
        return any(getattr(entry, "classname", "") == "TFrame" for entry, _ in self.primitives)

    def plot(self, figure: Any = None) -> Any:
        """Draw onto a matplotlib figure: the pad alone, filling a canvas of ROOT's default size."""
        whole = dict(self.members, fXlowNDC=0.0, fYlowNDC=0.0, fWNDC=1.0, fHNDC=1.0)
        return Canvas("TCanvas", {"TPad": whole}).plot(figure)

    def save(self, path: Any, **options: Any) -> Any:
        """Draw, and save as whatever the suffix says: ``.png``, ``.pdf``, ``.svg``...

        ``options`` are :meth:`matplotlib.figure.Figure.savefig`'s.
        """
        self.plot().savefig(path, **options)
        return path

    def save_as(self, path: Any, **options: Any) -> Any:
        """``TPad::SaveAs``: the same as :meth:`save`."""
        return self.save(path, **options)

    def __repr__(self) -> str:
        return f"<{self.classname} {self.name!r} of {len(self.primitives)} primitives>"


class Canvas(Pad):
    """A ``TCanvas``: the pad a window was, with its size in pixels.

        >>> c = f["c1"]                                 # doctest: +SKIP
        >>> c.width, c.height, [p.name for p in c.pads]
        (700, 500, ['c1_1', 'c1_2'])
        >>> c.save("c1.png")

    :meth:`plot` draws it with matplotlib, one axes per pad, each drawing
    what its pad does with the option it was drawn with.
    """

    __slots__ = ("canvas",)

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        super().__init__("TPad", members["TPad"])
        self.classname = classname
        #: The members the canvas adds to its pad: sizes, window and flags.
        self.canvas = members

    @property
    def width(self) -> int:
        """``fCw``: how wide the canvas was, in pixels."""
        return int(self.canvas.get("fCw", 0)) or 700

    @property
    def height(self) -> int:
        """``fCh``: how tall the canvas was, in pixels."""
        return int(self.canvas.get("fCh", 0)) or 500

    def palette(self) -> Colors:
        """The colours this canvas draws with: ROOT's, and any it saved."""
        colors = Colors()
        saved = [table for pad in self.walk() for table in pad.colors]
        if saved:
            colors.adopt(saved[0])
        if len(saved) > 1:
            colors.palette = [int(color.get("fNumber", 1)) for color in saved[1]]
        return colors

    def plot(self, figure: Any = None) -> Any:
        """Draw onto a matplotlib figure, made the canvas's size unless one is given.

        Each pad is an axes at its place in the canvas, framed by its
        margins; each thing it holds is drawn with the option it was drawn
        with in ROOT. Anything this does not know how to draw is left out
        with a warning naming it.
        """
        from .paint import paint

        return paint(self, figure)

    def __repr__(self) -> str:
        return f"<TCanvas {self.name!r} {self.width}x{self.height} of {len(self.pads)} pads>"
