"""``SaveAs`` and ``Print``: a pad as a picture, by the suffix of its name.

``.png``, ``.jpg``, ``.gif``, ``.svg``, ``.pdf``, ``.eps`` and ``.ps`` are
written by matplotlib at the canvas's size in pixels. A PDF can be a book
of pages, as ROOT makes one: ``Print("book.pdf[")`` opens it and prints
nothing, ``Print("book.pdf")`` adds this pad as a page while it is open,
``"book.pdf("`` opens it with this pad as its first page, ``"book.pdf)"``
adds the last page and closes it, and ``"book.pdf]"`` closes it. Each
picture is announced on the standard error, as ROOT announces it.

The pictures come out the same whichever matplotlib draws them: a GIF is
drawn as a PNG and handed to Pillow, which matplotlib requires, because
matplotlib before 3.11 writes no GIFs; and a book is begun only when its
first page comes, so one closed with none leaves no file, as matplotlib
3.10 and later leave it, rather than the empty one earlier releases kept
and warned about.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

__all__ = ["BOOKS", "FORMATS", "save"]

#: The pictures written, by suffix, and the format matplotlib writes each in.
FORMATS = {
    "png": "png", "jpg": "jpg", "jpeg": "jpg", "gif": "gif", "svg": "svg",
    "pdf": "pdf", "eps": "eps", "ps": "ps", "tiff": "tiff", "tif": "tiff",
}  # fmt: skip
#: The PDF books open, by file name: ``None`` until a first page is added to one.
BOOKS: dict[str, Any] = {}


def _figure(pad: Any) -> Any:
    from .snapshot import model

    return model(pad).plot()


def _split(filename: str, option: str) -> tuple[str, str, str]:
    """The file's name, the format it is written in, and the bracket after its suffix."""
    bracket = filename[-1:] if filename[-1:] in ("[", "]", "(", ")") else ""
    name = filename[:-1] if bracket else filename
    suffix = Path(name).suffix.lstrip(".").lower()
    wanted = (option or suffix).lower()
    if wanted not in FORMATS:
        raise ValueError(
            f"a pad is saved as {', '.join(sorted(FORMATS))}; {filename!r} asks for "
            f"{wanted or 'no format'!r}, which this does not write"
        )
    return name, FORMATS[wanted], bracket


def _announce(what: str) -> None:
    print(f"Info in <TCanvas::Print>: {what}", file=sys.stderr)


def save(pad: Any, filename: str, option: str = "") -> None:
    """Write ``pad`` to ``filename``, or add it to the book of that name, as ROOT does."""
    from matplotlib import rc_context

    from ...canvas.styles import DRAWING

    name, kind, bracket = _split(str(filename), str(option))
    with rc_context(DRAWING):  # text is drawn when saved, and drawn as ROOT's only so
        if kind == "pdf" and (bracket or name in BOOKS):
            _book(pad, name, bracket)
            return
        figure = _figure(pad)
        if kind == "gif":
            _gif(figure, name)
        else:
            figure.savefig(name, format=kind, dpi=figure.dpi)
    _announce(f"{kind} file {name} has been created")


def _gif(figure: Any, name: str) -> None:
    """``figure`` as a GIF: drawn as a PNG, and written by Pillow, as matplotlib 3.11 writes one."""
    from PIL import Image

    drawn = io.BytesIO()
    figure.savefig(drawn, format="png", dpi=figure.dpi)
    drawn.seek(0)
    with Image.open(drawn) as image:
        image.save(name, format="GIF")


def _book(pad: Any, name: str, bracket: str) -> None:
    """A page added to a book, or the book opened or closed, as the bracket says."""
    from matplotlib.backends.backend_pdf import PdfPages

    if bracket != "]" and name not in BOOKS:
        BOOKS[name] = None
    if bracket not in ("[", "]"):
        if BOOKS[name] is None:
            BOOKS[name] = PdfPages(name)
        BOOKS[name].savefig(_figure(pad))
        _announce(f"Current canvas added to pdf file {name}")
    if bracket in ("]", ")") and name in BOOKS:
        _close(BOOKS.pop(name))
        _announce(f"pdf file {name} has been closed")


def _close(book: Any) -> None:
    """Finish a book, if a page was ever added to it; one without is no file at all."""
    if book is not None:
        book.close()


def close_books() -> None:
    """Close every book still open, as ROOT does when it exits."""
    for name in list(BOOKS):
        _close(BOOKS.pop(name))
