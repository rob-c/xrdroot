"""FITS files written byte by byte for the tests: cards, blocks, images and tables."""

from __future__ import annotations

from typing import Any

import numpy as np

#: The size of a FITS block.
BLOCK = 2880


def card(key: str, value: Any = None, comment: str | None = None) -> str:
    """One 80-character card: a keyword with a value (and a comment), or a comment card."""
    if value is None:
        return f"{key:<8}{comment or ''}"[:80].ljust(80)
    text = value if isinstance(value, str) else ("T" if value is True else "F" if value is False
                                                 else repr(value))  # fmt: skip
    body = f"{key:<8}= {text:>20}" + (f" / {comment}" if comment is not None else "")
    return body[:80].ljust(80)


def padded(data: bytes, fill: bytes = b"\0") -> bytes:
    return data + fill * (-len(data) % BLOCK)


def unit(cards: list[str], data: bytes = b"") -> bytes:
    """A header of these cards and ``END``, then the data, both padded to whole blocks."""
    header = "".join(cards) + "END".ljust(80)
    return padded(header.encode("latin-1"), b" ") + padded(data)


def primary(**extra: Any) -> list[str]:
    """The cards of an empty primary unit."""
    return [card("SIMPLE", True, "file conforms"), card("BITPIX", 8), card("NAXIS", 0),
            *[card(k, v) for k, v in extra.items()]]  # fmt: skip


def image_unit(pixels: np.ndarray[Any, Any], bitpix: int, first: bool = True,
               extra: list[str] | None = None) -> bytes:  # fmt: skip
    """An image unit of these pixels (``NAXIS1`` the last axis), primary or an extension."""
    kinds = {8: ">u1", 16: ">i2", 32: ">i4", 64: ">i8", -32: ">f4", -64: ">f8"}
    head = [card("SIMPLE", True)] if first else [card("XTENSION", "'IMAGE   '")]
    head += [card("BITPIX", bitpix), card("NAXIS", pixels.ndim)]
    head += [card(f"NAXIS{k + 1}", n) for k, n in enumerate(pixels.shape[::-1])]
    if not first:
        head += [card("PCOUNT", 0), card("GCOUNT", 1)]
    return unit(head + (extra or []), pixels.astype(kinds[bitpix]).tobytes())


def binary_table(columns: list[tuple[str, str, bytes]], rows: int, heap: bytes = b"",
                 extra: list[str] | None = None) -> bytes:  # fmt: skip
    """A binary table: each column a name, a ``TFORM`` and its bytes in every row, row-major."""
    width = sum(len(data) // rows for _, _, data in columns) if rows else 0
    table = b"".join(
        b"".join(data[r * (len(data) // rows):(r + 1) * (len(data) // rows)]
                 for _, _, data in columns) for r in range(rows))  # fmt: skip
    head = [card("XTENSION", "'BINTABLE'"), card("BITPIX", 8), card("NAXIS", 2),
            card("NAXIS1", width), card("NAXIS2", rows), card("PCOUNT", len(heap)),
            card("GCOUNT", 1), card("TFIELDS", len(columns))]  # fmt: skip
    for n, (name, form, _) in enumerate(columns, 1):
        head += [card(f"TTYPE{n}", f"'{name}'"), card(f"TFORM{n}", f"'{form}'")]
    return unit(head + (extra or []), table + heap)


def ascii_table(columns: list[tuple[str, str, int]], rows: list[str], name: str = "") -> bytes:
    """An ASCII table: each column a name, a ``TFORM`` and where it starts; rows as text."""
    head = [card("XTENSION", "'TABLE   '"), card("BITPIX", 8), card("NAXIS", 2),
            card("NAXIS1", len(rows[0])), card("NAXIS2", len(rows)), card("PCOUNT", 0),
            card("GCOUNT", 1), card("TFIELDS", len(columns))]  # fmt: skip
    for n, (column, form, start) in enumerate(columns, 1):
        head += [card(f"TTYPE{n}", f"'{column}'"), card(f"TFORM{n}", f"'{form}'"),
                 card(f"TBCOL{n}", start)]  # fmt: skip
    if name:
        head.append(card("EXTNAME", f"'{name}'"))
    return unit(head, "".join(rows).encode("ascii"))
