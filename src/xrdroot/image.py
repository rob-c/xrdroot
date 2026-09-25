"""A picture kept in a ROOT file: ``TASImage``, which carries a whole PNG.

ROOT's image class writes itself by hand. ``TASImage::Streamer`` writes its
name and title, one byte saying how the picture is kept, and then - for every
picture read in from a file or drawn from a canvas - the PNG it would have
saved, length first. That PNG is the whole of the picture, so this module
keeps its bytes as they are and decodes them into pixels only when asked,
with nothing more than :mod:`zlib` and NumPy.

An image made from numbers is kept as the numbers and a palette instead,
which :mod:`.palette` reads and colours. What this refuses is a ``TASImage``
written by ROOT 4, which kept its zoom and its range but none of its pixels.
"""

from __future__ import annotations

import os
import struct
import zlib
from collections.abc import Callable
from typing import Any

import numpy as np

from .buffer import Buffer
from .errors import FormatError, UnsupportedFeatureError

__all__ = ["IMAGES", "Image", "decode_png", "read_image"]

#: The eight bytes every PNG starts with.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: How many samples a pixel of each PNG colour type has: grey, RGB, a
#: palette index, grey with alpha, and RGBA.
CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


class Image:
    """A ``TASImage``: a picture, its size, and its pixels.

        >>> image = xrdroot.open_root("gallery.root")["hsimple.png."]  # doctest: +SKIP
        >>> image.width, image.height, image.array.shape                 # doctest: +SKIP
        (696, 472, (472, 696, 4))
        >>> image.save("hsimple.png")                                    # doctest: +SKIP

    :attr:`png` is the file's own bytes, untouched; :attr:`array` is them
    decoded, top row first, as red, green, blue and alpha.

    An image made from data rather than read from a picture keeps that data:
    :attr:`values` is the grid, top row first, and :attr:`palette` the stops
    and colours that give :attr:`array` its pixels, which are ROOT's own; its
    :attr:`png` is those pixels encoded, since the file held none.
    """

    __slots__ = (
        "classname",
        "name",
        "title",
        "width",
        "height",
        "values",
        "palette",
        "_png",
        "_array",
    )

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        self.classname = classname
        self.name: str = members["fName"]
        self.title: str = members["fTitle"]
        self._png: bytes | None = members.get("png")
        #: The numbers an image made from data was made from, or ``None``.
        self.values: np.ndarray[Any, Any] | None = members.get("values")
        #: How those numbers are coloured: ``fPoints`` and a 16-bit level per channel.
        self.palette: dict[str, Any] | None = members.get("palette")
        self._array: np.ndarray[Any, Any] | None = None
        if self.values is None:
            self.width, self.height = png_size(self.png)
        else:
            self.height, self.width = self.values.shape

    def __repr__(self) -> str:
        return f"<Image {self.name!r} of {self.width}x{self.height} pixels>"

    @property
    def png(self) -> bytes:
        """The PNG the image was written as, byte for byte, or its pixels encoded as one."""
        if self._png is None:
            from .palette import encode_png

            self._png = encode_png(self.array)
        return self._png

    @property
    def array(self) -> np.ndarray[Any, Any]:
        """The pixels, ``height`` by ``width`` by RGBA, as ``uint8``: decoded once."""
        if self._array is None:
            self._array = self._pixels()
        return self._array

    def _pixels(self) -> np.ndarray[Any, Any]:
        if self.values is None:
            return decode_png(self.png)
        from .palette import colorize

        return colorize(self.values, self.palette or {})

    def _repr_png_(self) -> bytes:
        """What a notebook shows: the picture itself."""
        return self.png

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the picture to ``path``, in the format its extension names.

        A ``.png`` is the stored bytes written out as they are; anything else
        is the decoded pixels handed to matplotlib, which is what knows the
        other formats.
        """
        if os.fspath(path).lower().endswith(".png"):
            with open(path, "wb") as out:
                out.write(self.png)
            return
        try:
            from matplotlib import image as mpimage
        except ImportError as why:
            raise UnsupportedFeatureError(
                f"{self.name!r} is kept as a PNG, and saving it as anything else needs "
                f"matplotlib, which is not installed; save it as a .png, or install "
                f"xrdroot[plot]"
            ) from why
        mpimage.imsave(path, self.array)


def png_size(png: bytes) -> tuple[int, int]:
    """The width and height a PNG's header gives, without decoding it."""
    if not png.startswith(PNG_SIGNATURE) or png[12:16] != b"IHDR":
        raise FormatError("a TASImage says it holds a PNG, and its bytes are not one")
    width, height = struct.unpack_from(">II", png, 16)
    return int(width), int(height)


def _chunks(png: bytes) -> dict[bytes, bytes]:
    """Every chunk of a PNG by type, the image data joined into one."""
    found: dict[bytes, bytes] = {}
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(png):
        size, kind = struct.unpack_from(">I4s", png, pos)
        body = png[pos + 8 : pos + 8 + size]
        found[kind] = found.get(kind, b"") + body
        pos += 12 + size
    return found


def _header(chunks: dict[bytes, bytes]) -> tuple[int, int, int, int]:
    """Width, height, colour type and samples per pixel, of what this decodes."""
    width, height, depth, colour, _method, _filter, interlace = struct.unpack(
        ">IIBBBBB", chunks[b"IHDR"]
    )
    if depth != 8 or colour not in CHANNELS or interlace:
        raise UnsupportedFeatureError(
            f"this PNG is {depth}-bit, colour type {colour}{', interlaced' if interlace else ''}, "
            f"and this reader decodes only the 8-bit, non-interlaced PNGs ROOT writes; its "
            f"bytes are in .png, for a library that decodes the rest"
        )
    return width, height, colour, CHANNELS[colour]


def _paeth(line: bytearray, above: bytes, step: int) -> None:
    """Undo the Paeth filter in place: each byte by whichever neighbour is nearest."""
    for i in range(len(line)):
        a = line[i - step] if i >= step else 0
        b = above[i]
        c = above[i - step] if i >= step else 0
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        near = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        line[i] = (line[i] + near) & 0xFF


def _average(line: bytearray, above: bytes, step: int) -> None:
    """Undo the average filter in place: the mean of the left and upper bytes."""
    for i in range(len(line)):
        left = line[i - step] if i >= step else 0
        line[i] = (line[i] + ((left + above[i]) >> 1)) & 0xFF


def _sub(line: bytearray, above: bytes, step: int) -> None:
    """Undo the sub filter in place: a running sum along the row, sample by sample."""
    del above
    pixels = np.frombuffer(bytes(line), np.uint8).reshape(-1, step)
    line[:] = np.cumsum(pixels, axis=0, dtype=np.uint8).tobytes()


def _up(line: bytearray, above: bytes, step: int) -> None:
    """Undo the up filter in place: each byte plus the one above it."""
    del step
    line[:] = (np.frombuffer(bytes(line), np.uint8) + np.frombuffer(above, np.uint8)).tobytes()


#: The four PNG row filters by number; filter 0 is the row as it is.
FILTERS: dict[int, Callable[[bytearray, bytes, int], None]] = {
    1: _sub,
    2: _up,
    3: _average,
    4: _paeth,
}


def _unfiltered(raw: bytes, height: int, stride: int, step: int) -> bytes:
    """Every row of a PNG with the filter it was written under taken off."""
    rows = []
    above = bytes(stride)
    for row in range(height):
        start = row * (stride + 1)
        line = bytearray(raw[start + 1 : start + 1 + stride])
        undo = FILTERS.get(raw[start])
        if undo is None and raw[start]:
            raise FormatError(f"row {row} of a PNG names filter {raw[start]}, which is not one")
        if undo is not None:
            undo(line, above, step)
        above = bytes(line)
        rows.append(above)
    return b"".join(rows)


def _rgba(pixels: np.ndarray[Any, Any], colour: int, chunks: dict[bytes, bytes]) -> Any:
    """Samples of any colour type as RGBA: grey spread, a palette looked up."""
    if colour == 3:
        palette = np.frombuffer(chunks[b"PLTE"], np.uint8).reshape(-1, 3)
        alpha = np.full(len(palette), 255, np.uint8)
        given = np.frombuffer(chunks.get(b"tRNS", b""), np.uint8)[: len(palette)]
        alpha[: len(given)] = given
        table = np.column_stack([palette, alpha])
        return table[pixels[..., 0]]
    opaque = np.full((*pixels.shape[:2], 1), 255, np.uint8)
    if colour == 0:
        return np.concatenate([pixels.repeat(3, axis=2), opaque], axis=2)
    if colour == 4:
        return np.concatenate([pixels[..., :1].repeat(3, axis=2), pixels[..., 1:]], axis=2)
    if colour == 2:
        return np.concatenate([pixels, opaque], axis=2)
    return pixels


def decode_png(png: bytes) -> np.ndarray[Any, Any]:
    """A PNG's pixels as a ``height`` by ``width`` by 4 array of ``uint8``."""
    png_size(png)
    chunks = _chunks(png)
    width, height, colour, step = _header(chunks)
    stride = width * step
    try:
        raw = zlib.decompress(chunks.get(b"IDAT", b""))
    except zlib.error as why:
        raise FormatError(f"the pixels of a PNG would not inflate: {why}") from None
    if len(raw) != height * (stride + 1):
        raise FormatError(
            f"a {width}x{height} PNG inflated to {len(raw)} bytes, "
            f"where {height * (stride + 1)} were promised"
        )
    pixels = np.frombuffer(_unfiltered(raw, height, stride, step), np.uint8)
    rgba = _rgba(pixels.reshape(height, width, step), colour, chunks)
    return np.ascontiguousarray(rgba, dtype=np.uint8)


def read_image(_described: Any) -> Callable[[Buffer], dict[str, Any]]:
    """How a ``TASImage`` reads: ``TASImage::Streamer``, as ROOT 5 on writes it."""

    def read(buf: Buffer) -> dict[str, Any]:
        version, end = buf.header()
        name, title = buf.named()
        kept = buf.u8()
        if not version or kept > 1:
            raise UnsupportedFeatureError(
                f"{name!r} is a TASImage of version {version} whose bytes do not go on to "
                f"a picture, which is how ROOT 4 wrote one: its zoom and its range, and "
                f"none of its pixels"
            )
        row: dict[str, Any] = {"fName": name, "fTitle": title}
        if kept:
            row["png"] = buf.take(buf.i32())
        else:
            from .palette import read_vector_image

            row.update(read_vector_image(buf))
        buf.resume(end)
        return row

    return read


#: The classes this module reads, against how each is read.
IMAGES: dict[str, Callable[[Any], Callable[[Buffer], Any]]] = {"TASImage": read_image}
