"""A ``TASImage`` kept as numbers and a palette, rather than as a PNG.

An image made from data - ``TImage::SetImage`` handed a grid of doubles, as
the ``galaxy_image`` tutorial's picture of NGC 4254 was - is written as that
grid and the palette that colours it: ``TAttImage`` by hand, the palette's
stops and the 16-bit red, green, blue and alpha at each, then the width, the
height and the values, bottom row first. None of it is described by the file.

The colours are worked out the way libAfterImage's
``colorize_asimage_vector`` works them out - the stops stretched over the
values' range, each value placed between two of them and its channels
interpolated at 16 bits and truncated - and then kept at their top eight
bits, which is what ROOT's own ARGB array of the same image holds, pixel for
pixel.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np

from .buffer import Buffer

__all__ = ["colorize", "encode_png", "read_vector_image"]

#: The palette's channels in the order ``TImagePalette`` writes them.
CHANNEL_ORDER = ("fColorRed", "fColorGreen", "fColorBlue", "fColorAlpha")


def read_palette(buf: Buffer) -> dict[str, Any]:
    """A ``TImagePalette``: how many stops, where each is, and its four channels."""
    _version, end = buf.header()
    buf.tobject()
    count = buf.u32()
    palette: dict[str, Any] = {"fPoints": np.frombuffer(buf.take(8 * count), ">f8").astype(float)}
    for channel in CHANNEL_ORDER:
        palette[channel] = np.frombuffer(buf.take(2 * count), ">u2").astype(np.int64)
    buf.resume(end)
    return palette


def read_vector_image(buf: Buffer) -> dict[str, Any]:
    """The rest of a ``TASImage`` kept as values: its ``TAttImage``, then the grid.

    The grid comes back top row first, the way :attr:`Image.array` is, so
    that a value and the pixel it coloured share an index.
    """
    _version, end = buf.header()
    attributes = {"fImageQuality": buf.i32(), "fImageCompression": buf.u32()}
    attributes["fConstRatio"] = buf.bool()
    palette = read_palette(buf)
    buf.resume(end)
    width, height = buf.i32(), buf.i32()
    values = np.frombuffer(buf.take(8 * width * height), ">f8").astype(float)
    return {
        **attributes,
        "palette": palette,
        "values": values.reshape(height, width)[::-1].copy(),
    }


def _stop_from_above(stops: list[float], value: float, at: int) -> int:
    while at > 0:
        at -= 1
        if stops[at] < value:
            return at
    return 0


def _stop_from_below(stops: list[float], value: float, at: int) -> int:
    last = len(stops) - 1
    while stops[at + 1] < value:
        at += 1
        if at >= last:
            return last - 1
    return at


def stops_below(values: np.ndarray[Any, Any], stops: list[float]) -> np.ndarray[Any, Any]:
    """Which stop each value is coloured from, found the way libAfterImage finds it.

    It walks from the stop the last value used, down while that stop is
    above the value and up while the next is below it, so a value exactly
    on a stop is coloured from one side or the other by where the walk came
    from; the bottom row goes first, as it was written, and a value equal to
    the one before it in its row takes that one's stop without a walk.
    """
    at = min(len(stops) // 2, len(stops) - 2)
    found = np.empty(values.shape, np.int64)
    for y in range(values.shape[0] - 1, -1, -1):
        row = values[y].tolist()
        previous = None
        for x, value in enumerate(row):
            if value != previous:
                search = _stop_from_above if stops[at] > value else _stop_from_below
                at = search(stops, value, at)
                previous = value
            found[y, x] = at
    return found


def colorize(values: np.ndarray[Any, Any], palette: dict[str, Any]) -> np.ndarray[Any, Any]:
    """The RGBA pixels a palette gives a grid of values, as libAfterImage gives them."""
    low, high = float(values.min()), float(values.max())
    stops = low + (high - low) * palette["fPoints"]
    below = stops_below(values, stops.tolist())
    step = stops[1:] - stops[:-1]
    flat = step == 0
    past = values - stops[below]
    channels = []
    for name in CHANNEL_ORDER:
        levels = palette[name]
        slope = np.where(flat, 1.0, (levels[1:] - levels[:-1]) / np.where(flat, 1.0, step))
        channels.append((past * slope[below]).astype(np.int64) + levels[below])
    rgba = np.clip(np.stack(channels, axis=-1), 0, 0xFFFF) >> 8
    return rgba.astype(np.uint8)


def _chunk(kind: bytes, body: bytes) -> bytes:
    crc = zlib.crc32(kind + body)
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)


def encode_png(rgba: np.ndarray[Any, Any]) -> bytes:
    """An RGBA ``uint8`` array as the plainest PNG there is: unfiltered rows, deflated."""
    height, width = rgba.shape[:2]
    rows = np.concatenate([np.zeros((height, 1), np.uint8), rgba.reshape(height, -1)], axis=1)
    head = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", head)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes()))
        + _chunk(b"IEND", b"")
    )
