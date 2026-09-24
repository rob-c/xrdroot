"""Sparse histograms: many dimensions, and only the bins something fell into.

A ``THnSparse`` keeps no array of bins, because in ten dimensions that array
would not fit in any machine. It keeps the bins that were filled instead, in
chunks: each bin's coordinates packed into as few bits as its axes need, and
beside them a flat array of what is in each. This unpacks the coordinates,
lines them up with the contents, and leaves the grid to :meth:`to_dense` -
which is there for a histogram small enough to have one.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .errors import FormatError, UnsupportedFeatureError
from .hist import Axis

__all__ = ["SPARSE", "SparseHistogram"]

#: The array types a sparse histogram keeps its contents in, by the letter
#: ROOT's typedef for each ends with: ``THnSparseD`` is ``THnSparseT<TArrayD>``.
_CONTENTS = {"D": "TArrayD", "F": "TArrayF", "L": "TArrayL", "I": "TArrayI", "S": "TArrayS"}
_CONTENTS["C"] = "TArrayC"

#: The sparse histogram classes, as a key names them and as the typedef does.
SPARSE = tuple(f"THnSparseT<{held}>" for held in _CONTENTS.values()) + tuple(
    f"THnSparse{letter}" for letter in _CONTENTS
)

#: The most cells :meth:`SparseHistogram.to_dense` will lay out, which is
#: eighty megabytes of doubles: past that the grid is the thing the sparse
#: layout exists to avoid.
DENSE_LIMIT = 10_000_000


def _bits(count: int) -> int:
    """How many bits a coordinate of ``count`` values needs, as ROOT counts them."""
    return max(int(count).bit_length(), 1)


def _find(row: dict[str, Any], name: str) -> Any:
    """A member, wherever the class or one of the bases it is built on keeps it."""
    if name in row:
        return row[name]
    for value in row.values():
        if isinstance(value, dict):
            found = _find(value, name)
            if found is not None:
                return found
    return None


def _unpack(packed: np.ndarray[Any, Any], widths: list[int]) -> np.ndarray[Any, Any]:
    """Every bin's coordinates, out of the bytes they were packed into.

    ROOT packs each bin's coordinates little end first, one axis after
    another, each in just enough bits for its bins and the two at the ends,
    and rounds the whole up to a byte. So each axis's coordinate is a run of
    bits starting part way into one byte, and is put back together from the
    bytes that run covers.
    """
    columns = []
    offset = 0
    for width in widths:
        first, last = offset // 8, (offset + width - 1) // 8
        value = np.zeros(len(packed), dtype=np.int64)
        for place in range(first, last + 1):
            value |= packed[:, place].astype(np.int64) << (8 * (place - first))
        columns.append((value >> (offset % 8)) & ((1 << width) - 1))
        offset += width
    return np.stack(columns, axis=1)


class SparseHistogram:
    """A ``THnSparse``: axes, and the bins along them that hold something.

        >>> sparse = f["hn"]                            # doctest: +SKIP
        >>> sparse.coordinates()[:2], sparse.values()[:2]
        (array([[3, 0, 7], [4, 1, 7]]), array([2., 1.]))

    :meth:`coordinates` is one row per filled bin, a column per axis, counted
    from zero the way :class:`~.hist.Histogram` counts - so ``-1`` is the
    underflow and ``len(axis)`` the overflow - and :meth:`values` is what is
    in each of them, in the same order.
    """

    __slots__ = ("classname", "members", "axes", "_coordinates", "_values", "_variances")

    def __init__(self, classname: str, members: dict[str, Any]) -> None:
        axes, chunks = _find(members, "fAxes"), _find(members, "fBinContent")
        if axes is None or chunks is None:
            raise FormatError(f"a {classname} was written without its axes or its bins")
        #: The class the file says this is, such as ``THnSparseT<TArrayD>``.
        self.classname = classname
        #: Every member, as it was written.
        self.members = members
        #: One :class:`~.hist.Axis` per dimension, in the order they were declared.
        self.axes = tuple(Axis(axis) for axis in axes)
        widths = [_bits(len(axis) + 2) for axis in self.axes]
        self._coordinates, self._values, self._variances = _filled(chunks, widths)

    def __repr__(self) -> str:
        return (
            f"<{self.classname} {self.name!r} of {len(self.axes)} dimensions, "
            f"{len(self)} bins filled>"
        )

    def __len__(self) -> int:
        """How many bins hold something, which is all a sparse histogram keeps."""
        return len(self._values)

    @property
    def name(self) -> str:
        return str(_find(self.members, "fName"))

    @property
    def title(self) -> str:
        return str(_find(self.members, "fTitle"))

    @property
    def entries(self) -> float:
        """How many times it was filled."""
        return float(_find(self.members, "fEntries") or 0.0)

    @property
    def shape(self) -> tuple[int, ...]:
        """How many bins along each axis, not counting the two at the ends."""
        return tuple(len(axis) for axis in self.axes)

    def coordinates(self) -> np.ndarray[Any, Any]:
        """The bin each filled bin is, one row of ``len(axes)`` indices per bin."""
        return self._coordinates - 1

    def values(self) -> np.ndarray[Any, Any]:
        """What is in each filled bin, in the order :meth:`coordinates` gives them."""
        return self._values.copy()

    @property
    def weighted(self) -> bool:
        """Was it filled with weights, so that it keeps their squares too?"""
        return self._variances is not None

    def variances(self) -> np.ndarray[Any, Any]:
        """The sum of the squared weights in each filled bin, or its count without them."""
        return (self._variances if self._variances is not None else self._values).copy()

    def to_dense(self, flow: bool = False) -> np.ndarray[Any, Any]:
        """The bins as a grid, one dimension per axis, zero wherever nothing fell.

        Indexed the way :meth:`~.hist.Histogram.values` is; with ``flow`` each
        dimension gains the two bins at its ends. A grid of more than ten
        million cells is refused, because that is the size the sparse layout
        is there to keep out of memory.
        """
        widths = [len(axis) + 2 for axis in self.axes]
        cells = math.prod(widths)
        if cells > DENSE_LIMIT:
            raise UnsupportedFeatureError(
                f"{self.name!r} would be {cells} cells laid out densely, past the "
                f"{DENSE_LIMIT} this will make; read coordinates() and values() instead"
            )
        grid = np.zeros(widths, dtype=np.float64)
        np.add.at(grid, tuple(self._coordinates.T), self._values)
        if flow:
            return grid
        return grid[(slice(1, -1),) * len(widths)].copy()


def _filled(
    chunks: list[dict[str, Any]], widths: list[int]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Every chunk's bins, one after another: coordinates, contents, squared weights.

    The squared weights are ``None`` unless every chunk kept them, which is
    what a histogram filled without weights looks like.
    """
    parts = [_chunk(chunk, widths) for chunk in chunks]
    if not parts:
        return np.zeros((0, len(widths)), np.int64), np.zeros(0), None
    squares = [part[2] for part in parts]
    variances = None if any(one is None for one in squares) else np.concatenate(squares)
    coordinates = np.concatenate([part[0] for part in parts])
    return coordinates, np.concatenate([part[1] for part in parts]), variances


def _chunk(
    chunk: dict[str, Any], widths: list[int]
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """One ``THnSparseArrayChunk``: its bins' coordinates, contents and squared weights."""
    size = int(chunk["fSingleCoordinateSize"])
    need = (sum(widths) + 7) // 8
    if size != need:
        raise FormatError(
            f"a sparse histogram's bins are {size} bytes of coordinates each where its "
            f"axes need {need}, so its bins cannot be told apart"
        )
    contents = np.asarray(chunk.get("fContent") if chunk.get("fContent") is not None else ())
    raw = np.asarray(chunk["fCoordinates"], dtype=np.int8).view(np.uint8)
    count = len(raw) // size
    if len(contents) < count:
        raise FormatError(
            f"a sparse histogram's chunk has {count} bins and contents for {len(contents)}"
        )
    squares = chunk.get("fSumw2")
    variances = None if squares is None else np.asarray(squares, dtype=np.float64)[:count]
    coordinates = _unpack(raw[: count * size].reshape(count, size), widths)
    return coordinates, contents[:count].astype(np.float64), variances
