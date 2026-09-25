"""Every shape a column arrives in, as one: flat values and a ladder of offsets.

``TTree.arrays`` gives a number per entry as a NumPy array, a fixed-size
array as a NumPy array of more dimensions, a variable-length one as a
:class:`~xrdroot.tree.Jagged`, strings as a list of ``str`` and STL
containers as lists of lists. An expression indexes and loops over all of
them the same way, so each is put into one layout here: every value at the
bottom in one flat array, and above it one offsets array per dimension,
saying where each entry's - or each element's - children start and stop.
A fixed dimension is offsets that step evenly; a variable one is the
offsets the file gave.

Each offsets array keeps one extra copy of its last value at the end. That
makes the node one past the last a real node with no children, which is
where a cursor that has gone out of range is sent: whatever it asks for from
there on is empty, and nothing has to check it first.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..tree import Jagged

__all__ = ["Layout", "layout"]

Array = Any

#: What a row of a list-of-rows column can be.
SEQUENCES = (list, tuple, np.ndarray)


class Layout:
    """One column: the values at the bottom, and the offsets of each dimension above them."""

    __slots__ = ("name", "content", "offsets")

    def __init__(self, name: str, content: Array, offsets: list[Array]) -> None:
        self.name = name
        self.content = content
        #: One per dimension, outermost first, each with its last value repeated.
        self.offsets = [np.append(level, level[-1] if len(level) else 0) for level in offsets]

    @property
    def dims(self) -> int:
        return len(self.offsets)

    @property
    def rows(self) -> int:
        """How many entries the column holds."""
        return len(self.offsets[0]) - 2 if self.offsets else len(self.content)

    def nodes(self, level: int) -> int:
        """How many nodes there are one level below ``level``: where out of range points."""
        if level + 1 < self.dims:
            return len(self.offsets[level + 1]) - 2
        return len(self.content)

    def sizes(self, level: int, node: Array) -> Array:
        """How many children each node has at ``level``."""
        offsets = self.offsets[level]
        return offsets[node + 1] - offsets[node]

    def child(self, level: int, node: Array, index: Array, ok: Array) -> Array:
        """The ``index``-th child of each node, or the empty node past the end where not ``ok``."""
        return np.where(ok, self.offsets[level][node] + index, self.nodes(level))

    def values(self, node: Array) -> Array:
        """The values at the bottom, for nodes that point at them."""
        if not len(self.content):
            return np.zeros(len(node), self.content.dtype)
        return self.content[np.minimum(node, len(self.content) - 1)]


def _fixed(shape: tuple[int, ...]) -> list[Array]:
    """The offsets of the fixed dimensions of an array of ``shape``, outermost first."""
    levels = []
    for depth in range(1, len(shape)):
        parents = int(np.prod(shape[:depth]))
        levels.append(np.arange(parents + 1, dtype=np.int64) * shape[depth])
    return levels


def _array(name: str, value: Array) -> Layout:
    if value.dtype == object:
        return _objects(name, list(value))
    if value.dtype.kind == "S":
        value = value.astype(str)
    return Layout(name, value.reshape(-1), _fixed(value.shape))


def _jagged(name: str, value: Jagged) -> Layout:
    content = value.content
    inner = _fixed(content.shape)
    return Layout(name, content.reshape(-1), [value.offsets, *inner])


def _offsets(lengths: Array) -> Array:
    offsets = np.zeros(len(lengths) + 1, np.int64)
    np.cumsum(lengths, out=offsets[1:])
    return offsets


def _is_level(name: str, values: list[Any]) -> bool:
    """Are these rows of something, rather than the values at the bottom?"""
    rows = [isinstance(value, SEQUENCES) for value in values]
    if any(rows) and not all(rows):
        raise UnsupportedFeatureError(
            f"{name!r} mixes rows with single values in one level, which is not one "
            f"column an expression can index"
        )
    return bool(rows) and rows[0]


def _leaves(name: str, values: list[Any]) -> Array:
    if any(isinstance(value, dict) for value in values):
        raise UnsupportedFeatureError(
            f"{name!r} holds whole objects or maps, one dictionary per value, which an "
            f"expression cannot do arithmetic on; name the split members instead, as "
            f"{name}.member, or read the column itself with arrays()"
        )
    if values and all(isinstance(value, str) for value in values):
        return np.asarray(values, dtype=str)
    if not values:
        return np.zeros(0, np.float64)
    return np.asarray(values)


def _objects(name: str, values: list[Any]) -> Layout:
    """Lists of values or of rows, a Python object per entry, made flat level by level."""
    levels = []
    while _is_level(name, values):
        levels.append(_offsets(np.fromiter(map(len, values), np.int64, len(values))))
        values = list(itertools.chain.from_iterable(values))
    return Layout(name, _leaves(name, values), levels)


def layout(name: str, value: Any, dims: int | None = None) -> Layout:
    """The layout of one column as ``TTree.arrays`` gives it.

    ``dims``, when known, is how many dimensions the branch has: a batch whose
    every row is empty cannot say how deep its rows would have gone, and is
    given that many empty levels.
    """
    if isinstance(value, Jagged):
        found = _jagged(name, value)
    elif isinstance(value, np.ndarray):
        found = _array(name, value)
    else:
        found = _objects(name, list(value))
    while dims is not None and found.dims < dims and not len(found.content):
        found.offsets.append(np.zeros(2, np.int64))
    return found
