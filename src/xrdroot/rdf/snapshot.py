"""``Snapshot``: the entries that reach a node, and the columns asked for, written to a file.

What ROOT's ``Snapshot`` writes, this writes with :func:`xrdroot.create` - or
:func:`xrdroot.update`, for ``mode="UPDATE"`` - as a ``TTree`` or, with
``rntuple=True``, an RNTuple: a column per column, numbers as numbers,
collections as rows of their own length, strings as strings. Each batch is
written as the loop comes to it, in the order of the entries, so a snapshot
far larger than memory costs one batch of it; the columns are declared from
the first batch that has anything in it.

If the loop fails the file is abandoned rather than closed, which leaves the
place it was going as it was: no file is better than half of one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..jagged import take
from ..rntuple.writer import spec_of as field_of
from ..wbranch import Vector
from ..writer import create
from ..wtree import spec_of as column_of
from ..wupdate import update
from .actions import Action
from .graph import Definition, Selector
from .values import Objects

if TYPE_CHECKING:
    from .loop import Batch

__all__ = ["Snapshot", "MODES"]

#: The :mod:`array` code of each number a ``std::vector`` branch holds, by NumPy's name.
VECTOR_CODES = {
    "int8": "b",
    "uint8": "B",
    "int16": "h",
    "uint16": "H",
    "int32": "i",
    "uint32": "I",
    "int64": "q",
    "uint64": "Q",
    "float32": "f",
    "float64": "d",
    "bool": "?",
}

#: What ``mode`` may be, as ROOT's ``RSnapshotOptions::fMode`` spells it.
MODES = ("RECREATE", "UPDATE")


class _Writing:
    """The file being written, and the tree or RNTuple in it once it is declared."""

    __slots__ = ("file", "table", "last", "written")

    def __init__(self, file: Any) -> None:
        self.file = file
        self.table: Any = None
        #: How many entries have gone in so far.
        self.written = 0
        #: The latest batch, kept when it held nothing, to declare the columns from at the end.
        self.last: dict[str, Any] | None = None


class Snapshot(Action):
    """Writes each batch's columns out as it comes; its result is where they went."""

    kind = "Snapshot"

    def __init__(
        self,
        node: Selector,
        inputs: Sequence[Definition],
        target: tuple[str, str],
        rntuple: bool,
        compression: str | None,
        mode: str,
        auto_flush: int = 0,
    ) -> None:
        super().__init__(node, inputs)
        self.tree_name, self.filename = target
        self.rntuple = rntuple
        self.compression = compression
        self.mode = mode
        #: ``RSnapshotOptions::fAutoFlush``: entries per cluster of a tree, when positive.
        self.auto_flush = auto_flush

    def partial(self, batch: Batch) -> Any:
        return {
            each.name: values
            for each, values in zip(self.inputs, self.columns(batch), strict=False)
        }

    def _open(self) -> _Writing:
        if self.mode == "UPDATE":
            return _Writing(update(self.filename, compression=self.compression))
        return _Writing(create(self.filename, compression=self.compression))

    def _declare(self, writing: _Writing, part: dict[str, Any]) -> None:
        if self.rntuple:
            fields = {name: field_of(name, values) for name, values in part.items()}
            writing.table = writing.file.rntuple(self.tree_name, fields)
        else:
            columns = {name: _tree_spec(name, values) for name, values in part.items()}
            writing.table = writing.file.tree(self.tree_name, columns)

    def merge(self, acc: Any, part: Any) -> Any:
        writing = self._open() if acc is None else acc
        count = len(next(iter(part.values()))) if part else 0
        if not count:
            writing.last = part
            return writing
        if writing.table is None:
            self._declare(writing, part)
        columns = _expanded(part, self.rntuple)
        if self.auto_flush > 0 and not self.rntuple:
            self._clustered(writing, columns, count)
        else:
            writing.table.extend(columns)
        return writing

    def _clustered(self, writing: _Writing, part: dict[str, Any], count: int) -> None:
        """A batch written in pieces that end where each cluster of ``auto_flush`` entries
        does, each cluster's baskets flushed as it closes, as ROOT's ``fAutoFlush`` has it."""
        start = 0
        while start < count:
            stop = min(count, start + self.auto_flush - writing.written % self.auto_flush)
            rows = np.arange(start, stop)
            writing.table.extend({name: take(values, rows) for name, values in part.items()})
            writing.written += stop - start
            if writing.written % self.auto_flush == 0:
                writing.table.flush()
            start = stop

    def finish(self, acc: Any, flow: dict[int, tuple[int, int]]) -> Any:
        if acc.table is None and acc.last is not None:
            self._declare(acc, acc.last)  # nothing passed: the columns, and no entries
        acc.file.close()
        return self.tree_name, self.filename

    def abort(self, acc: Any) -> None:
        if acc is not None:
            acc.file.__exit__(RuntimeError, None, None)


def _tree_spec(name: str, values: Any) -> Any:
    """A column's branch: a collection of numbers per entry - an ``RVec`` - is the
    ``std::vector`` branch ROOT's ``Snapshot`` writes it as, not a counted array, and a
    collection of objects a ``std::vector`` of their class, split as ROOT splits it."""
    if isinstance(values, Objects):
        return values.kind.spec(name)
    spec = column_of(name, values)
    if isinstance(spec, tuple) and spec[1] is None:
        return Vector(VECTOR_CODES[np.dtype(spec[0]).name])
    return spec


def _expanded(part: dict[str, Any], rntuple: bool) -> dict[str, Any]:
    """A batch's columns as a tree takes them: a collection of objects is the branch of
    objects ROOT writes, so one column becomes each column under it."""
    if rntuple:
        return part
    columns: dict[str, Any] = {}
    for name, values in part.items():
        if isinstance(values, Objects):
            columns.update(values.kind.columns(name, values.rows))
        else:
            columns[name] = values
    return columns
