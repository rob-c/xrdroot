"""A reader's numeric branch buffer, shared by its pointers, leaves and fill address.

Reading another entry reloads this buffer from the immutable batch. Merely
selecting the same entry again, or restarting the reader's cursor, does
not reload it. Each branch loads when its reader is first dereferenced.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .leaflist import TYPE_NAMES

__all__ = ["ReaderBuffer", "ReaderPointer", "buffer_for"]

#: Fixed numeric leaf types; text and object branches retain their existing readers.
DTYPES = {name: np.dtype(code) for code, name in TYPE_NAMES.items() if code != "C"}
DTYPES.update(Long_t=np.dtype("int64"), ULong_t=np.dtype("uint64"))


def _converted(value: Any, dtype: np.dtype[Any]) -> Any:
    """Convert before NumPy stores: narrow C++ integers wrap rather than raising on NumPy2."""
    if dtype.kind == "b":
        return bool(value)
    if dtype.kind in "iu":
        bits = dtype.itemsize * 8
        number = int(value) & ((1 << bits) - 1)
        return number - (1 << bits) if dtype.kind == "i" and number >> (bits - 1) else number
    return dtype.type(value)


class ReaderPointer:
    """A bound branch address; moving a reader to another tree does not retarget it."""

    _cint_cell = True

    def __init__(self, buffer: ReaderBuffer, index: int) -> None:
        self.buffer = buffer
        self.index = index

    @property
    def value(self) -> Any:
        return self[0]

    @value.setter
    def value(self, value: Any) -> None:
        self[0] = value

    def __len__(self) -> int:
        return self.buffer.size - self.index

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            if index.stop is not None or index.step is not None:
                raise TypeError("A reader pointer offset uses an open slice without a step.")
            start = 0 if index.start is None else operator.index(index.start)
            return self.buffer.reference(self.index + start)
        return self.buffer.item(self.index + operator.index(index))

    def __setitem__(self, index: int, value: Any) -> None:
        self.buffer.write(self.index + operator.index(index), value)


class ReaderBuffer:
    """One fixed numeric leaf's current memory, not a view that can modify saved entries."""

    def __init__(self, tree: Any, leaf: Any, dtype: np.dtype[Any]) -> None:
        self.tree = tree
        self.leaf = leaf
        self.dtype = dtype
        self.size: int = leaf.size
        self.entry = -1
        self.data = np.empty(self.size, dtype=dtype)
        self.address = tree._addresses.get(leaf.column)
        if self.address is None and tree._store is not None:
            self.address = tree._store.slots[leaf.column].address
        self._pointers: dict[int, ReaderPointer] = {}

    def read(self, entry: int) -> Any:
        if self.entry != entry:
            self.reload(entry)
        return self.peek()

    def reload(self, entry: int) -> None:
        # A direct branch GetEntry reads even the same entry into its existing storage.
        value = self.tree._batch.value(self.leaf.column, entry)
        if self.address is None:
            self.data[:] = np.asarray(value).reshape(-1)
        else:
            self.address.put(value.item() if isinstance(value, np.generic) else value)
        self.entry = entry

    def peek(self) -> Any:
        if self.address is not None:
            return self.address.get(None if self.size == 1 else self.size)
        return self.data[0] if self.size == 1 else self.data

    def item(self, index: int) -> Any:
        self._check(index)
        value = self.peek()
        found = value if self.size == 1 else value[index]
        return found.item() if isinstance(found, np.generic) else found

    def write(self, index: int, value: Any) -> None:
        self._check(index)
        converted = _converted(value, self.dtype)
        if self.address is None:
            self.data[index] = converted
        elif self.size == 1:
            self.address.put(converted)
        else:
            values = self.address.get(self.size)
            values[index] = converted
            self.address.put(values)

    def reference(self, index: int = 0) -> ReaderPointer:
        index = operator.index(index)
        if not 0 <= index <= self.size:
            raise IndexError(f"a reader buffer has {self.size} values, not address {index}")
        if index not in self._pointers:
            self._pointers[index] = ReaderPointer(self, index)
        return self._pointers[index]

    def _check(self, index: int) -> None:
        if not 0 <= index < self.size:
            raise IndexError(f"a reader buffer has {self.size} values, not element {index}")


def buffer_for(tree: Any, leaf: Any) -> ReaderBuffer | None:
    """Only fixed numeric leaves borrow mutable storage; collection/object reads stay intact."""
    dtype = DTYPES.get(leaf.typename)
    if dtype is None or leaf.text or leaf.vector or leaf.counter is not None:
        return None
    buffers: dict[str, ReaderBuffer] = tree._reader_buffers
    if leaf.column not in buffers:
        buffers[leaf.column] = ReaderBuffer(tree, leaf, dtype)
    return buffers[leaf.column]
