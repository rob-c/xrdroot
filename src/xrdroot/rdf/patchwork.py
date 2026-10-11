"""Data some of whose entries have no value of a column: trees that differ, friends that match.

:class:`Patched` reads trees of different names - a chain of ``file?#tree``
parts - one after another, each with the branches it has: a column is
every branch of that name, and an entry of a tree without it has no value.
:class:`Befriended` reads a tree beside its friends, each column of a
friend as ``alias.name``: the friend's entry of the same number, or - for a
friend with an index - the one whose index values are the tree's own, as
``TTreeIndex`` matches them; an entry with no match has no value of the
friend's columns. :meth:`~.sources.Source.available` says which entries
have a value, for :mod:`.missing`'s ``DefaultValueFor`` and filters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..errors import UnsupportedFeatureError
from ..file import open_root
from ..tree import concatenate
from .sources import SampleInfo, Source, Table

__all__ = ["Befriended", "Patched"]


def _absent(name: str, count: int, like: Any) -> Any:
    """``count`` zeros of the type of ``like``, for entries with no value of ``name``."""
    if not isinstance(like, np.ndarray) or like.ndim != 1:
        raise UnsupportedFeatureError(
            f"the column {name!r} is missing in some entries, and holds more than one number "
            f"per entry; such a column is read here only where every entry has it"
        )
    return np.zeros(count, dtype=like.dtype)


class Patched(Source):
    """Trees of their own names in their own files, read one after another."""

    owned = True

    def __init__(self, parts: Sequence[tuple[str, str]]) -> None:
        self.parts = list(parts)
        self._tables: list[Table] = []

    def __getstate__(self) -> dict[str, Any]:
        return {"parts": self.parts}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["parts"])  # type: ignore[misc]

    def tables(self) -> list[Table]:
        """Each part's tree, opened the first time anything is asked."""
        if not self._tables:
            for path, tree in self.parts:
                file = open_root(path)
                self._tables.append(Table(file[tree], owned=True, file=file))
        return self._tables

    def __len__(self) -> int:
        return sum(len(one) for one in self.tables())

    def names(self) -> list[str]:
        return list(dict.fromkeys(name for one in self.tables() for name in one.names()))

    def readable(self) -> list[str]:
        return list(dict.fromkeys(name for one in self.tables() for name in one.readable()))

    def boundaries(self) -> list[int]:
        return [0, *np.cumsum([len(one) for one in self.tables()]).tolist()]

    def _spans(self, start: int, stop: int) -> list[tuple[Table, int, int]]:
        """Each part's share of entries ``start`` to ``stop``, counted in the part."""
        bounds = self.boundaries()
        return [
            (one, max(start, low) - low, min(stop, high) - low)
            for one, low, high in zip(self.tables(), bounds, bounds[1:], strict=False)
            if max(start, low) < min(stop, high)
        ]

    def _like(self, name: str) -> Any:
        """A read of nothing of ``name`` from a part that has it: the column's type."""
        return next(one.read(name, 0, 0) for one in self.tables() if name in one.names())

    def read(self, name: str, start: int, stop: int) -> Any:
        pieces = [
            one.read(name, low, high) if name in one.names()
            else _absent(name, high - low, self._like(name))
            for one, low, high in self._spans(start, stop)
        ]  # fmt: skip
        return concatenate(pieces) if pieces else self._like(name)

    def available(self, name: str, start: int, stop: int) -> Any:
        if all(name in one.names() for one in self.tables()):
            return None
        pieces = [
            np.full(high - low, name in one.names()) for one, low, high in self._spans(start, stop)
        ]
        return np.concatenate(pieces) if pieces else np.zeros(0, np.bool_)

    def sample(self, index: int) -> SampleInfo:
        bounds = self.boundaries()
        path, tree = self.parts[index]
        return SampleInfo(f"{path}/{tree}", (bounds[index], bounds[index + 1]))

    def cxx_type(self, name: str) -> str | None:
        return None

    def describe(self) -> str:
        return "TChain of " + ", ".join(f"{path}?#{tree}" for path, tree in self.parts)

    def close(self) -> None:
        for one in self._tables:
            one.close()
        self._tables = []


def _keys(source: Source, name: str, start: int, stop: int) -> Any:
    """An index's values for entries ``start`` to ``stop``: a column's, or a number's."""
    if name in source.names():
        return np.asarray(source.read(name, start, stop)).astype(np.int64)
    return np.full(stop - start, int(float(name)), dtype=np.int64)


class Friend:
    """One friend of a tree: what it is called, its data, and the names it is indexed by."""

    def __init__(self, alias: str, source: Source, index: tuple[str, str] | None = None) -> None:
        self.alias, self.source, self.index = alias, source, index
        self._lookup: dict[tuple[int, int], int] | None = None
        self._columns: dict[str, Any] = {}

    def lookup(self) -> dict[tuple[int, int], int]:
        """``TTreeIndex``: the first entry for each pair of index values."""
        if self._lookup is None:
            assert self.index is not None
            count = len(self.source)
            major, minor = (_keys(self.source, name, 0, count) for name in self.index)
            self._lookup = {}
            for entry, key in enumerate(zip(major.tolist(), minor.tolist(), strict=True)):
                self._lookup.setdefault(key, entry)
        return self._lookup

    def column(self, name: str) -> Any:
        """Every entry's value of ``name``, read once."""
        if name not in self._columns:
            self._columns[name] = self.source.read(name, 0, len(self.source))
        return self._columns[name]


class Befriended(Source):
    """A tree's data and its friends', each friend's columns named ``alias.name``."""

    def __init__(self, main: Source, friends: Sequence[Friend]) -> None:
        self.main, self.friends = main, list(friends)
        self.owned = main.owned

    def __len__(self) -> int:
        return len(self.main)

    def names(self) -> list[str]:
        theirs = [f"{one.alias}.{name}" for one in self.friends for name in one.source.names()]
        return list(dict.fromkeys([*self.main.names(), *theirs]))

    def readable(self) -> list[str]:
        theirs = [f"{one.alias}.{name}" for one in self.friends for name in one.source.readable()]
        return list(dict.fromkeys([*self.main.readable(), *theirs]))

    def boundaries(self) -> list[int]:
        return self.main.boundaries()

    def sample(self, index: int) -> SampleInfo:
        return self.main.sample(index)

    def cxx_type(self, name: str) -> str | None:
        found = self._split(name)
        return self.main.cxx_type(name) if found is None else found[0].source.cxx_type(found[1])

    def describe(self) -> str:
        return self.main.describe() + ", with friends " + ", ".join(f.alias for f in self.friends)

    def _split(self, name: str) -> tuple[Friend, str] | None:
        """The friend and its column a name of ``alias.name`` reads, or ``None`` for the tree's."""
        if name in self.main.names():
            return None
        alias, _, column = name.partition(".")
        return next(((one, column) for one in self.friends if one.alias == alias), None)

    def _entries(self, friend: Friend, start: int, stop: int) -> Any:
        """The friend's entry for each of the tree's from ``start`` to ``stop``; -1 for none."""
        if friend.index is None:
            numbers = np.arange(start, stop, dtype=np.int64)
            numbers[numbers >= len(friend.source)] = -1
            return numbers
        lookup = friend.lookup()
        major, minor = (_keys(self.main, name, start, stop) for name in friend.index)
        keys = zip(major.tolist(), minor.tolist(), strict=True)
        return np.array([lookup.get(key, -1) for key in keys], dtype=np.int64)

    def read(self, name: str, start: int, stop: int) -> Any:
        found = self._split(name)
        if found is None:
            return self.main.read(name, start, stop)
        friend, column = found
        if friend.index is None and stop <= len(friend.source):
            return friend.source.read(column, start, stop)
        entries = self._entries(friend, start, stop)
        every = friend.column(column)
        _absent(name, 0, every)  # a column of more than one number an entry is refused
        values = every[np.maximum(entries, 0)]
        values[entries < 0] = 0
        return values

    def available(self, name: str, start: int, stop: int) -> Any:
        found = self._split(name)
        if found is None:
            return self.main.available(name, start, stop)
        friend = found[0]
        if friend.index is None and stop <= len(friend.source):
            return None
        return self._entries(friend, start, stop) >= 0

    def close(self) -> None:
        for one in [self.main, *(friend.source for friend in self.friends)]:
            if one.owned:
                one.close()
