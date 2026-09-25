"""Where a frame's entries come from: a tree, a chain, an RNTuple, or nothing at all.

Every source answers the same few questions - how many entries, which
columns, a column's values for a range of entries, where one file ends and
the next begins - so the event loop never asks which kind it has. A range is
read with the branch's own ``array``, which reads the baskets (or pages) that
hold it and keeps the last one, so consecutive batches never read a basket
twice. A source also pickles, so that a worker process can open it again:
a tree or a chain by where its files are, an RNTuple the same way.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..chain import Chain, _expanded
from ..errors import UnsupportedFeatureError
from ..file import ROOTFile, open_root
from ..formula.select import TreeNames
from ..rntuple import RNTuple
from ..tree import TTree, concatenate

__all__ = ["Source", "Table", "Empty", "Joined", "SampleInfo", "open_named", "wrap"]


class SampleInfo:
    """``rdfsampleinfo_``: which file of the data a batch of entries is from.

    What a ``DefinePerSample`` callable is given: ``id`` is the file and the
    tree, ``"file.root/Events"``, as ROOT spells it, and ``entries`` the range
    of entry numbers the file holds in the data as a whole.
    """

    __slots__ = ("id", "entries")

    def __init__(self, id: str, entries: tuple[int, int]) -> None:
        self.id = id
        self.entries = entries

    def __repr__(self) -> str:
        return f"<SampleInfo {self.id!r} entries {self.entries[0]} to {self.entries[1]}>"

    def AsString(self) -> str:
        return self.id

    def Contains(self, text: str) -> bool:
        return text in self.id

    def EntryRange(self) -> tuple[int, int]:
        return self.entries

    as_string = AsString
    contains = Contains
    entry_range = EntryRange


class Source:
    """What an event loop reads from."""

    #: Whether closing the frame closes this: true for files the frame opened.
    owned = False

    def __len__(self) -> int:
        raise NotImplementedError

    def names(self) -> list[str]:
        """Every column a frame over this can use, by name."""
        raise NotImplementedError

    def readable(self) -> list[str]:
        """The columns read when nobody says which: ``AsNumpy()``, ``Snapshot``."""
        return self.names()

    def read(self, name: str, start: int, stop: int) -> Any:
        """One column's values for the entries from ``start`` up to ``stop``."""
        raise NotImplementedError

    def boundaries(self) -> list[int]:
        """Where each file's entries start, and where the last one's stop."""
        return [0, len(self)]

    def sample(self, index: int) -> SampleInfo:
        """What the file at ``index`` of :meth:`boundaries` is: nothing, when there is none."""
        bounds = self.boundaries()
        return SampleInfo("", (bounds[index], bounds[index + 1]))

    def cxx_type(self, name: str) -> str | None:
        """The C++ type the data declares a column as, when it says so itself."""
        raise NotImplementedError

    def describe(self) -> str:
        """What the data is, in words, for ``Describe`` and a frame's ``repr``."""
        raise NotImplementedError

    def close(self) -> None:
        """Close what this opened; asked only of a source that is :attr:`owned`."""
        raise NotImplementedError


def _tree_file(tree: TTree) -> str:
    return str(tree._source.name)


class Table(Source):
    """A tree, a chain of them, or an RNTuple: anything read a column at a time by name."""

    def __init__(self, table: Any, owned: bool = False, file: ROOTFile | None = None) -> None:
        self.table = table
        self.owned = owned
        #: The file this opened to find the table in, closed with it.
        self.file = file

    def __len__(self) -> int:
        return len(self.table)

    def names(self) -> list[str]:
        if isinstance(self.table, RNTuple):
            return list(self.table.keys())
        return TreeNames(self.table).names

    def readable(self) -> list[str]:
        return list(self.table.readable())

    def read(self, name: str, start: int, stop: int) -> Any:
        return self.table[name].array(start, stop)

    def boundaries(self) -> list[int]:
        if isinstance(self.table, Chain):
            return list(self.table.starts())
        return [0, len(self.table)]

    def sample(self, index: int) -> SampleInfo:
        bounds = self.boundaries()
        where = self.table.files[index] if isinstance(self.table, Chain) else self._file()
        return SampleInfo(f"{where}/{self.table.name}", (bounds[index], bounds[index + 1]))

    def _file(self) -> str:
        if isinstance(self.table, TTree):
            return _tree_file(self.table)
        return str(self.table._store.source.name)

    def cxx_type(self, name: str) -> str | None:
        if isinstance(self.table, RNTuple):
            return self.table.cxx_types().get(name)
        return None

    def describe(self) -> str:
        kind = type(self.table).__name__
        if isinstance(self.table, Chain):
            return f"{kind} {self.table.name!r} of " + ", ".join(self.table.files)
        return f"{kind} {self.table.name!r} in {self._file()}"

    def close(self) -> None:
        if isinstance(self.table, Chain):
            self.table.close()
        if self.file is not None:
            self.file.close()


class Empty(Source):
    """``RDataFrame(n)``: ``n`` entries with no columns, for ``Define`` to fill in."""

    def __init__(self, entries: int) -> None:
        if isinstance(entries, bool) or entries < 0:
            raise ValueError(f"an empty frame has a number of entries, and {entries!r} is not one")
        self.entries = int(entries)

    def __len__(self) -> int:
        return self.entries

    def names(self) -> list[str]:
        return []

    def describe(self) -> str:
        return f"{self.entries} empty entries"


class Joined(Source):
    """RNTuples of one name in several files, read end to end as a chain reads trees."""

    owned = True

    def __init__(self, name: str, targets: Sequence[str]) -> None:
        self.name = name
        self.targets = list(targets)
        self._parts: list[Table] = []

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "targets": self.targets}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["name"], state["targets"])  # type: ignore[misc]

    def parts(self) -> list[Table]:
        """Every file's RNTuple, opened the first time anything is asked."""
        if not self._parts:
            for target in self.targets:
                file = open_root(target)
                self._parts.append(Table(file[self.name], owned=True, file=file))
        return self._parts

    def __len__(self) -> int:
        return sum(len(part) for part in self.parts())

    def names(self) -> list[str]:
        return self.parts()[0].names()

    def readable(self) -> list[str]:
        return self.parts()[0].readable()

    def cxx_type(self, name: str) -> str | None:
        return self.parts()[0].cxx_type(name)

    def boundaries(self) -> list[int]:
        counts = [len(part) for part in self.parts()]
        return [0, *np.cumsum(counts).tolist()]

    def read(self, name: str, start: int, stop: int) -> Any:
        bounds = self.boundaries()
        pieces = [
            part.read(name, max(start, low) - low, min(stop, high) - low)
            for part, low, high in zip(self.parts(), bounds, bounds[1:])
            if max(start, low) < min(stop, high)
        ]
        return concatenate(pieces) if pieces else self.parts()[0].read(name, 0, 0)

    def sample(self, index: int) -> SampleInfo:
        bounds = self.boundaries()
        return SampleInfo(f"{self.targets[index]}/{self.name}", (bounds[index], bounds[index + 1]))

    def describe(self) -> str:
        return f"RNTuple {self.name!r} of " + ", ".join(self.targets)

    def close(self) -> None:
        for part in self._parts:
            part.close()
        self._parts = []


def _targets(files: Any) -> list[str]:
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    found = [str(target) for source in files for target in _expanded(source)]
    if not found:
        raise ValueError("an RDataFrame of a tree by name needs at least one file to find it in")
    return found


def open_named(name: str, files: Any) -> Source:
    """The tree or RNTuple called ``name`` in ``files``: a path, a URL, a glob, or several."""
    targets = _targets(files)
    first = open_root(targets[0])
    try:
        kind = first.classnames().get(name.rpartition(";")[0] or name)
    finally:
        first.close()
    if kind == "ROOT::RNTuple":
        return Joined(name, targets)
    return Table(Chain(name, targets), owned=True)


def wrap(data: Any) -> Source:
    """A tree, chain or RNTuple handed over as it is, which the frame leaves open."""
    if isinstance(data, (TTree, Chain, RNTuple)):
        return Table(data)
    raise UnsupportedFeatureError(
        f"an RDataFrame is made from a TTree, a Chain, an RNTuple, a tree's name and its "
        f"files, or a number of empty entries, and not from a {type(data).__name__}"
    )
