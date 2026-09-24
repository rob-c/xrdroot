"""Chains: one tree, written as many files, read as though it were one.

A dataset is rarely one file. It is a tree of the same name and the same
branches in each of a few hundred, and what analysis wants is the entries of
all of them end to end - which is what ROOT's ``TChain`` is, and what this
is. Entry numbers run on from one file into the next, a range crossing a
boundary reads from both sides of it, and ``iterate`` walks the lot in
batches that do not care where one file stops.

Files are opened when they are first needed, and each is opened once. The
one thing that needs every file is the number of entries - ``len``, a range
counted from the end, or an entry number in the second file, all need to
know how long the ones before it are - and that costs one small read of the
header, the key list and the tree's record per file, and no baskets.
"""

from __future__ import annotations

import glob
import os
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from xrdclient.url import parse

from .errors import UnsupportedFeatureError
from .tree import DEFAULT_STEP, _bounds, concatenate, scattered

if TYPE_CHECKING:
    from xrdclient.config import Config

    from .file import ROOTFile
    from .tree import TTree

__all__ = ["Chain", "ChainedBranch", "chain"]


def _expanded(source: Any) -> list[Any]:
    """One source as the files it stands for: a glob of local paths, or itself."""
    from .file import ROOTFile

    if isinstance(source, ROOTFile):
        return [source]
    text = os.fspath(source)
    url = parse(text)
    if not url.is_local or not any(mark in url.path for mark in "*?["):
        return [text]
    found = sorted(glob.glob(url.path))
    if not found:
        raise FileNotFoundError(f"{text!r} matches no file, and a chain of it would be empty")
    return found


class _Link:
    """One file of a chain: where it is, and the file and tree once opened."""

    __slots__ = ("target", "file", "tree", "owned")

    def __init__(self, target: Any) -> None:
        from .file import ROOTFile

        #: A path or URL, or a file somebody else opened.
        self.target = target
        self.owned = not isinstance(target, ROOTFile)
        self.file: ROOTFile | None = None if self.owned else target
        self.tree: TTree | None = None

    @property
    def name(self) -> str:
        return str(self.target) if self.owned else str(self.target.name)

    def portable(self) -> str:
        """What another process opens this file from."""
        if not self.owned:
            reopen = self.target._source._reopen
            if reopen is None:
                raise UnsupportedFeatureError(
                    f"{self.target.name} was opened from a file object, which cannot be "
                    f"sent to another process; build the chain from its URL to read it in one"
                )
            return str(reopen.target)
        return str(self.target)


class Chain:
    """Trees of one name across many files, read as one tree.

        >>> events = xrdroot.chain("Events", ["a.root", "b.root"])   # doctest: +SKIP
        >>> len(events), events["pt"].array(95, 105)
        >>> for batch in events.iterate(["pt", "eta"], step=100_000): ...

    Everything a :class:`~.tree.TTree` reads with, it reads with - ``[]``,
    ``arrays``, ``iterate``, ``keys``, ``typenames``, ``show`` - and entry
    numbers count across every file in the order they were given. A column
    whose type differs from one file to another is refused by name rather
    than read as two things. It closes the files it opened itself, and can
    be sent to a worker process, which opens them again there.
    """

    __slots__ = ("tree_name", "config", "_links", "_counts")

    def __init__(
        self, tree_name: str, sources: Iterable[Any], config: Config | None = None
    ) -> None:
        #: The name of the tree in every file.
        self.tree_name = tree_name
        #: How remote files are opened.
        self.config = config
        # One path or URL on its own is one source, not a run of characters.
        if isinstance(sources, (str, os.PathLike)) or not isinstance(sources, Iterable):
            sources = [sources]
        self._links = [_Link(target) for source in sources for target in _expanded(source)]
        if not self._links:
            raise ValueError(f"a chain of {tree_name!r} needs at least one file to hold it")
        self._counts: list[int] | None = None

    def __repr__(self) -> str:
        return f"<Chain {self.tree_name!r} over {len(self._links)} files>"

    @property
    def name(self) -> str:
        return self.tree_name

    @property
    def files(self) -> list[str]:
        """Every file of the chain, in the order its entries come in."""
        return [link.name for link in self._links]

    def tree(self, index: int) -> TTree:
        """The tree in one file of the chain, opening the file if it is not open yet."""
        link = self._links[index]
        if link.tree is None:
            if link.file is None:
                from .file import open_root

                link.file = open_root(link.target, config=self.config)
            link.tree = link.file[self.tree_name]
        return link.tree

    @property
    def counts(self) -> list[int]:
        """How many entries each file holds, which opens every file the first time."""
        if self._counts is None:
            self._counts = [len(self.tree(index)) for index in range(len(self._links))]
        return list(self._counts)

    @property
    def num_entries(self) -> int:
        return sum(self.counts)

    def __len__(self) -> int:
        return self.num_entries

    def starts(self) -> list[int]:
        """The chain's number for the first entry of each file, and one past the last."""
        starts = [0]
        for count in self.counts:
            starts.append(starts[-1] + count)
        return starts

    def spans(self, start: int, stop: int) -> Iterator[tuple[int, int, int]]:
        """Which files hold entries ``[start, stop)`` of the chain, and which of theirs."""
        starts = self.starts()
        for index in range(len(self._links)):
            low, high = max(start, starts[index]), min(stop, starts[index + 1])
            if low < high:
                yield index, low - starts[index], high - starts[index]

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __contains__(self, name: object) -> bool:
        return name in self.tree(0).branches

    def __getitem__(self, name: str) -> ChainedBranch:
        if name not in self.tree(0).branches:
            raise KeyError(
                f"{name!r} is not a branch of {self.tree_name!r} in {self._links[0].name}; "
                f"there is " + ", ".join(self.tree(0).branches)
            )
        return ChainedBranch(self, name)

    def keys(self) -> list[str]:
        """Every column of the first file's tree, which every file is taken to share."""
        return self.tree(0).keys()

    def readable(self) -> list[str]:
        """The columns ``arrays`` reads when given no names: the first file's readable ones."""
        return self.tree(0).readable()

    def typenames(self) -> dict[str, str]:
        """What each column of the first file's tree holds, in Python's words."""
        return self.tree(0).typenames()

    def show(self) -> str:
        """The first file's one-line-per-column summary."""
        return self.tree(0).show()

    def arrays(
        self,
        names: Sequence[str] | None = None,
        entry_start: int = 0,
        entry_stop: int | None = None,
        *,
        library: str = "np",
        entries: Any = None,
    ) -> Any:
        """Several columns at once, over the same range of the chain's entries.

        As :meth:`~.tree.TTree.arrays`, with the entries counted across every
        file; ``entries`` is a set of those numbers, or an
        :class:`~.entries.EntryList` - one made over a chain has a list per
        file, and each file's is taken from it.
        """
        from .library import convert

        wanted = self.readable() if names is None else list(names)
        if entries is None:
            columns = {name: self[name].array(entry_start, entry_stop) for name in wanted}
        else:
            rows = self._selected(entries)
            columns = {name: self[name].pick(rows) for name in wanted}
        return convert(columns, library)

    def _selected(self, entries: Any) -> np.ndarray[Any, Any]:
        from .entries import EntryList, selected

        if isinstance(entries, EntryList) and entries.lists:
            starts = self.starts()
            pieces = [np.zeros(0, np.int64)]
            for index, link in enumerate(self._links):
                for one in entries.lists_for(self.tree_name, link.name):
                    pieces.append(one.entries + starts[index])
            entries = np.concatenate(pieces)
        return selected(entries, self.num_entries)

    def iterate(
        self,
        names: Sequence[str] | None = None,
        *,
        step: int = DEFAULT_STEP,
        entry_start: int = 0,
        entry_stop: int | None = None,
        library: str = "np",
    ) -> Iterator[Any]:
        """Walk the chain in batches of ``step`` entries, whichever files they are in.

        A batch that runs off the end of one file carries on into the next,
        so every batch but the last is ``step`` long however the files are
        cut.
        """
        if step <= 0:
            raise ValueError("step must be at least one entry")
        at, stop = _bounds(self.num_entries, entry_start, entry_stop)
        while at < stop:
            yield self.arrays(names, at, min(at + step, stop), library=library)
            at += step

    def close(self) -> None:
        """Close the files this chain opened; those it was given are left open."""
        for link in self._links:
            if link.owned and link.file is not None:
                link.file.close()
                link.file, link.tree = None, None

    def __enter__(self) -> Chain:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        """Where the files are, and how long each is if that is known yet."""
        return {
            "tree_name": self.tree_name,
            "targets": [link.portable() for link in self._links],
            "config": self.config,
            "counts": self._counts,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Arrive in another process knowing the files, and opening none of them yet."""
        self.__init__(state["tree_name"], state["targets"], state["config"])  # type: ignore[misc]
        self._counts = state["counts"]


class ChainedBranch:
    """One column of a chain: the same branch in every file, read end to end.

        >>> events["pt"].array(0, 1_000_000)          # doctest: +SKIP

    Arrays of numbers are joined as NumPy joins them, :class:`~.tree.Jagged`
    rows with each file's offsets carried on from the last, and lists of
    values as one list.
    """

    __slots__ = ("chain", "name")

    def __init__(self, chain: Chain, name: str) -> None:
        self.chain = chain
        self.name = name

    def __repr__(self) -> str:
        return f"<ChainedBranch {self.name!r} over {len(self.chain.files)} files>"

    def __len__(self) -> int:
        return len(self.chain)

    def _branch(self, index: int) -> Any:
        tree = self.chain.tree(index)
        if self.name not in tree.branches:
            raise KeyError(
                f"{self.name!r} is a branch of {self.chain.tree_name!r} in "
                f"{self.chain.files[0]} but not in {self.chain.files[index]}"
            )
        return tree.branches[self.name]

    @property
    def typename(self) -> str | None:
        """What the column holds, which every file of the chain has to agree on.

        Finding out opens every file, since a file that disagrees is a column
        that cannot be read as one thing, and that is refused by name.
        """
        first: str | None = self._branch(0).typename
        for index in range(1, len(self.chain.files)):
            other = self._branch(index).typename
            if other != first:
                raise UnsupportedFeatureError(
                    f"{self.name!r} holds {first} in {self.chain.files[0]} and {other} in "
                    f"{self.chain.files[index]}, and a chain reads a column as one type; "
                    f"read the files apart, or convert one of them"
                )
        return first

    def array(
        self, entry_start: int = 0, entry_stop: int | None = None, *, entries: Any = None
    ) -> Any:
        """The values of a range of the chain's entries, from whichever files hold them."""
        if entries is not None:
            return self.pick(self.chain._selected(entries))
        self.typename  # noqa: B018 - asked for the refusal a disagreement makes
        start, stop = _bounds(len(self.chain), entry_start, entry_stop)
        pieces = [
            self._branch(index).array(low, high)
            for index, low, high in self.chain.spans(start, stop)
        ]
        return concatenate(pieces) if pieces else self._branch(0).array(0, 0)

    def pick(self, rows: np.ndarray[Any, Any]) -> Any:
        """The values of the chain's entries numbered in ``rows``, in that order."""
        self.typename  # noqa: B018 - asked for the refusal a disagreement makes
        starts = self.chain.starts()

        def read(home: int, inside: np.ndarray[Any, Any]) -> Any:
            return self._branch(home).pick(inside - starts[home])

        return scattered(rows, starts, read, lambda: self._branch(0).array(0, 0))


def chain(tree_name: str, sources: Iterable[Any], *, config: Config | None = None) -> Chain:
    """The tree ``tree_name`` of every file in ``sources``, read as one tree.

        >>> events = xrdroot.chain("Events", ["run1/*.root", "root://host//store/x.root"])
        ...                                                    # doctest: +SKIP

    ``sources`` are paths, URLs, or files already open - or one of those on its
    own, such as ``"run1/*.root"``; a local path with a
    ``*``, ``?`` or ``[`` in it is a glob and stands for every file it
    matches, in sorted order. Nothing is opened until something needs it,
    and ``config`` is how remote ones are opened when it does.
    """
    return Chain(tree_name, sources, config)
