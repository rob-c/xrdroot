"""``rootcp``: objects copied from one ROOT file into another, and trees cut on the way.

    >>> xrdroot.copy("in.root", "out.root", ["hists/*", "events"])          # doctest: +SKIP
    >>> xrdroot.copy("in.root", "skim.root", ["events"], cut="nMuon >= 2")  # doctest: +SKIP

What is copied goes across as it was wherever it can: an object as the very
record it was, its classes described in the new file as the old one
described them, and a tree as its baskets, byte for byte, behind a new
``TTree`` record. Nothing is decoded that need not be, so a copy is as
faithful as the bytes and costs little more than reading them.

A tree given a ``cut``, or ``columns`` computed from expressions, is
``TTree::CopyTree``: its entries are read a batch at a time through the
formula engine - ``tree.arrays(..., cut=...)`` - and only those that pass
are written, through :class:`~xrdroot.WritableTree`. ``columns`` naming
branches alone keeps the fast way: only those branches' baskets go across,
with the counters their runs need.
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from ..errors import UnsupportedFeatureError
from ..file import Directory, Key, ROOTFile
from ..objects import TREE_CLASSES
from ..writer import WritableDirectory, WritableFile, create
from ..wupdate import update
from .files import DIRECTORIES, NTUPLE, _is_local_file, _local_path, _opened, resolve_compression
from .ntuples import NtupleMerge
from .records import copy_record, fits
from .trees import TreeMerge, normal_codes

__all__ = ["copy"]


def _everything(directory: Directory, prefix: str = "") -> list[tuple[str, Key]]:
    """Every name in ``directory`` and below, with its newest key, directories first."""
    found: list[tuple[str, Key]] = []
    newest: dict[str, Key] = {}
    for key in directory._keys:
        newest[key.name] = max(newest.get(key.name, key), key, key=lambda one: one.cycle)
    for name, key in newest.items():
        path = f"{prefix}{name}"
        found.append((path, key))
        if key.classname in DIRECTORIES:
            found.extend(_everything(directory._subdirectory(key), f"{path}/"))
    return found


def _chosen(top: ROOTFile, keys: Any) -> list[tuple[str, str]]:
    """Which paths are copied, and the path each goes to.

    ``None`` is everything at the top of the file, directories and all; a
    string or a list of them are paths - or shell patterns of paths - each
    copied to the same path; a mapping gives each path a new one.
    """
    if isinstance(keys, Mapping):
        return [(str(path), str(to)) for path, to in keys.items()]
    everything = [path for path, _key in _everything(top)]
    if keys is None:
        return [(path, path) for path in everything if "/" not in path]
    patterns = [keys] if isinstance(keys, str) else list(keys)
    return [(path, path) for path in _outermost(_picked(top, everything, patterns))]


def _picked(top: ROOTFile, everything: list[str], patterns: list[Any]) -> list[str]:
    """Every path any of the patterns picks out, in the order the patterns come."""
    matched: list[str] = []
    for pattern in patterns:
        matched.extend(_matched(top, everything, str(pattern)))
    return matched


def _matched(top: ROOTFile, everything: list[str], pattern: str) -> list[str]:
    """The paths a pattern picks out, or a refusal naming what there is instead."""
    found = [path for path in everything if fnmatch.fnmatchcase(path, pattern)]
    if not found:
        raise KeyError(f"{pattern!r} is nothing in {top.name}; there is {', '.join(everything)}")
    return found


def _outermost(paths: list[str]) -> list[str]:
    """The paths not inside another of them, each once: a directory copied brings what is in it."""
    kept = dict.fromkeys(paths)
    return [path for path in kept if not any(path.startswith(f"{other}/") for other in kept)]


def _located(top: ROOTFile, path: str) -> tuple[Directory, Key]:
    """The directory a path's last part is in, and that part's newest key."""
    head, slash, name = path.rpartition("/")
    directory: Directory = top[head] if slash else top
    if not isinstance(directory, Directory):
        raise KeyError(f"{head!r} in {top.name} is not a directory, so {path!r} is not there")
    return directory, directory._key(name)


def _with_counters(tree: Any, names: Sequence[str]) -> list[str]:
    """The branches ``names`` asks for, each run's counter coming along before it."""
    by_leaf = {id(branch.leaf): name for name, branch in tree.branches.items()}
    wanted: dict[str, None] = {}
    for name in names:
        count = tree[name].leaf.count
        counter = by_leaf.get(id(count)) if count is not None else None
        if counter is not None:
            wanted[counter] = None
        wanted[name] = None
    return list(wanted)


class _Copier:
    """One copy in progress: where from, where to, and what to do to its trees."""

    def __init__(
        self,
        top: ROOTFile,
        out: WritableDirectory,
        filter_: Callable[[str], bool] | None,
        cut: str | None,
        columns: Sequence[str] | Mapping[str, str] | None,
        fast: bool,
    ) -> None:
        self.top = top
        self.out = out
        self.filter = filter_
        self.cut = cut
        self.columns = columns
        self.fast = fast
        self.verbatim = normal_codes(top.compression) == normal_codes(out._codes)
        self.infos: dict[str, dict[tuple[str, int], bytes]] = {}
        self.written: list[str] = []

    def copy(self, path: str, to: str) -> None:
        directory, key = _located(self.top, path)
        if key.classname in DIRECTORIES:
            self.out.mkdir(to)
            self.written.append(to)
            below = directory._subdirectory(key)
            for name in below.keys():
                self.copy(f"{path}/{name}", f"{to}/{name}")
            return
        here, leaf = self.out._place(to)
        if key.classname in TREE_CLASSES:
            self._tree(directory, key, here, leaf, path)
        elif key.classname == NTUPLE:
            ntuple = directory._read(key)
            NtupleMerge(here, leaf, ntuple).add(ntuple, f"{path!r} in {self.top.name}")
        else:
            self._object(directory, key, here, leaf)
        self.written.append(to)

    def _object(self, directory: Directory, key: Key, here: WritableDirectory, leaf: str) -> None:
        """An object as the record it was - or, under a name of another length, decoded and
        written again, which a class this library writes can be and any other is refused."""
        if fits(here, leaf, key):
            copy_record(here, leaf, key, directory._source, self.infos)
            return
        value = directory._read(key)
        if isinstance(value, Mapping):  # members only: a class the writer has no layout for
            raise UnsupportedFeatureError(
                f"{key.name!r}, a {key.classname} in {self.top.name}, cannot go under a name "
                f"of another length: as the record it was, it points at places within itself "
                f"counted from its key, and it is not a class this writer carries a layout for "
                f"to write it again; copy it under a name as long as its own"
            )
        here[leaf] = value

    def _tree(
        self, directory: Directory, key: Key, here: WritableDirectory, leaf: str, path: str
    ) -> None:
        tree = directory._read(key)
        where = f"{path!r} in {self.top.name}"
        applies = self.filter is None or self.filter(path)
        columns = self.columns if applies else None
        names: list[str] | None = None
        expressions: Mapping[str, str] | None = None
        if isinstance(columns, Mapping):
            names, expressions = [], columns
        elif columns is not None:
            names = _with_counters(tree, [columns] if isinstance(columns, str) else columns)
        merging = TreeMerge(
            here,
            leaf,
            tree,
            where,
            names=names,
            cut=self.cut if applies else None,
            expressions=expressions,
            fast=self.fast,
        )
        merging.add(tree, where, verbatim=self.verbatim)


def _destination(
    destination: Any, source_codes: int, compression: Any
) -> tuple[WritableDirectory, WritableFile | None]:
    """Where the copies go, and the file to close afterwards if it was opened here."""
    if isinstance(destination, WritableDirectory):
        return destination, None
    algorithm, level = resolve_compression(compression, source_codes)
    exists = _is_local_file(destination) and os.path.exists(_local_path(destination))
    if exists:
        opened = update(destination, compression=algorithm, level=level)
    else:
        opened = create(destination, compression=algorithm, level=level)
    return opened, opened


def copy(
    source: Any,
    destination: Any,
    keys: str | Iterable[str] | Mapping[str, str] | None = None,
    *,
    tree_filter: Callable[[str], bool] | None = None,
    cut: str | None = None,
    columns: Sequence[str] | Mapping[str, str] | None = None,
    compression: Any = None,
    fast: bool = True,
) -> list[str]:
    """Copy objects from ``source`` into ``destination``, as ``rootcp`` does.

        >>> xrdroot.copy("in.root", "out.root")                               # doctest: +SKIP
        >>> xrdroot.copy("in.root", "out.root", {"events": "skims/events"})   # doctest: +SKIP
        >>> xrdroot.copy("in.root", "skim.root", "events", cut="n > 1",
        ...              columns=["n", "pt"])                                  # doctest: +SKIP

    ``source`` is anything :func:`~xrdroot.open_root` opens, or a file already
    open. ``destination`` is a file - added to if it is there, made if not,
    compressed as ``source`` is unless ``compression`` says otherwise, as
    :func:`merge` takes it - or a directory of a file being written, which is
    used as it is and left open.

    ``keys`` picks what is copied: everything at the top of the file when
    ``None``, directories and all; a path or a shell pattern of paths, or a
    list of them, each copied to the same path; or a mapping of each path to
    the path it goes to. A name already in the destination becomes its next
    cycle.

    ``cut`` keeps only the entries of a tree that pass it, and ``columns``
    only the branches named - their counters come too - or, as a mapping of
    name to expression, columns computed as ``TTree::Draw`` would. Both apply
    to every tree copied unless ``tree_filter``, given each tree's path,
    says which. ``fast=False`` reads every tree's entries and writes them
    afresh even where its baskets could have gone across as they were.

    The paths written come back, in the order they were.
    """
    top, owned = _opened(source)
    try:
        out, opened = _destination(destination, top.compression, compression)
        if opened is None:
            return _copied(top, out, keys, tree_filter, cut, columns, fast)
        with opened:
            return _copied(top, out, keys, tree_filter, cut, columns, fast)
    finally:
        if owned:
            top.close()


def _copied(
    top: ROOTFile,
    out: WritableDirectory,
    keys: Any,
    tree_filter: Callable[[str], bool] | None,
    cut: str | None,
    columns: Sequence[str] | Mapping[str, str] | None,
    fast: bool,
) -> list[str]:
    copier = _Copier(top, out, tree_filter, cut, columns, fast)
    for path, to in _chosen(top, keys):
        copier.copy(path, to)
    return copier.written
