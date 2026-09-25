"""``hadd``: many ROOT files made into one, object by object and directory by directory.

    >>> xrdroot.merge("all.root", ["run1.root", "run2.root", "run3.root"])   # doctest: +SKIP

Every name in every input is merged with the same name in every other, as
ROOT's ``TFileMerger`` merges them: directories are walked all the way down,
and what a later file holds that no earlier one did is taken up too.
Histograms and profiles add up, efficiencies add up what passed and what was
tried, graphs gather their points, trees and RNTuples hold every input's
entries one after another - a tree's baskets copied across as they are
where they can be, which is what makes this fast - and anything else, which
ROOT has no ``Merge`` for, is carried over as it is from every file it is
in, one cycle per file, with a :class:`MergeWarning` saying so once.

The inputs are read one at a time, in order, each opened, taken in and
closed before the next: a thousand files cost one open file and the
histograms being added up, not a thousand handles. What goes wrong is
refused with the name of the object and the file it is in - two histograms
binned differently, two trees with different branches, the same name
holding different classes - and a merge that raises leaves no output
behind, since :func:`~xrdroot.create` takes back what it wrote.
"""

from __future__ import annotations

import fnmatch
import os
import warnings
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, NamedTuple

from xrdclient.url import parse

from ..compression import CODES, LEVELS
from ..errors import FormatError, ROOTError
from ..file import Directory, Key, ROOTFile, open_root
from ..objects import TREE_CLASSES
from ..winfo import INFOS
from ..writer import WritableFile, create
from ..wupdate import AS_FILED, _setting, update
from .baskets import Moved
from .objects import mergeable
from .records import copy_record
from .trees import TreeMerge, normal_codes

__all__ = ["MergeWarning", "Merged", "merge", "resolve_compression", "walk"]

#: The classes a key names a directory by, old and new.
DIRECTORIES = ("TDirectory", "TDirectoryFile")
#: The class of an RNTuple's anchor, which a directory lists like any key.
NTUPLE = "ROOT::RNTuple"


class MergeWarning(UserWarning):
    """Something a merge did that ROOT's ``hadd`` does too, and that is worth knowing."""


class Merged(NamedTuple):
    """What a merge did: how each object came to be, and how its trees' baskets went.

    ``objects`` maps each path in the output to ``"merged"``, ``"copied"``,
    ``"tree"``, ``"rntuple"`` or ``"directory"``; ``baskets`` counts the tree
    baskets copied byte for byte, packed again on the way, and left where
    they were; ``entries`` counts the tree entries that went the slow way,
    read and written again; ``skipped`` names the inputs passed over.
    """

    objects: dict[str, str]
    baskets: Moved
    entries: int
    skipped: list[str]


def resolve_compression(compression: Any, first: int) -> tuple[str | None, int | None]:
    """The algorithm and level a setting asks for, however it was spelled.

    ``None`` is the first input's, as ``hadd -ff`` has it; a number is
    ROOT's ``algorithm * 100 + level`` - ``101``, ``505``, ``0`` for none -
    as ``hadd -f505`` spells it; a name is that algorithm at its usual level,
    and a pair a name and a level.
    """
    if compression is None:
        return _setting(AS_FILED, None, normal_codes(first))
    if isinstance(compression, bool) or not isinstance(compression, (int, str, tuple)):
        raise ValueError(
            f"compression is {compression!r}; it is None for the first input's, a number "
            f"such as 101 or 505 as ROOT spells a setting, an algorithm's name, or a pair "
            f"of a name and a level"
        )
    if isinstance(compression, int):
        return _setting(AS_FILED, None, normal_codes(compression))
    name, level = (compression, None) if isinstance(compression, str) else compression
    return _setting(name, level, 0)


def codes_of(algorithm: str | None, level: int | None) -> int:
    """ROOT's number for an algorithm and level, which is what a file's header holds."""
    if algorithm is None:
        return 0
    return CODES[algorithm] * 100 + (LEVELS[algorithm] if level is None else level)


class _Selection:
    """Which names are taken: ``hadd``'s ``-L`` list, to skip or to take only.

    A name matches by itself or by its path from the top of the file, and
    either may be a shell pattern. A directory named in ``only`` takes
    everything in it; one named in ``skip`` takes nothing.
    """

    def __init__(self, skip: Iterable[str] | None, only: Iterable[str] | None) -> None:
        self.skip = _patterns(skip)
        self.only = None if only is None else _patterns(only)

    def skipped(self, path: str) -> bool:
        return _matches(self.skip, path)

    def listed(self, path: str) -> bool:
        """Is ``path`` named in ``only``, which takes a directory whole?"""
        return self.only is not None and _matches(self.only, path)

    def taken(self, path: str, inside: bool) -> bool:
        """Is the object at ``path`` merged; ``inside`` says a directory above was listed."""
        return self.only is None or inside or _matches(self.only, path)


def _patterns(given: Iterable[str] | str | None) -> list[str]:
    if given is None:
        return []
    return [given] if isinstance(given, str) else [str(item) for item in given]


def _matches(patterns: list[str], path: str) -> bool:
    name = path.rpartition("/")[2]
    return any(fnmatch.fnmatchcase(path, p) or fnmatch.fnmatchcase(name, p) for p in patterns)


def walk(directory: Directory, prefix: str = "") -> Iterator[tuple[str, list[Key]]]:
    """Every name in ``directory``, with its path and its keys, oldest cycle first."""
    held: dict[str, list[Key]] = {}
    for key in directory._keys:
        held.setdefault(key.name, []).append(key)
    for name, keys in held.items():
        keys.sort(key=lambda key: key.cycle)
        yield f"{prefix}{name}", keys


def _is_local_file(target: Any) -> bool:
    return isinstance(target, (str, os.PathLike)) and parse(os.fspath(target)).is_local


def _local_path(target: Any) -> str:
    return parse(os.fspath(target)).path


class _Merger:
    """One merge in progress: the output, and what each name in it has come to so far."""

    def __init__(self, out: WritableFile, fast: bool, keep: bool, trees: bool) -> None:
        self.out = out
        self.fast = fast
        self.keep = keep
        self.trees = trees
        self.codes = normal_codes(out._codes)
        self.kinds: dict[str, str] = {}
        self.classes: dict[str, str] = {}
        self.values: dict[str, Any] = {}
        #: What is worth saying about the merge, said as warnings once it is done.
        self.notes: dict[str, str] = {}
        self.infos: dict[str, dict[tuple[str, int], bytes]] = {}
        self.moved = Moved(0, 0, 0)
        self.entries = 0
        #: The compression the input being taken in was written with.
        self.input_codes = 0

    # -- taking in one input ---------------------------------------------

    def take(self, top: ROOTFile, label: str, selection: _Selection, in_place: bool) -> None:
        """Everything in one input, down every directory, merged into what is there so far."""
        self.input_codes = normal_codes(top.compression)
        self._take_directory(top, "", label, selection, in_place, False)

    def _take_directory(
        self,
        directory: Directory,
        prefix: str,
        label: str,
        selection: _Selection,
        in_place: bool,
        inside: bool,
    ) -> None:
        for path, keys in walk(directory, prefix):
            newest = keys[-1]
            if selection.skipped(path):
                continue
            if newest.classname in DIRECTORIES:
                listed = inside or selection.listed(path)
                self._enter(directory, path, newest, label, selection, in_place, listed)
            elif selection.taken(path, inside):
                self._take_object(directory, path, keys, label, in_place)

    def _enter(
        self,
        directory: Directory,
        path: str,
        key: Key,
        label: str,
        selection: _Selection,
        in_place: bool,
        inside: bool,
    ) -> None:
        self._require_class(path, key.classname, label, "TDirectory")
        if selection.only is None or inside:
            self.out.mkdir(path)
        below = directory._subdirectory(key)
        self._take_directory(below, f"{path}/", label, selection, in_place, inside)

    def _require_class(self, path: str, classname: str, label: str, kind: str) -> None:
        """Refuse a name that holds one class in one file and another in the next."""
        classname = kind if kind == "TDirectory" else classname
        before = self.classes.setdefault(path, classname)
        if before != classname:
            raise ValueError(
                f"{path!r} is a {classname} in {label} and a {before} in the files before "
                f"it; hadd merges an object only with objects of its own class"
            )

    def _take_object(
        self, directory: Directory, path: str, keys: list[Key], label: str, in_place: bool
    ) -> None:
        newest = keys[-1]
        self._require_class(path, newest.classname, label, "")
        if newest.classname in TREE_CLASSES:
            if self.trees:
                self._take_tree(directory, path, newest, label, in_place)
            return
        if newest.classname == NTUPLE:
            self._take_ntuple(directory, path, newest, label)
            return
        value = directory._read(newest) if newest.classname in INFOS else None
        merger = mergeable(value)
        if merger is None:
            self._carry(directory, path, keys, label, in_place)
            return
        self.kinds[path] = "merged"
        before = self.values.get(path)
        self.values[path] = value if before is None else _merged(merger, before, value, path, label)

    def _take_tree(
        self, directory: Directory, path: str, key: Key, label: str, in_place: bool
    ) -> None:
        tree = directory._read(key)
        where = f"{path!r} in {label}"
        merging = self.values.get(path)
        if merging is None:
            here, leaf = self.out._place(path)
            merging = self.values[path] = TreeMerge(here, leaf, tree, where, fast=self.fast)
            self.kinds[path] = "tree"
        verbatim = self.keep or self.input_codes == self.codes
        merging.add(tree, where, verbatim=verbatim, in_place=in_place)
        self.moved = Moved(*merging.moved)

    def _take_ntuple(self, directory: Directory, path: str, key: Key, label: str) -> None:
        from .ntuples import NtupleMerge

        ntuple = directory._read(key)
        merging = self.values.get(path)
        if merging is None:
            here, leaf = self.out._place(path)
            merging = self.values[path] = NtupleMerge(here, leaf, ntuple)
            self.kinds[path] = "rntuple"
        merging.add(ntuple, f"{path!r} in {label}")

    def _carry(
        self, directory: Directory, path: str, keys: list[Key], label: str, in_place: bool
    ) -> None:
        """An object nothing here merges: every cycle of it copied over, as ROOT does."""
        self.kinds[path] = "copied"
        self.notes.setdefault(
            path,
            f"{path!r} is a {keys[-1].classname}, which is not merged, so every input's "
            f"copy of it is carried over as it is, a cycle each, as hadd does",
        )
        if in_place:
            return  # already in the file being written, where it stays
        here, leaf = self.out._place(path)
        for key in keys:
            copy_record(here, leaf, key, directory._source, self.infos)

    # -- the end ------------------------------------------------------------

    def finish(self) -> None:
        """Write every object added up; the trees and RNTuples go in as the file closes."""
        for path, kind in self.kinds.items():
            if kind == "merged":
                self.out[path] = self.values[path]
        for merging in self.values.values():
            if isinstance(merging, TreeMerge):
                self.entries += merging.slow_entries


def _merged(merger: Any, before: Any, value: Any, path: str, label: str) -> Any:
    """One object merged into what came before it, the refusal naming where it is."""
    try:
        return merger(before, value)
    except (ValueError, TypeError) as why:
        raise ValueError(
            f"{path!r} in {label} cannot be merged with the files before it: {why}"
        ) from None


def _open_output(
    output: Any, algorithm: str | None, level: int | None, append: bool, force: bool
) -> tuple[WritableFile, bool]:
    """The file being written, and whether it was already there to be added to."""
    exists = _is_local_file(output) and os.path.exists(_local_path(output))
    if append and exists:
        return update(output, compression=algorithm, level=level), True
    if exists and not force:
        raise FileExistsError(
            f"{os.fspath(output)} is already there; merge(..., force=True) - hadd -f - "
            f"writes over it, and append=True - hadd -a - merges into it"
        )
    return create(output, compression=algorithm, level=level), False


def _opened(target: Any) -> tuple[ROOTFile, bool]:
    """An input as an open file, and whether it was opened here to be closed here."""
    if isinstance(target, ROOTFile):
        return target, False
    return open_root(os.fspath(target) if isinstance(target, os.PathLike) else target), True


def _label(target: Any) -> str:
    return target.name if isinstance(target, ROOTFile) else os.fspath(target)


def _first_codes(inputs: Sequence[Any], skip_errors: bool) -> int:
    """The compression the first input that opens was written with."""
    for target in inputs:
        try:
            opened, owned = _opened(target)
        except (OSError, ROOTError):
            if skip_errors:
                continue
            raise
        codes: int = opened.compression
        if owned:
            opened.close()
        return codes
    return 0


def merge(
    output: Any,
    inputs: Sequence[Any],
    *,
    compression: Any = None,
    force: bool = False,
    fast: bool = True,
    skip_keys: Iterable[str] | None = None,
    only_keys: Iterable[str] | None = None,
    append: bool = False,
    keep_compression: bool | None = None,
    skip_errors: bool = False,
    trees: bool = True,
) -> Merged:
    """Merge ``inputs`` into one ROOT file at ``output``, as ``hadd output inputs...`` does.

        >>> xrdroot.merge("all.root", ["a.root", "b.root"])                  # doctest: +SKIP
        >>> xrdroot.merge("all.root", paths, compression=505, force=True)     # doctest: +SKIP

    ``inputs`` are anything :func:`~xrdroot.open_root` opens, or files
    already open; ``output`` anything :func:`~xrdroot.create` writes to.
    An output already there is refused unless ``force`` writes over it
    (``hadd -f``) or ``append`` merges into it, what it holds taken as the
    first input (``hadd -a``) - its trees' baskets left where they are, and
    each merged object a new cycle of its name.

    ``compression`` is the output's: the first input's unless said, or
    ROOT's number for a setting (``hadd -f505``), or an algorithm's name, or
    a pair of a name and a level. A tree's baskets are copied byte for byte
    when their input was compressed as the output is, or whatever it was
    when ``keep_compression`` says so (``hadd -fk``) - which, unless told
    otherwise, it does when ``compression`` is left alone, and does not
    when a setting is given (``keep_compression=False`` with ``compression``
    left alone is ``hadd -ff``); otherwise each is decompressed and
    compressed again, still without decoding an entry. ``fast=False``
    (``hadd -O``) reads every tree's entries and writes them afresh instead.

    ``skip_keys`` and ``only_keys`` are names - or paths from the top of the
    file, or shell patterns of either - to leave out or to take alone
    (``hadd -L`` with ``-Ltype``); ``trees=False`` leaves the trees out
    (``hadd -T``). ``skip_errors`` passes over an input that will not open
    rather than stopping (``hadd -k``).

    What was done comes back as a :class:`Merged`.
    """
    given = list(inputs)
    if not given:
        raise ValueError("a merge needs at least one input to merge")
    _require_distinct(output, given)
    algorithm, level = resolve_compression(compression, _first_codes(given, skip_errors))
    keep = compression is None if keep_compression is None else keep_compression
    out, appending = _open_output(output, algorithm, level, append, force)
    with out:
        merger = _Merger(out, fast, keep, trees)
        skipped: list[str] = []
        selection = _Selection(skip_keys, only_keys)
        if appending:
            _take_output(merger, output, selection)
        for target in given:
            _take_input(merger, target, selection, skip_errors, skipped)
        merger.finish()
    for note in merger.notes.values():
        warnings.warn(note, MergeWarning, stacklevel=2)
    return Merged(merger.kinds, merger.moved, merger.entries, skipped)


def _require_distinct(output: Any, inputs: list[Any]) -> None:
    if not _is_local_file(output):
        return
    mine = os.path.realpath(_local_path(output))
    for target in inputs:
        if _is_local_file(target) and os.path.realpath(_local_path(target)) == mine:
            raise ValueError(
                f"{os.fspath(output)} is the output and one of the inputs; append=True merges "
                f"into a file that is already there, and a file is never read while it is "
                f"written over"
            )


def _take_output(merger: _Merger, output: Any, selection: _Selection) -> None:
    """What a file being appended to already holds, taken in as the first input."""
    with open_root(os.fspath(output)) as there:
        merger.take(there, os.fspath(output), selection, in_place=True)


def _take_input(
    merger: _Merger, target: Any, selection: _Selection, skip_errors: bool, skipped: list[str]
) -> None:
    """One input, opened, taken in and closed - or passed over, if told to and it will not open."""
    label = _label(target)
    try:
        opened, owned = _opened(target)
    except (OSError, FormatError) as why:
        if not skip_errors:
            raise
        skipped.append(label)
        merger.notes[label] = f"{label} is passed over: {why}"
        return
    try:
        merger.take(opened, label, selection, in_place=False)
    finally:
        if owned:
            opened.close()

