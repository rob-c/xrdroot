"""``file.root:dir/pattern``: what ROOT's command-line tools mean by a source.

ROOT's ``rootls``, ``rootrm`` and ``rootcp`` share one reading of a source
(``RootObjTree``): the file, then after the first colon a path whose every
``/``-separated part is a shell pattern, matched against the names of the
keys at that depth - or against ``name;cycle`` when the part has a ``;``.
What matches is gathered as a tree of :class:`Node`\\ s, each directory's
keys sorted by name and then newest cycle first, and its ends sorted into
the directories the pattern named and everything else, the leaves. A part
that matched nothing is an error naming the pattern up to it, and the rest
is still listed. A directory the pattern names has its keys gathered too,
and with ``recursive`` every directory below it.
"""

from __future__ import annotations

import fnmatch
import os
from collections import deque
from dataclasses import dataclass, field

from ..errors import ROOTError
from ..file import Directory, Key, ROOTFile, open_root

__all__ = ["Node", "Matches", "split_source", "matching", "DIRECTORIES", "missing_file"]

#: The classes a key holding a directory is written as.
DIRECTORIES = ("TDirectory", "TDirectoryFile")

#: The schemes a source may be a URL of, as ROOT's tools accept them.
SCHEMES = ("http", "https", "root", "gs", "s3")


@dataclass
class Node:
    """One key in the tree of what a source matched - or, first of all, the file itself."""

    name: str
    classname: str
    cycle: int
    key: Key | None
    parent: int
    depth: int
    directory: Directory | None = None
    children: list[int] = field(default_factory=list)

    @property
    def is_directory(self) -> bool:
        return self.key is None or self.classname in DIRECTORIES


@dataclass
class Matches:
    """What one source matched: the tree of nodes, the directories and leaves it named."""

    filename: str
    pattern: str
    file: ROOTFile | None = None
    nodes: list[Node] = field(default_factory=list)
    dirs: list[int] = field(default_factory=list)
    leaves: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: What ROOT itself said first, whole lines: ``TFile::TFile``'s word for a missing file.
    said: list[str] = field(default_factory=list)

    def path(self, index: int) -> str:
        """The node's path in its file, from the top: ``tof/plane0``."""
        names = []
        while index > 0:
            node = self.nodes[index]
            names.append(node.name)
            index = node.parent
        return "/".join(reversed(names))

    def close(self) -> None:
        if self.file is not None:
            self.file.close()


def split_source(text: str) -> tuple[str, str]:
    """The file a source names, and the pattern after its colon.

    A plain path splits at its first colon, as ROOT's tools split it. A URL
    has colons of its own - ``root://host:1094//f.root`` - so it splits
    after the last ``.root`` a colon follows, and is the whole file without one.
    """
    scheme, sep, _ = text.partition("://")
    if not sep:
        name, _, pattern = text.partition(":")
        return name, pattern
    if scheme not in SCHEMES:
        raise ValueError("unknown file protocol")
    at = text.rfind(".root:")
    if at < 0:
        return text, ""
    return text[: at + 5], text[at + 6 :]


def missing_file(name: str) -> str | None:
    """ROOT's ``TFile::TFile`` complaint for a local file that is not there, or ``None``."""
    if "://" in name or os.path.exists(name):
        return None
    return f"Error in <TFile::TFile>: file {os.path.join(os.getcwd(), name)} does not exist"


def _sorted(keys: list[Key]) -> list[Key]:
    return sorted(keys, key=lambda key: (key.name.encode(), -key.cycle))


def _matches(part: str, key: Key) -> bool:
    name = f"{key.name};{key.cycle}" if ";" in part else key.name
    return fnmatch.fnmatchcase(name, part)


class _Builder:
    """The breadth-first walk ``GetMatchingPathsInFile`` makes of one file."""

    def __init__(self, found: Matches, parts: list[str], recursive: bool) -> None:
        self.found = found
        self.parts = parts
        self.recursive = recursive
        self.matched = [False] * len(parts)

    def build(self, file: ROOTFile) -> None:
        found = self.found
        found.nodes.append(Node(found.filename, "TFile", 0, None, -1, 0, file))
        if not self.parts:
            found.dirs.append(0)
        queue: deque[int] = deque([0])
        while queue:
            queue.extend(self._expand(queue.popleft()))
        for at, part in enumerate(self.parts):
            if not self.matched[at] and part not in ("", "*"):
                spelled = "/".join(self.parts[: at + 1])
                found.errors.append(f"'{found.filename}:{spelled}' matches no objects.")

    def _expand(self, index: int) -> list[int]:
        """Add a directory's keys - those its depth's pattern matches - and say which to open."""
        found = self.found
        node = found.nodes[index]
        assert node.directory is not None
        depth = node.depth
        part = self.parts[depth] if depth < len(self.parts) else None
        more = []
        for key in _sorted(node.directory._keys):
            if part is not None and not _matches(part, key):
                continue
            child = len(found.nodes)
            found.nodes.append(Node(key.name, key.classname, key.cycle, key, index, depth + 1))
            node.children.append(child)
            if self._place(child, part is not None):
                more.append(child)
        return more

    def _place(self, child: int, matching: bool) -> bool:
        """File a new node among the matches; whether the walk goes into it."""
        found = self.found
        node = found.nodes[child]
        if matching:
            self.matched[node.depth - 1] = True
        final = node.depth == len(self.parts)
        if matching and (final or not node.is_directory):
            (found.dirs if node.is_directory else found.leaves).append(child)
        descend = node.is_directory and (matching or self.recursive)
        if descend:
            parent = found.nodes[node.parent].directory
            assert parent is not None
            node.directory = parent._subdirectory(node.key)  # type: ignore[arg-type]
        return descend


def matching(source: str, recursive: bool = False) -> Matches:
    """What ``source`` names: its file opened, its matches found, or why not."""
    filename, pattern = split_source(source)
    found = Matches(filename, pattern)
    missing = missing_file(filename)
    if missing is not None:
        found.said.append(missing)
        found.errors.append(f"Failed to open file '{filename}'")
        return found
    try:
        found.file = open_root(filename)
    except (OSError, ValueError, ROOTError):
        found.errors.append(f"Failed to open file '{filename}'")
        return found
    parts = pattern.split("/") if pattern else []
    _Builder(found, parts, recursive).build(found.file)
    found.leaves.sort(key=lambda index: _order(found.nodes[index]))
    return found


def _order(node: Node) -> tuple[bytes, int]:
    return node.name.encode(), -node.cycle
