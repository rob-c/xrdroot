"""``FILE[:path]``: how every subcommand names a file, and a thing inside it.

A URL has colons of its own - ``root://host:1094//f.root``, ``s3://bucket``
- so the path is split off at the last ``.root`` followed by a ``:``, as
:func:`xrdroot.session.split_location` does, and a file whose name does not
end in ``.root`` says its path with ``-k`` instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..file import Directory, Key, ROOTFile, open_root
from ..objects import TREE_CLASSES
from ..session import split_location

__all__ = [
    "CONTAINERS",
    "TABLES",
    "location",
    "opened",
    "resolve",
    "key_of",
    "walk",
]

#: The classes a key holding a directory is written as.
CONTAINERS = ("TDirectory", "TDirectoryFile")
#: The classes whose entries are read column by column: trees and RNTuples.
TABLES = (*TREE_CLASSES, "ROOT::RNTuple")


def location(text: str, key: str | None = None) -> tuple[str, str]:
    """The file ``text`` names and the path in it, ``key`` winning over a ``:path``."""
    found, path = split_location(text)
    if found is None:
        found, path = text, ""
    if key is not None:
        path = key.strip("/")
    return found, path


@contextmanager
def opened(text: str, key: str | None = None) -> Iterator[tuple[ROOTFile, str]]:
    """The file ``FILE[:path]`` names, open for as long as the ``with`` lasts, and the path."""
    found, path = location(text, key)
    with open_root(found) as file:
        yield file, path


def resolve(file: ROOTFile, path: str) -> Any:
    """What ``path`` names in ``file``: the file itself for no path at all."""
    return file[path] if path else file


def key_of(file: ROOTFile, path: str) -> tuple[Directory, Key]:
    """The directory ``path`` is in and the key it is read through."""
    parent, _, name = path.rpartition("/")
    directory = file[parent] if parent else file
    return directory, directory.key(name)


def walk(directory: Directory, prefix: str = "") -> Iterator[tuple[str, Key, Directory]]:
    """Every key under ``directory``, newest cycles, directories first-in-order and all.

    Each comes with its path from where the walk started and the directory
    it is in; a directory's own key comes before what it holds.
    """
    for name in directory.keys():
        key = directory.key(name)
        path = f"{prefix}{name}"
        yield path, key, directory
        if key.classname in CONTAINERS:
            yield from walk(directory[name], f"{path}/")
