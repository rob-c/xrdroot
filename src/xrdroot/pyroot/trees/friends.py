"""Friends, indices and entry lists: which entry of which tree ``GetEntry`` reads.

A friend is read beside a tree: entry ``i`` of the tree reads entry ``i``
of the friend into the friend's addresses - or, when the friend has an
index (``BuildIndex``), the entry whose index values are the tree's own
values of the same names, as ``TTreeIndex`` finds it. An entry list names
the entries a loop, a ``Draw`` or a ``Scan`` goes through.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import ListOf
from .reading import _Reading

__all__ = ["_Friends"]


class _Index:
    """``TTreeIndex``: the entry for each pair of major and minor values."""

    def __init__(self, major: str, minor: str, keys: list[tuple[int, int]]) -> None:
        self.major = major
        self.minor = minor
        self.entries: dict[tuple[int, int], int] = {}
        for entry, key in enumerate(keys):
            self.entries.setdefault(key, entry)

    def find(self, major: Any, minor: Any) -> int:
        return self.entries.get((int(major), int(minor)), -1)


class _Friends(_Reading):
    """``AddFriend``, ``BuildIndex`` and ``SetEntryList``."""

    def AddFriend(self, friend: Any, alias: Any = "", warn: bool = False) -> Any:
        """Read ``friend`` beside this tree: a tree, or a tree's name and the file it is in."""
        if isinstance(friend, str):
            friend, alias = _opened(friend, alias)
        name = str(alias or friend.GetName())
        self._friends.append((name, friend))
        if self._store is None:
            self._befriend(self._source)
        self._changed()
        return friend

    def GetListOfFriends(self) -> ListOf:
        return ListOf(friend for _, friend in self._friends)

    def GetFriend(self, name: str) -> Any:
        return next((friend for alias, friend in self._friends if alias == name), None)

    def RemoveFriend(self, friend: Any) -> None:
        self._friends = [(alias, other) for alias, other in self._friends if other is not friend]

    def _load_friends(self, entry: int) -> None:
        for _, friend in self._friends:
            index = friend._index
            if index is None:
                friend._load(entry, None)
                continue
            found = index.find(self._key(index.major, entry), self._key(index.minor, entry))
            if found >= 0:
                friend._load(found, None)

    def _key(self, name: str, entry: int) -> Any:
        """This tree's value of an index's name, for finding the friend's matching entry."""
        if name in self._columns():
            return np.asarray(self._batch.value(name, entry)).reshape(-1)[0]
        return float(name) if _is_number(name) else entry

    def BuildIndex(self, major: str, minor: str = "0") -> int:
        """Index the entries by ``major`` and ``minor``; how many were indexed comes back."""
        read = self._backing().arrays([major, minor], aliases=self._aliases)
        keys = list(zip(_integers(read[major]), _integers(read[minor])))
        self._index = _Index(major, minor, keys)
        return len(keys)

    def GetEntryNumberWithIndex(self, major: Any, minor: Any = 0) -> int:
        """The entry whose index values are these, or ``-1``."""
        if self._index is None:
            return -1
        return int(self._index.find(major, minor))

    def GetEntryWithIndex(self, major: Any, minor: Any = 0) -> int:
        """Read the entry whose index values are these; its bytes, or ``-1`` if there is none."""
        entry = self.GetEntryNumberWithIndex(major, minor)
        return -1 if entry < 0 else self.GetEntry(entry)

    def SetEntryList(self, entries: Any, option: str = "") -> None:
        """Go through only the entries in ``entries`` from now on; ``None`` for all again."""
        self._entry_list = entries

    def GetEntryList(self) -> Any:
        return self._entry_list


def _integers(values: Any) -> list[int]:
    return [int(value) for value in np.broadcast_to(np.asarray(values), (len(values),))]


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _opened(name: str, filename: Any) -> tuple[Any, str]:
    """The tree ``AddFriend("alias=name", "file.root")`` names, and the alias it is under."""
    from ... import open_root
    from . import wrap

    alias, _, tree_name = name.rpartition("=")
    source = filename if hasattr(filename, "Get") else open_root(str(filename))
    found = source.Get(tree_name) if hasattr(source, "Get") else wrap(source[tree_name])
    return found, alias or tree_name
