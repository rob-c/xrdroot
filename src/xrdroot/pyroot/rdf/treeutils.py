"""``ROOT::Internal::TreeUtils``: what a tree or chain is made of, as distributed RDF asks it.

Distributed ``RDataFrame`` does not send a ``TChain`` to its tasks: it sends
the names of the trees and of the files they are in, and of the friends,
and each task builds its own chain of them again. These are the questions
it asks of a tree - :func:`file_names`, :func:`tree_names`,
:func:`friend_info`, :func:`indexed_friend` - and :func:`chain_of`, which
builds the chain from the answers.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    "Pair",
    "RFriendInfo",
    "file_names",
    "tree_names",
    "friend_info",
    "indexed_friend",
    "chain_of",
]


class Pair:
    """``std::pair``: ``first`` and ``second``."""

    __slots__ = ("first", "second")

    def __init__(self, first: Any, second: Any) -> None:
        self.first = first
        self.second = second

    def __iter__(self) -> Any:
        return iter((self.first, self.second))

    def __repr__(self) -> str:
        return f"({self.first!r}, {self.second!r})"


class RFriendInfo:
    """``ROOT::Internal::TreeUtils::RFriendInfo``: a tree's friends, by name.

    For each friend, ``fFriendNames`` has its name and the alias it is
    known by, ``fFriendFileNames`` the files it is read from and
    ``fFriendChainSubNames`` - for a chain - the tree's name in each.
    """

    def __init__(self) -> None:
        self.fFriendNames: list[Pair] = []
        self.fFriendFileNames: list[list[str]] = []
        self.fFriendChainSubNames: list[list[str]] = []

    def add(self, name: str, alias: str, files: list[str], subnames: list[str]) -> None:
        self.fFriendNames.append(Pair(name, alias))
        self.fFriendFileNames.append(files)
        self.fFriendChainSubNames.append(subnames)

    def __bool__(self) -> bool:
        return bool(self.fFriendNames)

    def __repr__(self) -> str:
        return f"<RFriendInfo of {[pair.second for pair in self.fFriendNames]}>"


def _is_chain(tree: Any) -> bool:
    return hasattr(tree, "GetListOfFiles")


def file_names(tree: Any) -> list[str]:
    """``GetFileNamesFromTree``: each file the tree or chain is read from, in order."""
    if _is_chain(tree):
        return [str(element.GetTitle()) for element in tree.GetListOfFiles()]
    source = getattr(tree._xrd, "_source", None)
    name = str(getattr(source, "name", "<memory>"))
    if name.startswith("<"):  # a tree being filled is read back from a file in memory
        raise RuntimeError(
            f"the tree {tree.GetName()!r} is in memory and in no file, and distributed "
            f"RDataFrame reads trees from files; write it to a file and read it from there"
        )
    return [name]


def tree_names(tree: Any) -> list[str]:
    """``GetTreeFullPaths``: the tree's name in each of its files."""
    if _is_chain(tree):
        return [str(element.GetName()) for element in tree.GetListOfFiles()]
    return [str(tree.GetName())]


def friend_info(tree: Any) -> RFriendInfo:
    """``GetFriendInfo``: the friends of a tree, each by name, alias, files and tree names."""
    info = RFriendInfo()
    for alias, friend in getattr(tree, "_friends", []):
        subnames = tree_names(friend) if _is_chain(friend) else []
        info.add(str(friend.GetName()), str(alias), file_names(friend), subnames)
    return info


def indexed_friend(tree: Any) -> str | None:
    """``TreeUsesIndexedFriends``: the name of a friend read through a ``TTreeIndex``."""
    for _, friend in getattr(tree, "_friends", []):
        if getattr(friend, "_index", None) is not None:
            return str(friend.GetName())
    return None


def chain_of(trees: Iterable[str], files: Iterable[str], friends: RFriendInfo | None) -> Any:
    """A ``TChain`` of each tree in its file, with friends of the names in ``friends``."""
    from ..trees import TChain

    trees, files = list(trees), list(files)
    made = TChain(trees[0] if trees else "")
    for tree, file in zip(trees, files, strict=True):
        made.Add(f"{file}?#{tree}")
    for (name, alias), paths, names in zip(
        friends.fFriendNames if friends else [],
        friends.fFriendFileNames if friends else [],
        friends.fFriendChainSubNames if friends else [],
        strict=True,
    ):
        made.AddFriend(chain_of(names or [name] * len(paths), paths, None), alias)
    return made
