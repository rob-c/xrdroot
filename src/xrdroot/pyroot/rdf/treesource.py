"""What a frame of a PyROOT ``TTree`` or ``TChain`` reads, when the tree alone is not all.

A chain of ``file?#tree`` parts whose trees have names of their own is a
:class:`~xrdroot.rdf.patchwork.Patched` patchwork, and a tree with a friend
indexed by ``BuildIndex`` - or one of another length - reads it beside it as
a :class:`~xrdroot.rdf.patchwork.Befriended` friend: the entries no tree or
no friend has a value for are then the ones ``DefaultValueFor`` and
``FilterAvailable`` deal with. Any other tree is read as it is.
"""

from __future__ import annotations

from typing import Any

from ...rdf.patchwork import Befriended, Friend, Patched
from ...rdf.sources import Source, Table

__all__ = ["source_of"]


def _patched(tree: Any) -> Patched | None:
    """A chain of trees of different names, as a patchwork of them; else ``None``."""
    files = getattr(tree, "_files", None)
    if not files or len({name for _, name in files}) < 2:
        return None
    return Patched(files)


def _index(friend: Any) -> tuple[str, str] | None:
    found = getattr(friend, "_index", None)
    return None if found is None else (str(found.major), str(found.minor))


def _matched(tree: Any, friend: Any) -> bool:
    """Whether a friend is read entry for entry beside the tree, as its own data reads it."""
    return _index(friend) is None and len(friend._xrd) == len(tree._xrd)


def source_of(tree: Any) -> Source | None:
    """The source a frame of ``tree`` reads, or ``None`` when the tree's own is all of it."""
    patched = _patched(tree)
    friends = list(getattr(tree, "_friends", []))
    matched = patched is None and all(_matched(tree, friend) for _, friend in friends)
    if matched:
        return None
    main: Source = patched if patched is not None else Table(tree._xrd)
    if not friends:
        return main
    made = [Friend(alias, source_of(one) or Table(one._xrd), _index(one)) for alias, one in friends]
    return Befriended(main, made)
