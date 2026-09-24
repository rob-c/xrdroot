"""Finding a tree's friends again, from what ROOT wrote down about them.

ROOT records a friend by the name of its tree and the name of its file as the
job that made the friendship knew them - an absolute path on a machine long
gone, as often as not, or a name relative to wherever that job ran. What is
nearly always true is that the friend was shipped beside the tree, so that is
where it is looked for: the name as written if it is there, else the same
name in this file's own directory, else the file's last part there.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from xrdclient.url import parse

if TYPE_CHECKING:
    from .file import Source
    from .objects import FriendRecord

__all__ = ["friend_target", "open_friend"]


def _local_target(written: str, own: str) -> str | None:
    """Where a friend in a local file is, or ``None`` when it is this file."""
    here = os.path.dirname(own)
    candidates = [written] if os.path.isabs(written) else []
    candidates += [os.path.join(here, written), os.path.join(here, os.path.basename(written))]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return None if os.path.samefile(candidate, own) else candidate
    raise FileNotFoundError(
        f"a friend of a tree in {own} is in {written!r}, which is neither there nor "
        f"beside {own}; put it beside, or add the friend by hand with add_friend"
    )


def _remote_target(written: str, own: str) -> str | None:
    """Where a friend of a tree in a remote file is: beside it, on the same server."""
    if not parse(written).is_local:
        return written  # a URL of its own, which says where it is outright
    name = os.path.basename(written)
    if name == own.rstrip("/").rpartition("/")[2]:
        return None
    return f"{own.rpartition('/')[0]}/{name}"


def friend_target(written: str, own: str) -> str | None:
    """Where the file a friend is in is, from the name ROOT wrote and this file's.

    ``None`` means this file: ROOT writes no name at all for a friend in the
    same file as the tree, and a name that turns out to be this file is the
    same thing said the long way.
    """
    if not written:
        return None
    url = parse(own)
    if url.is_local:
        return _local_target(written, url.path)
    return _remote_target(written, own)


def open_friend(record: FriendRecord, source: Source) -> Any:
    """The tree a ``TFriendElement`` names, opened from wherever it turns out to be.

    A friend in another file opens that file, which is then closed with the
    file the tree befriending it is in. One in the same file is read out of
    the same handle, which nothing has to close twice.
    """
    from .file import ROOTFile, open_root

    target = friend_target(record.file_name, source.name)
    if target is None:
        return ROOTFile(source)[record.tree_name]
    config = source._reopen.config if source._reopen is not None else None
    opened = open_root(target, config=config)
    source.companions.append(opened)
    return opened[record.tree_name]
