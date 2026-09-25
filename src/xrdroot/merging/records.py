"""A record copied from one file into another as it was, bytes and all.

This is how anything that is not decoded on the way goes across - an object
of a class this library has no layout for, or one ``hadd`` does not merge -
and it is lossless: what the new file holds is what the old one held, down
to members nothing here knows the meaning of. The file being written is made
to describe the record's classes the way the old file did (see
:mod:`.infos`), so it reads back there as it read here.

A record points at places within itself counted from the start of its key,
so the one thing that must not change is the key's length. The name, title
and class are the same, and the key is written in the width it had; a key
that would be longer anyway - the file has grown past 2 GB since - is
refused by name rather than written with every such place wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..errors import UnsupportedFeatureError
from ..writer import WIDE
from .infos import carry

if TYPE_CHECKING:  # pragma: no cover - for the type checker, not for running
    from ..file import Key, Source
    from ..writer import WritableDirectory

__all__ = ["copy_record", "fits"]


def fits(directory: WritableDirectory, name: str, key: Key) -> bool:
    """Would a copy of ``key``'s record under ``name`` here have a key of the same length?"""
    wide = key.version > WIDE
    return directory._key_length(key.classname, name, key.title, 0, wide) == key.keylen


def copy_record(
    directory: WritableDirectory,
    name: str,
    key: Key,
    source: Source,
    cache: dict[str, dict[tuple[str, int], bytes]],
) -> None:
    """Write ``key``'s record from ``source`` into ``directory`` under ``name``, as it was.

    It becomes the next cycle of ``name`` here, as any object written
    twice does, and the file comes to describe every class ``source``
    describes that it did not already.
    """
    directory._check_leaf(name)
    if not fits(directory, name, key):
        raise UnsupportedFeatureError(
            f"{key.name!r}, a {key.classname} from {source.name}, would need a longer key "
            f"here than it had there, and a record copied as it was points at places "
            f"within itself counted from the start of its key; copy it into a file that "
            f"has not grown past 2 GB, or under a name as long as its own"
        )
    body = source.read(key.seek_key + key.keylen, key.nbytes - key.keylen)
    carry(directory._file, source, cache)
    cycle = directory._next_cycle(name)
    wide = key.version > WIDE
    directory._put(
        key.classname, name, key.title, body, cycle, listed=True, objlen=key.objlen, wide=wide
    )
