"""``TUUID``: a universally unique identifier, made from the time and the host or at random.

ROOT's default ``TUUID`` is a version 1 UUID - the time and the machine -
and ``TUUID::UUIDv4()`` a random version 4 one; Python's :mod:`uuid` makes
both, laid out as RFC 4122 lays them out, which is how ROOT prints them.
UUIDs order and compare as their 128 bits do, so a ``std::set<TUUID>``
holds each one once.
"""

from __future__ import annotations

import uuid
from typing import Any

__all__ = ["TUUID"]


class TUUID:
    """``TUUID``: made now from the time, from its text, or at random by ``UUIDv4``."""

    def __init__(self, text: Any = None) -> None:
        self._uuid = uuid.uuid1() if text is None else uuid.UUID(str(text))

    @staticmethod
    def UUIDv4() -> TUUID:
        """``UUIDv4``: a random UUID, version 4."""
        made = TUUID.__new__(TUUID)
        made._uuid = uuid.uuid4()
        return made

    def AsString(self) -> str:
        """``AsString``: ``xxxxxxxx-xxxx-Vxxx-xxxx-xxxxxxxxxxxx``, ``V`` the version."""
        return str(self._uuid)

    def GetUUIDNumber(self) -> int:
        """``GetUUIDNumber``: the UUID's process-local number; xrdroot numbers none, so 0."""
        return 0

    def Compare(self, other: TUUID) -> int:
        """``Compare``: -1, 0 or 1, as the two UUIDs' bits order."""
        return (self._uuid > other._uuid) - (self._uuid < other._uuid)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TUUID) and self._uuid == other._uuid

    def __lt__(self, other: TUUID) -> bool:
        return self._uuid < other._uuid

    def __hash__(self) -> int:
        return hash(self._uuid)

    def Hash(self) -> int:
        """``Hash``: a number the same for equal UUIDs."""
        return hash(self._uuid) & 0xFFFFFFFF

    def __repr__(self) -> str:
        return f"<TUUID {self.AsString()}>"
