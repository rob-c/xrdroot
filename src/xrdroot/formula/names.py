"""Which branch a name in an expression means.

ROOT is generous about names, and so is this. A split object's member is
``P3.Px`` and also ``evt.P3.Px``, the object's name in front of it; a friend's
column is ``alias.x``; a fixed-size array whose branch is called
``ArrayI16[10]`` is written ``ArrayI16``. When a dotted name could be read
more than one way the longest branch it spells wins, so ``evt.P3.Px`` is the
member and never the object ``P3`` with something left over.
"""

from __future__ import annotations

import difflib
from collections.abc import Collection, Iterator

__all__ = ["Names"]


class Names:
    """The branches an expression may name, and how to find one from what was written."""

    __slots__ = ("names", "exact", "bare")

    def __init__(self, names: Collection[str]) -> None:
        self.names = list(names)
        self.exact = set(self.names)
        #: A name with its array size taken off, against the branch it is.
        self.bare: dict[str, str] = {}
        for name in self.names:
            if "[" in name:
                self.bare.setdefault(name.partition("[")[0], name)

    def find(self, written: str) -> str | None:
        """The branch spelled exactly ``written``, or with its array size left off."""
        if written in self.exact:
            return written
        return self.bare.get(written)

    def resolve(self, written: str) -> tuple[str, str] | None:
        """The branch a dotted name starts with, and what of the name is left after it.

        ``None`` means no branch at all. Something left over is a member of an
        object the tree holds whole, which the caller refuses by name.
        """
        parts = written.split(".")
        for stop in range(len(parts), 0, -1):
            for start in self._starts(parts, stop):
                found = self.find(".".join(parts[start:stop]))
                if found is not None:
                    return found, ".".join(parts[stop:])
        return None

    def _starts(self, parts: list[str], stop: int) -> Iterator[int]:
        """Where a name may start: the front, or past a leading object's name."""
        yield 0
        for start in range(1, stop):
            if self.find(".".join(parts[:start])) is None:
                return
            yield start

    def nearest(self, written: str, others: Collection[str] = ()) -> str:
        """A clause naming the branches that look most like ``written``, for a refusal."""
        close = difflib.get_close_matches(written, [*self.names, *others], n=3, cutoff=0.6)
        if close:
            return "the nearest are " + ", ".join(repr(name) for name in close)
        if not self.names:
            return "there are no branches to read"
        shown = ", ".join(repr(name) for name in self.names[:8])
        more = f" and {len(self.names) - 8} more" if len(self.names) > 8 else ""
        return f"there is {shown}{more}"
