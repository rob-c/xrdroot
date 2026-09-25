"""Python source, written a line at a time, remembering the C++ line behind each.

The source map is what makes a translated macro's errors readable: a
``ZeroDivisionError`` on Python line 40 is reported as the C++ line that
line was written for.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .errors import Where

__all__ = ["Writer", "INDENT"]

#: One level of Python indentation.
INDENT = "    "


class Writer:
    """Lines of Python and, for each, the C++ place it came from (if it came from one)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.places: list[Where | None] = []
        self.depth = 0

    def line(self, text: str, where: Where | None = None) -> None:
        self.lines.append(INDENT * self.depth + text if text else "")
        self.places.append(where)

    def blank(self) -> None:
        if self.lines and self.lines[-1]:
            self.line("")

    @contextmanager
    def indented(self) -> Iterator[None]:
        """Lines written inside the ``with`` are one level deeper; an empty body gets ``pass``."""
        self.depth += 1
        start = len(self.lines)
        try:
            yield
        finally:
            if len(self.lines) == start:
                self.line("pass")
            self.depth -= 1

    def mark(self) -> int:
        return len(self.lines)

    def insert(self, at: int, text: str, where: Where | None = None) -> None:
        """A line put back at ``at``, at the depth the lines around it have."""
        after = self.lines[at] if at < len(self.lines) else ""
        indent = after[: len(after) - len(after.lstrip())] if after.strip() else ""
        self.lines.insert(at, indent + text)
        self.places.insert(at, where)

    def text(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"

    def source_map(self) -> dict[int, tuple[str, int]]:
        """Python line (from 1) to the C++ file and line it was written for."""
        return {
            number: (where.file, where.line)
            for number, where in enumerate(self.places, start=1)
            if where is not None
        }
