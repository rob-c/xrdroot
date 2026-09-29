"""A frame's entries from columns already read: what a CSV file or an SQLite query gave.

Both are read whole when the frame is made, as ROOT's ``RCsvDS`` reads its
file into memory, and kept as a column of values each - a NumPy array of
numbers, or of Python objects for strings and blobs - with the C++ type the
source says each is. A file named by an ``http://`` or ``https://`` URL is
fetched first, as ROOT's raw-file layer fetches one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..remote import REMOTE, fetch
from .sources import Source

__all__ = ["Columns", "REMOTE", "fetch"]

class Columns(Source):
    """Named columns of equal length, each with its C++ type."""

    #: What ROOT warns of about the data as it reads it, when anything.
    warning: str | None = None

    def __init__(self, columns: dict[str, Any], types: dict[str, str], label: str) -> None:
        self.columns = {name: np.asarray(values) for name, values in columns.items()}
        self.types = dict(types)
        self.label = label
        self.entries = len(next(iter(self.columns.values()))) if self.columns else 0

    def __len__(self) -> int:
        return self.entries

    def names(self) -> list[str]:
        return list(self.columns)

    def read(self, name: str, start: int, stop: int) -> Any:
        return self.columns[name][start:stop]

    def cxx_type(self, name: str) -> str | None:
        return self.types.get(name)

    def describe(self) -> str:
        return self.label
