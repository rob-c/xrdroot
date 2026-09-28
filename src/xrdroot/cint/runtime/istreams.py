"""``std::ifstream`` and ``std::istringstream``: reading numbers and words as ``>>`` does.

``in >> x >> y;`` is translated into one ``extract`` per target, each given
the C++ type of what it reads into: an ``int`` reads an integer, a
``double`` a number, a ``std::string`` a word, a ``char`` one character.
As in C++, whitespace separates what is read, and a read that fails - at
the end of the file, or on text that is not a number - leaves the stream
failed, which ``good()``, ``fail()``, ``eof()`` and the stream's truth say.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["istream", "ifstream", "istringstream", "stream_after", "find_if"]

#: What ``>>`` reads for each kind of target: the text it takes, from the front of the rest.
PATTERNS = {
    "integral": re.compile(r"\s*([+-]?\d+)"),
    "floating": re.compile(
        r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|nan))", re.I
    ),
    "word": re.compile(r"\s*(\S+)"),
    "char": re.compile(r"\s*(\S)"),
}

#: The C++ types, by name, that read as each kind.
KINDS = {
    **dict.fromkeys(
        [
            *"int short long unsigned bool size_t Int_t Long64_t UInt_t".split(),
            *["long long", "unsigned int", "unsigned long", "unsigned short"],
        ],
        "integral",
    ),
    **dict.fromkeys("double float Double_t Float_t long double".split(), "floating"),
    **dict.fromkeys(["char", "unsigned char", "signed char"], "char"),
}


class istream:
    """Text read with ``>>``: whitespace-separated, converted to what each target is."""

    def __init__(self, text: str = "") -> None:
        self._text = text
        self._at = 0
        self._failed = False
        self._open = True

    def extract(self, ctype: str) -> Any:
        """The next value, read as C++ reads into a ``ctype``; ``0``/``''`` once it has failed."""
        kind = KINDS.get(ctype, "word")
        found = PATTERNS[kind].match(self._text, self._at) if not self._failed else None
        if found is None:
            self._failed = True
            return "" if kind == "word" else 0
        self._at = found.end()
        word = found.group(1)
        if kind == "integral":
            return int(word)
        if kind == "floating":
            return float(word)
        return ord(word) if kind == "char" else word

    def getline(self, delimiter: Any = "\n") -> str:
        """``std::getline(stream, line[, delimiter])``: the text up to the next delimiter."""
        if self._at >= len(self._text):
            self._failed = True
            return ""
        stop = chr(delimiter) if isinstance(delimiter, int) else str(delimiter)
        end = self._text.find(stop, self._at)
        end = len(self._text) if end < 0 else end
        line = self._text[self._at : end]
        self._at = end + 1
        return line

    def good(self) -> bool:
        return not self._failed

    def fail(self) -> bool:
        return self._failed

    def eof(self) -> bool:
        return self._failed and not self._text[self._at :].strip()

    def is_open(self) -> bool:
        return self._open

    def close(self) -> None:
        self._open = False

    def __bool__(self) -> bool:
        return not self._failed


class ifstream(istream):
    """``std::ifstream(path)``: a file's text, read with ``>>``; not open if it cannot be read."""

    def __init__(self, path: Any = None, *mode: Any) -> None:
        super().__init__("")
        self._open = False
        if path is not None:
            self.open(path)

    def open(self, path: Any, *mode: Any) -> None:
        try:
            self._text = Path(str(path)).read_text(encoding="utf-8", errors="replace")
        except OSError:
            self._failed = True
            return
        self._at, self._failed, self._open = 0, False, True


class istringstream(istream):
    """``std::istringstream(text)``: a string, read with ``>>``."""

    def __init__(self, text: Any = "") -> None:
        super().__init__(str(text))

    def str(self, text: Any = None) -> str:
        if text is not None:
            self._text, self._at, self._failed = f"{text}", 0, False
            return ""
        return self._text


def stream_after(stream: Any, *stored: Any) -> Any:
    """The stream a read inside an expression read from, once what it read has been stored.

    ``while (in >> a >> b)`` tests the stream after both reads, as C++'s
    ``>>`` returns it; the reads are this call's arguments, made in order.
    """
    return stream


def find_if(first: Any, last: Any, predicate: Any) -> Any:
    """``std::find_if``: the first item from ``first`` on that ``predicate`` holds for, else None.

    ``first`` is anything Python iterates; ``last`` is where to stop, when an
    item is it. The item found is what dereferencing the iterator gives.
    """
    for item in first:
        if last is not None and item is last:
            break
        if predicate(item):
            return item
    return None
