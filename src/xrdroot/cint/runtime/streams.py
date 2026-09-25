"""``std::cout << x`` in Python, printing what C++'s iostreams print.

``cout`` here is an object whose ``<<`` writes and hands itself back, so the
translated chain reads exactly as the C++ did. What it writes follows the
stream's format state as C++'s does: a floating value has six significant
digits by default (``%g``), ``std::fixed`` and ``std::scientific`` switch
to ``%f`` and ``%e``, ``std::setprecision`` sets the digits, ``std::setw``
pads only the next value, ``std::setfill``, ``std::left``/``right``,
``std::boolalpha`` and ``std::hex`` do what they say, and a ``bool`` is
``1`` or ``0`` until ``boolalpha``.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

__all__ = [
    "ostream",
    "ostringstream",
    "cout",
    "cerr",
    "clog",
    "endl",
    "flush",
    "fixed",
    "scientific",
    "defaultfloat",
    "left",
    "right",
    "boolalpha",
    "noboolalpha",
    "hex",
    "dec",
    "oct",
    "showpos",
    "noshowpos",
    "setw",
    "setprecision",
    "setfill",
    "Manipulator",
]


class Manipulator:
    """``std::endl``, ``std::setw(8)`` and the rest: a change to a stream's format state."""

    def __init__(self, name: str, **changes: Any) -> None:
        self.name = name
        self.changes = changes

    def __repr__(self) -> str:
        return f"std::{self.name}"

    def apply(self, stream: ostream) -> None:
        for key, value in self.changes.items():
            setattr(stream, key, value)


endl = Manipulator("endl", _pending="\n")
flush = Manipulator("flush")
fixed = Manipulator("fixed", floatfield="fixed")
scientific = Manipulator("scientific", floatfield="scientific")
defaultfloat = Manipulator("defaultfloat", floatfield=None)
left = Manipulator("left", adjust="left")
right = Manipulator("right", adjust="right")
boolalpha = Manipulator("boolalpha", alpha=True)
noboolalpha = Manipulator("noboolalpha", alpha=False)
hex = Manipulator("hex", base=16)  # noqa: A001
dec = Manipulator("dec", base=10)
oct = Manipulator("oct", base=8)  # noqa: A001
showpos = Manipulator("showpos", plus=True)
noshowpos = Manipulator("noshowpos", plus=False)


def setw(width: int) -> Manipulator:
    return Manipulator("setw", width=int(width))


def setprecision(digits: int) -> Manipulator:
    return Manipulator("setprecision", digits=int(digits))


def setfill(char: Any) -> Manipulator:
    return Manipulator("setfill", fill=chr(char) if isinstance(char, int) else str(char))


class ostream:  # noqa: N801
    """A C++ output stream over a Python text file: ``<<`` writes, as C++ would format it."""

    def __init__(self, target: TextIO | None = None, name: str = "stdout") -> None:
        self._target = target
        self._name = name
        self.floatfield: str | None = None
        self.adjust = "right"
        self.alpha = False
        self.base = 10
        self.plus = False
        self.width = 0
        self.digits = 6
        self.fill = " "
        self._pending = ""

    @property
    def target(self) -> Any:
        """Where it writes: for ``cout`` and ``cerr``, whatever ``sys.stdout``/``stderr`` is now."""
        if self._target is not None:
            return self._target
        return getattr(sys, self._name)

    def __lshift__(self, value: Any) -> ostream:
        if isinstance(value, Manipulator):
            value.apply(self)
            if self._pending:
                self.target.write(self._pending)
                self._pending = ""
            return self
        text = self.format(value)
        self.write(self.pad(text))
        return self

    def write(self, text: str) -> None:
        self.target.write(text)

    def pad(self, text: str) -> str:
        width, self.width = self.width, 0
        if len(text) >= width:
            return text
        filler = self.fill * (width - len(text))
        return text + filler if self.adjust == "left" else filler + text

    def format(self, value: Any) -> str:
        """``value`` as this stream, in its present state, would write it."""
        if isinstance(value, bool):
            return ("true" if value else "false") if self.alpha else str(int(value))
        if isinstance(value, str):
            return value
        if _is_integer(value):
            return self._integer(int(value))
        if _is_floating(value):
            return self._floating(float(value))
        if value is None:
            return "0"
        return str(value)

    def _integer(self, value: int) -> str:
        if self.base == 16:
            return f"{value & 0xFFFFFFFF if value < 0 else value:x}"
        if self.base == 8:
            return f"{value & 0xFFFFFFFF if value < 0 else value:o}"
        return f"+{value}" if self.plus and value >= 0 else str(value)

    def _floating(self, value: float) -> str:
        sign = "+" if self.plus else ""
        if self.floatfield == "fixed":
            return f"%{sign}.{self.digits}f" % value
        if self.floatfield == "scientific":
            return f"%{sign}.{self.digits}e" % value
        return f"%{sign}.{max(self.digits, 1)}g" % value

    # -- the member functions macros call ----------------------------------

    def precision(self, digits: int | None = None) -> int:
        old = self.digits
        if digits is not None:
            self.digits = int(digits)
        return old

    def setf(self, *flags: Any) -> None:
        for flag in flags:
            if isinstance(flag, Manipulator):
                flag.apply(self)

    def unsetf(self, *flags: Any) -> None:
        self.floatfield = None

    def flush(self) -> ostream:
        return self

    def put(self, char: Any) -> ostream:
        self.write(chr(char) if isinstance(char, int) else str(char))
        return self

    def good(self) -> bool:
        return True


def _is_integer(value: Any) -> bool:
    return hasattr(value, "__index__") and not isinstance(value, float)


def _is_floating(value: Any) -> bool:
    return isinstance(value, float) or type(value).__name__ in ("float32", "float16")


class ostringstream(ostream):  # noqa: N801
    """``std::ostringstream``: a stream into a string, handed back by ``.str()``."""

    def __init__(self, initial: str = "") -> None:
        super().__init__(None, "")
        self._parts = [str(initial)] if initial else []

    @property
    def target(self) -> Any:
        return self

    def write(self, text: str) -> None:
        self._parts.append(text)

    def str(self, text: Any = None) -> str:  # noqa: A003
        if text is not None:
            self._parts = [f"{text}"]
            return ""
        return "".join(self._parts)


cout = ostream(None, "stdout")
cerr = ostream(None, "stderr")
clog = ostream(None, "stderr")
