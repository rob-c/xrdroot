"""``TString``: ROOT's string, which changes in place where Python's cannot.

``s.ReplaceAll("a", "b")`` changes ``s`` in ROOT, and a script counts on it,
so a ``TString`` here is a string that can change: it compares, hashes,
adds, formats and opens as a path (``os.fspath``) as the text it holds,
``str(s)`` is that text, and every method ROOT's has is here by ROOT's name,
the changing ones handing back the string itself as ROOT's do.
"""

from __future__ import annotations

from typing import Any

from .cformat import c_format

__all__ = ["TString", "TSubString", "kNPOS", "Printf"]

#: ``TString::kNPOS``: what ``Index`` finds when it finds nothing.
kNPOS = -1


def Printf(fmt: Any, *args: Any) -> None:
    """``Printf``: ``printf`` with a newline after, as ROOT's is."""
    print(c_format(str(fmt), *args))


def _text(value: Any) -> str:
    return str(value)


def _cased(text: str, case: int) -> str:
    return text.lower() if case == TString.kIgnoreCase else text


class TString:
    """``TString``: a string of characters that can be changed where it is."""

    #: ``ECaseCompare``.
    kExact = 0
    kIgnoreCase = 1
    #: ``EStripType``.
    kLeading = 1
    kTrailing = 2
    kBoth = 3
    kNPOS = kNPOS

    def __init__(self, text: Any = "", length: Any = None) -> None:
        held = "" if text is None else _text(text)
        self._s = held[: int(length)] if length is not None else held

    # -- as Python's string ------------------------------------------------------

    def __str__(self) -> str:
        return self._s

    def __repr__(self) -> str:
        return repr(self._s)

    def __fspath__(self) -> str:
        return self._s

    def __format__(self, spec: str) -> str:
        return format(self._s, spec)

    def __len__(self) -> int:
        return len(self._s)

    def __eq__(self, other: object) -> bool:
        return self._s == _text(other)

    def __ne__(self, other: object) -> bool:
        return self._s != _text(other)

    def __lt__(self, other: object) -> bool:
        return self._s < _text(other)

    def __le__(self, other: object) -> bool:
        return self._s <= _text(other)

    def __gt__(self, other: object) -> bool:
        return self._s > _text(other)

    def __ge__(self, other: object) -> bool:
        return self._s >= _text(other)

    def __hash__(self) -> int:
        return hash(self._s)

    def __bool__(self) -> bool:
        return True

    def __add__(self, other: Any) -> TString:
        return TString(self._s + _text(other))

    def __radd__(self, other: Any) -> TString:
        return TString(_text(other) + self._s)

    def __iadd__(self, other: Any) -> TString:
        self._s += _text(other)
        return self

    def __contains__(self, other: object) -> bool:
        return _text(other) in self._s

    def __getitem__(self, index: Any) -> str:
        return self._s[index]

    def __iter__(self) -> Any:
        return iter(self._s)

    def __call__(self, start: int, length: int | None = None) -> Any:
        """``s(i)`` is a character, ``s(start, length)`` the part of the string there."""
        if length is None:
            return self._s[start]
        return TString(self._s[start : start + length])

    def __getattr__(self, name: str) -> Any:
        """Python's string methods, for what ROOT has no name for."""
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._s, name)

    # -- what it holds -----------------------------------------------------------------

    def Data(self) -> str:
        """``Data``: the characters, as a Python string."""
        return self._s

    def Length(self) -> int:
        return len(self._s)

    def Capacity(self) -> int:
        return len(self._s)

    def Sizeof(self) -> int:
        return len(self._s) + 1

    def IsNull(self) -> bool:
        return not self._s

    def Hash(self, case: int = 0) -> int:
        return hash(_cased(self._s, case)) & 0xFFFFFFFF

    def Contains(self, pattern: Any, case: int = 0) -> bool:
        return _cased(_text(pattern), case) in _cased(self._s, case)

    def BeginsWith(self, pattern: Any, case: int = 0) -> bool:
        return _cased(self._s, case).startswith(_cased(_text(pattern), case))

    def EndsWith(self, pattern: Any, case: int = 0) -> bool:
        return _cased(self._s, case).endswith(_cased(_text(pattern), case))

    def Index(self, pattern: Any, start: int = 0, case: int = 0) -> int:
        """``Index``: where ``pattern`` first is, from ``start``, or ``kNPOS``."""
        return _cased(self._s, case).find(_cased(_text(pattern), case), int(start))

    def First(self, c: Any) -> int:
        return self._s.find(_text(c)[:1] if not isinstance(c, int) else chr(c))

    def Last(self, c: Any) -> int:
        return self._s.rfind(_text(c)[:1] if not isinstance(c, int) else chr(c))

    def CountChar(self, c: Any) -> int:
        return self._s.count(chr(c) if isinstance(c, int) else _text(c))

    def CompareTo(self, other: Any, case: int = 0) -> int:
        mine, theirs = _cased(self._s, case), _cased(_text(other), case)
        return (mine > theirs) - (mine < theirs)

    def EqualTo(self, other: Any, case: int = 0) -> bool:
        return self.CompareTo(other, case) == 0

    def IsDigit(self) -> bool:
        stripped = self._s.replace(" ", "")
        return bool(stripped) and stripped.isdigit()

    def IsAlpha(self) -> bool:
        return bool(self._s) and self._s.isalpha()

    def IsAlnum(self) -> bool:
        return bool(self._s) and self._s.isalnum()

    def IsAscii(self) -> bool:
        return all(ord(c) < 128 for c in self._s)

    def IsWhitespace(self) -> bool:
        return not self._s.strip()

    def IsFloat(self) -> bool:
        try:
            float(self._s)
        except ValueError:
            return False
        return True

    def IsDec(self) -> bool:
        return self.IsDigit()

    def Atoi(self) -> int:
        return int(float(self._s)) if self.IsFloat() else 0

    def Atoll(self) -> int:
        return self.Atoi()

    def Atof(self) -> float:
        return float(self._s) if self.IsFloat() else 0.0

    # -- changing it, in place ---------------------------------------------------------

    def Append(self, text: Any, times: int = 1) -> TString:
        piece = chr(text) if isinstance(text, int) else _text(text)
        self._s += piece * int(times)
        return self

    def Prepend(self, text: Any, times: int = 1) -> TString:
        piece = chr(text) if isinstance(text, int) else _text(text)
        self._s = piece * int(times) + self._s
        return self

    def Insert(self, position: int, text: Any) -> TString:
        self._s = self._s[:position] + _text(text) + self._s[position:]
        return self

    def Remove(self, position: int, length: int | None = None) -> TString:
        """``Remove(pos)``: everything from ``pos``; ``Remove(pos, n)``: ``n`` characters there."""
        end = len(self._s) if length is None else position + int(length)
        self._s = self._s[:position] + self._s[end:]
        return self

    def Replace(self, position: int, length: int, text: Any, count: int | None = None) -> TString:
        piece = _text(text) if count is None else _text(text)[: int(count)]
        self._s = self._s[:position] + piece + self._s[position + int(length) :]
        return self

    def ReplaceAll(self, old: Any, new: Any, *counts: Any) -> TString:
        self._s = self._s.replace(_text(old), _text(new))
        return self

    def ToLower(self) -> TString:
        self._s = self._s.lower()
        return self

    def ToUpper(self) -> TString:
        self._s = self._s.upper()
        return self

    def Chop(self) -> TString:
        self._s = self._s[:-1]
        return self

    def Clear(self) -> None:
        self._s = ""

    def Resize(self, length: int) -> None:
        self._s = self._s[:length].ljust(length)

    def Form(self, fmt: Any, *args: Any) -> None:
        """``Form``: make this string ``printf``'s text for ``fmt`` and ``args``."""
        self._s = c_format(_text(fmt), *args)

    def Strip(self, kind: int = 2, c: Any = " ") -> TSubString:
        """``Strip``: the string without ``c`` at its start, end or both - a copy."""
        char = chr(c) if isinstance(c, int) else _text(c)
        strip = {1: str.lstrip, 2: str.rstrip, 3: str.strip}[int(kind)]
        return TSubString(strip(self._s, char))

    def Tokenize(self, delimiters: Any) -> Any:
        """``Tokenize``: a ``TObjArray`` of ``TObjString``, one per run between delimiters."""
        import re

        from .collections import TObjArray, TObjString

        pieces = [part for part in re.split(f"[{re.escape(_text(delimiters))}]", self._s) if part]
        made = TObjArray()
        for piece in pieces:
            made.Add(TObjString(piece))
        return made

    def Copy(self) -> TString:
        return TString(self._s)

    # -- ROOT's static makers -----------------------------------------------------------

    @staticmethod
    def Format(fmt: Any, *args: Any) -> TString:
        """``TString::Format``: a new string, ``printf``'s text for ``fmt`` and ``args``."""
        return TString(c_format(_text(fmt), *args))

    @staticmethod
    def Itoa(value: int, base: int) -> TString:
        digits = "0123456789abcdefghijklmnopqrstuvwxyz"
        number, sign, out = abs(int(value)), "-" if value < 0 else "", ""
        while True:
            number, digit = divmod(number, int(base))
            out = digits[digit] + out
            if not number:
                return TString(sign + out)

    @staticmethod
    def UItoa(value: int, base: int) -> TString:
        return TString.Itoa(abs(int(value)), base)


class TSubString(TString):
    """``TSubString``: part of a ``TString``, which is a string of its own here."""
