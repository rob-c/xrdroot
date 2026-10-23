"""C strings and ``std::string`` as Python ``str``, with C's functions over them.

A ``char name[20]``, a ``const char*`` and a ``std::string`` all become a
Python ``str`` here; ``sprintf(name, ...)`` and ``strcpy(name, ...)`` become
assignments to ``name``. What is left is the functions that read strings:
``strlen``, ``strcmp``, ``atoi``, ``std::to_string`` (which formats a double
with ``%f``, as C++ says), and the ``std::string`` members the translator
rewrites to Python when the type says a value is a string.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "strlen",
    "strcmp",
    "strncmp",
    "strcasecmp",
    "strstr",
    "strchr",
    "strrchr",
    "atoi",
    "atol",
    "atof",
    "stoi",
    "stod",
    "to_string",
    "char_at",
    "npos",
    "find",
    "rfind",
    "substr",
    "cstr",
    "find_first_of",
    "find_last_of",
    "find_first_not_of",
    "find_last_not_of",
    "resize",
    "tolower",
    "toupper",
    "transformed",
]

#: ``std::string::npos``: what ``find`` gives back when there is nothing to find.
npos = 2**64 - 1


def cstr(value: Any) -> str:
    """Anything a macro treats as text - a TString, a char, None - as a Python ``str``."""
    if value is None:
        return ""
    if isinstance(value, int):
        return chr(value)
    return str(value)


def strlen(text: Any) -> int:
    return len(cstr(text))


def _compare(a: str, b: str) -> int:
    return (a > b) - (a < b)


def strcmp(a: Any, b: Any) -> int:
    """C's ``strcmp``: negative, zero or positive, as ``a`` sorts before, with or after ``b``."""
    return _compare(cstr(a), cstr(b))


def strncmp(a: Any, b: Any, count: int) -> int:
    return _compare(cstr(a)[: int(count)], cstr(b)[: int(count)])


def strcasecmp(a: Any, b: Any) -> int:
    return _compare(cstr(a).lower(), cstr(b).lower())


def strstr(haystack: Any, needle: Any) -> str | None:
    """C's ``strstr``: the rest of ``haystack`` from where ``needle`` is, or ``None``."""
    text = cstr(haystack)
    at = text.find(cstr(needle))
    return None if at < 0 else text[at:]


def strchr(haystack: Any, char: Any) -> str | None:
    return strstr(haystack, cstr(char))


def strrchr(haystack: Any, char: Any) -> str | None:
    """C's ``strrchr``: the rest of ``haystack`` from the last ``char`` in it, or ``None``."""
    text = cstr(haystack)
    at = text.rfind(cstr(char))
    return None if at < 0 else text[at:]


def _leading(text: Any, pattern: str) -> str:
    found = re.match(pattern, cstr(text).lstrip())
    return found.group() if found else ""


def atoi(text: Any) -> int:
    """C's ``atoi``: the integer the text starts with, ``0`` if none."""
    digits = _leading(text, r"[+-]?\d+")
    return int(digits) if digits else 0


atol = atoi


def atof(text: Any) -> float:
    """C's ``atof``: the number the text starts with, ``0.0`` if none."""
    number = _leading(text, r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?(?:inf|nan)")
    return float(number) if number else 0.0


def stoi(text: Any, *rest: Any) -> int:
    """``std::stoi``, which throws where ``atoi`` gives ``0``."""
    digits = _leading(text, r"[+-]?\d+")
    if not digits:
        raise ValueError(f"std::stoi: {cstr(text)!r} does not start with a number")
    return int(digits)


def stod(text: Any, *rest: Any) -> float:
    if not _leading(text, r"[+-]?(?:\d|\.\d|inf|nan)"):
        raise ValueError(f"std::stod: {cstr(text)!r} does not start with a number")
    return atof(text)


def to_string(value: Any) -> str:
    """``std::to_string``: an integer as digits, a floating value with ``%f``."""
    if isinstance(value, bool) or hasattr(value, "__index__"):
        return str(int(value))
    return f"{float(value):f}"


def char_at(text: Any, index: Any) -> int:
    """``s[i]`` of a string, as the ``char`` - a number - C++ gives."""
    string = cstr(text)
    at = int(index)
    return ord(string[at]) if at < len(string) else 0


def find(text: Any, what: Any, start: int = 0) -> int:
    """``std::string::find``: where ``what`` first is, or :data:`npos`."""
    at = cstr(text).find(cstr(what), int(start))
    return npos if at < 0 else at


def rfind(text: Any, what: Any, start: int = npos) -> int:
    string = cstr(text)
    at = string.rfind(cstr(what), 0, min(int(start), len(string)) + len(cstr(what)))
    return npos if at < 0 else at


def substr(text: Any, start: int = 0, count: int = npos) -> str:
    """``std::string::substr(start, count)``."""
    string = cstr(text)
    begin = int(start)
    if begin > len(string):
        raise IndexError(f"substr starts at {begin}, past the end of a string {len(string)} long")
    return string[begin : begin + min(int(count), len(string))]


def _first(text: Any, chars: Any, start: Any, inside: bool) -> int:
    string, wanted = cstr(text), cstr(chars)
    for at in range(int(start), len(string)):
        if (string[at] in wanted) == inside:
            return at
    return npos


def _last(text: Any, chars: Any, start: Any, inside: bool) -> int:
    string, wanted = cstr(text), cstr(chars)
    for at in range(min(int(start), len(string) - 1), -1, -1):
        if (string[at] in wanted) == inside:
            return at
    return npos


def find_first_of(text: Any, chars: Any, start: Any = 0) -> int:
    """``s.find_first_of(chars)``: where the first of any of ``chars`` is, or :data:`npos`."""
    return _first(text, chars, start, True)


def find_first_not_of(text: Any, chars: Any, start: Any = 0) -> int:
    return _first(text, chars, start, False)


def find_last_of(text: Any, chars: Any, start: Any = npos) -> int:
    """``s.find_last_of(chars)``: where the last of any of ``chars`` is, or :data:`npos`."""
    return _last(text, chars, start, True)


def find_last_not_of(text: Any, chars: Any, start: Any = npos) -> int:
    return _last(text, chars, start, False)


def resize(text: Any, count: Any, fill: Any = 0) -> str:
    """``s.resize(n[, c])``: cut to ``n`` characters, or padded to them with ``c`` (a NUL)."""
    string, size = cstr(text), int(count)
    return string[:size] + cstr(fill) * max(size - len(string), 0)


def tolower(char: Any) -> int:
    """C's ``tolower`` of a character, which is a number."""
    return ord(chr(int(char)).lower()) if 0 <= int(char) < 128 else int(char)


def toupper(char: Any) -> int:
    return ord(chr(int(char)).upper()) if 0 <= int(char) < 128 else int(char)


def _same(item: Any) -> Any:
    return item


def transformed(source: Any, start: Any, stop: Any, target: Any, at: Any, op: Any) -> Any:
    """``std::transform(first, last, out, op)``: ``target`` with ``op`` of each source item.

    A string is rebuilt, one ``char`` (a number) at a time, and handed back
    for the translation to store; a container is written in place. With no
    ``op`` it is ``std::copy``.
    """
    if op is None:
        op = _same
    if isinstance(source, str) or isinstance(target, str):
        codes = [ord(c) for c in cstr(source)][int(start) : stop]
        made = "".join(chr(int(op(code)) % 256) for code in codes)
        text, begin = cstr(target), int(at)
        return text[:begin] + made + text[begin + len(made) :]
    items = list(source[int(start) : stop])
    for offset, item in enumerate(items):
        target[int(at) + offset] = op(item)
    return target
