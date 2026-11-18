"""``sscanf``: C's reading of a text by a format - the mirror of the ``printf`` family.

Only what a ROOT program here scans is read: ``%s`` into a character array, ``%d``, ``%i``,
``%u``, ``%o`` and ``%x`` into an integer, ``%f``, ``%e``, ``%g`` and ``%a`` into a number,
and ``%c`` into a character, each with the width C allows. A suppression (``%*d``) reads a
field and keeps nothing, and ``%%`` is the character itself. The length modifiers are dropped
after they have said how wide the field is, as the ``printf`` family's are: every number is
a Python ``int`` or ``float`` here, so nothing is truncated.

Where the value lands is C's pointer: anything with a read/write ``.value`` - a
:class:`~.cells.Cell`, an :class:`~.cells.AttrRef`, an :class:`~.cells.ItemRef`, which is
what ``&x``, ``&d.field`` and ``&a[i]`` become - or an array's first element, which is where
a pointer into one points. ``%s`` writes into a character buffer
(:func:`~.buffers.text_into`), and refuses text Python cannot change in place: a ``char
name[20]`` the translation holds as a ``str`` is such text, so the caller must hand the scan
memory that can be written (``new char[n]``, or an array a library writes into) instead.

The count is C's: the fields stored, or ``EOF`` when the text ran out before one was.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Any

from .buffers import text_into

__all__ = ["sscanf", "EOF"]

#: ``<cstdio>``'s ``EOF``: what ``sscanf`` returns when the text runs out before any field.
EOF = -1

#: C's whitespace: what a field never runs past, and what a whitespace in the format matches.
SPACES = frozenset(" \t\n\v\f\r")

#: One conversion specification: suppression, width, the length modifier dropped, conversion.
SPEC = re.compile(r"%(?P<suppress>\*)?(?P<width>\d+)?(?:hh|h|ll|l|L|q|j|z|t)?"
                  r"(?P<conv>[diouxXeEfFgGaAcs%])")  # fmt: skip

#: The conversions into a real number.
REALS = frozenset("eEfFgGaA")

#: The conversions into an integer, and the base each reads: ``%i`` reads its own from the text.
BASES = {"d": 10, "i": 0, "u": 10, "o": 8, "x": 16, "X": 16}

#: What an integer field's characters are, by base. A field is as long as these match and
#: never longer: what is left unread is where the next conversion starts. An ``%i`` field is
#: tried as ``0x``'s hexadecimal, then as octal - a leading zero and octal digits at most, so
#: that ``"09"`` is a ``0`` and a ``9`` left unread, as C reads it - then as decimal.
INTEGERS = {
    8: re.compile(r"[+-]?[0-7]+"),
    10: re.compile(r"[+-]?\d+"),
    16: re.compile(r"[+-]?(?:0[xX])?[0-9a-fA-F]+"),
    0: re.compile(r"[+-]?(?:0[xX][0-9a-fA-F]+|0[0-7]*|[1-9]\d*|0)"),
}

#: What a real field's characters are: C's decimal floating constant, and its ``inf``/``nan``.
REAL = re.compile(r"[+-]?(?:inf(?:inity)?|nan|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
                  re.IGNORECASE)  # fmt: skip

#: What one step of the scan did: it went on, the text did not match, or it had run out. A
#: step that went on says how many fields it stored - one, or none for ``%%`` and a
#: suppression.
MATCHED, MISMATCH, EXHAUSTED = 0, -1, -2

#: No value was given for a conversion to store its field in.
_NO_TARGET = object()


class _Scan:
    """The text a scan reads, and where in it the scan has got to."""

    def __init__(self, text: str) -> None:
        self.text, self.at = text, 0

    @property
    def done(self) -> bool:
        """Is there nothing left to read?"""
        return self.at >= len(self.text)

    def space(self) -> None:
        """Skip the whitespace a whitespace in the format stands for any amount of."""
        while self.at < len(self.text) and self.text[self.at] in SPACES:
            self.at += 1

    def take(self, count: int) -> str:
        """The next ``count`` characters, whitespace and all: ``%c``'s field, which has none
        of the skipping every other conversion does."""
        found = self.text[self.at : self.at + count]
        self.at += len(found)
        return found

    def word(self, width: int) -> str:
        """The next run of characters that are not whitespace: ``%s``'s field."""
        start = self.at
        end = min(len(self.text), start + width) if width else len(self.text)
        while self.at < end and self.text[self.at] not in SPACES:
            self.at += 1
        return self.text[start : self.at]

    def number(self, conv: str, width: int) -> str | None:
        """The characters of the number ``conv`` reads, or ``None`` where the text's next
        characters are not one - which C leaves unread."""
        pattern = REAL if conv in REALS else INTEGERS[BASES[conv]]
        found = pattern.match(self.text, self.at, self.at + width if width else len(self.text))
        if found is None:
            return None
        self.at = found.end()
        return found.group()


def _tokens(fmt: str) -> Iterator[tuple[str, Any]]:
    """The format as the whitespace, the characters and the conversions it is made of."""
    at = 0
    while at < len(fmt):
        char = fmt[at]
        if char in SPACES:
            at += 1
            yield "space", None
        elif char != "%":
            at += 1
            yield "literal", char
        else:
            found = SPEC.match(fmt, at)
            if found is None:
                raise ValueError(
                    f"the conversion %{fmt[at + 1: at + 2]} of the format {fmt!r} is not one "
                    "xrdroot's sscanf reads: it reads %d %i %u %o %x %f %e %g %a %c %s and %%"
                )
            at = found.end()
            yield "convert", found


def _whitespace(scan: _Scan, item: Any, given: Iterator[Any]) -> int:
    """A whitespace in the format: as much of the text's own as there is."""
    scan.space()
    return MATCHED


def _literal(scan: _Scan, item: Any, given: Iterator[Any]) -> int:
    """A character in the format: the text's next one, which must be the same."""
    if scan.done:
        return EXHAUSTED
    if scan.text[scan.at] != item:
        return MISMATCH
    scan.at += 1
    return MATCHED


def _conversion(scan: _Scan, found: Any, given: Iterator[Any]) -> int:
    """One conversion: the field read, the value made of it, and where it is stored."""
    conv = found.group("conv")
    if conv == "%":
        return _literal(scan, "%", given)
    if conv != "c":  # every conversion but %c skips the whitespace before its field
        scan.space()
    if scan.done:
        return EXHAUSTED
    field = _field(scan, conv, int(found.group("width") or 0))
    if field is None:
        return MISMATCH
    if found.group("suppress"):
        return MATCHED
    target = next(given, _NO_TARGET)
    if target is _NO_TARGET:
        raise TypeError(
            f"sscanf's format {found.string!r} has more conversions than the values it was "
            "given to store them in"
        )
    _store(target, _value(field, conv), conv)
    return MATCHED + 1


#: What one token of the format does to the scan.
_STEPS: dict[str, Callable[[_Scan, Any, Iterator[Any]], int]] = {
    "space": _whitespace,
    "literal": _literal,
    "convert": _conversion,
}


def _field(scan: _Scan, conv: str, width: int) -> str | None:
    """What one conversion reads: its characters, or ``None`` where the text's next characters
    are not the field it wants."""
    if conv == "c":
        return scan.take(width or 1)
    if conv == "s":
        return scan.word(width)
    return scan.number(conv, width)


def _value(field: str, conv: str) -> Any:
    """What a field holds: the text of ``%s`` or ``%c``, or the number of a numeric one."""
    if conv in "cs":
        return field
    if conv in REALS:
        return float(field)
    return int(field, _base_of(field) if BASES[conv] == 0 else BASES[conv])


def _base_of(field: str) -> int:
    """The base ``%i`` reads from the field itself: hexadecimal after ``0x``, octal after a
    leading zero, decimal otherwise."""
    digits = field.lstrip("+-")
    if digits[:2].lower() == "0x":
        return 16
    return 8 if digits.startswith("0") else 10


def _store(target: Any, value: Any, conv: str) -> None:
    """Write a field's value where C's pointer points."""
    if conv == "s":
        _store_text(target, value)
    elif conv == "c":
        _store_chars(target, value)
    elif hasattr(target, "value"):
        target.value = value
    else:
        target[0] = value


def _store_text(target: Any, text: str) -> None:
    """``%s``: the text and its NUL into the character buffer the target points at."""
    if isinstance(target, str):
        raise TypeError(
            "sscanf stores the text of a %s conversion in a character array, and a char array "
            "the translation holds as a Python str is text Python cannot change in place; scan "
            "into memory from new char[n], or a char array a library writes into, instead"
        )
    text_into(target, text, True)


def _store_chars(target: Any, chars: str) -> None:
    """``%c``: the characters read - one, unless a width said more - and no NUL, as C stores
    them. A single character read into a ``char`` variable is its code."""
    if hasattr(target, "value"):
        target.value = ord(chars[:1])
    else:
        target[: len(chars)] = [ord(char) for char in chars]


def _text(text: Any) -> str:
    """The text a scan reads: a character buffer as the string it holds, bytes as Latin-1."""
    if isinstance(text, (bytes, bytearray)):
        return bytes(text).decode("latin-1")
    return str(text)


def sscanf(text: Any, fmt: Any, *targets: Any) -> int:
    """``sscanf``: the fields of ``text`` that ``fmt`` names, each stored where its target is."""
    scan = _Scan(_text(text))
    given = iter(targets)
    stored = 0
    for kind, item in _tokens(str(fmt)):
        outcome = _STEPS[kind](scan, item, given)
        if outcome == EXHAUSTED:
            return stored or EOF
        if outcome == MISMATCH:
            return stored
        stored += outcome
    return stored
