"""A FITS header: 80-character cards in 2880-byte blocks, read as CFITSIO reads them.

Every card is a keyword, a value and a comment. CFITSIO hands a program
the value as it is written - a string still in its quotes, a number as its
digits - and the comment after its slash, which is what ``TFITSHDU`` keeps
and prints. A card with no ``= `` after its keyword - ``COMMENT``,
``HISTORY``, a blank keyword - has no value, and all of it past column 8 is
its comment.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import FormatError

__all__ = ["BLOCK", "CARD", "Record", "header", "number", "record", "text"]

#: The size of every header block and every data block of a FITS file.
BLOCK = 2880

#: The width of one header card.
CARD = 80

#: The keywords whose cards never carry a value, whatever follows them.
NO_VALUE = frozenset({"", "COMMENT", "HISTORY", "END", "CONTINUE"})


@dataclass(frozen=True)
class Record:
    """One header card: its keyword, its value as written, and its comment."""

    keyword: str
    value: str
    comment: str


def _quoted(rest: str) -> int:
    """Where a quoted value ends: past its closing quote, a doubled quote being one inside it."""
    at = 1
    while at < len(rest):
        if rest[at] == "'" and rest[at + 1:at + 2] != "'":
            return at + 1
        at += 2 if rest[at] == "'" else 1
    return len(rest)


def _value_end(rest: str) -> int:
    """Where a value written from the start of ``rest`` ends."""
    if rest.startswith("'"):
        return _quoted(rest)
    if rest.startswith("("):
        return rest.find(")") + 1 or len(rest)
    ends = [at for at in (rest.find(" "), rest.find("/")) if at >= 0]
    return min(ends, default=len(rest))


def record(card: str) -> Record:
    """A card as ``fits_read_keyn`` gives it: keyword, value and comment."""
    keyword = card[:8].rstrip()
    if keyword in NO_VALUE or card[8:10] != "= ":
        return Record(keyword, "", card[8:].rstrip())
    rest = card[10:].lstrip()
    end = _value_end(rest)
    after = rest[end:].lstrip()
    comment = after[1:] if after.startswith("/") else after
    comment = comment[1:] if comment.startswith(" ") else comment
    return Record(keyword, rest[:end].rstrip(), comment.rstrip())


def _before_blanks(records: list[Record]) -> list[Record]:
    """The records without the blank cards just before ``END``, which CFITSIO counts as room
    left for more rather than as records."""
    while records and records[-1] == Record("", "", ""):
        records.pop()
    return records


def header(data: bytes, at: int) -> tuple[list[Record], int]:
    """The records of the header starting at ``at``, and where its data begins."""
    records: list[Record] = []
    while at + BLOCK <= len(data):
        block = data[at:at + BLOCK].decode("latin-1")
        at += BLOCK
        for start in range(0, BLOCK, CARD):
            card = block[start:start + CARD]
            if card.startswith("END") and not card[3:].strip():
                return _before_blanks(records), at
            records.append(record(card))
    raise FormatError("This FITS header has no END card before the file ends.")


def text(value: str) -> str:
    """A string value without its quotes, its doubled quotes single and trailing blanks gone."""
    inner = value[1:-1] if value.startswith("'") and value.endswith("'") else value
    return inner.replace("''", "'").rstrip()


def number(value: str) -> float:
    """A number as written, FORTRAN's ``D`` exponent included."""
    return float(value.replace("D", "E").replace("d", "e"))
