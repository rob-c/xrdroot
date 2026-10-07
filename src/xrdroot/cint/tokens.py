"""The words of a C++ macro, each knowing the file and line it came from.

The lexer cuts source text the way a C++ compiler's first phases do: a
backslash at the end of a line joins it to the next, comments are space,
and what is left is identifiers, numbers (the C "pp-number", decoded later),
string and character literals (raw ones too) and punctuators, longest first.
A ``#`` that starts a line begins a directive; the preprocessor reads the rest
of that line as its arguments, so every token knows whether it began a line.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .errors import Refusal, Where

__all__ = ["Token", "tokenize", "unescape"]

#: Every punctuator C++ has that a macro can use, longest first so ``>>=`` wins over ``>``.
PUNCTUATORS = sorted(
    """>>= <<= ... ->* <=> -> ++ -- << >> <= >= == != && || += -= *= /= %= &= |= ^= :: ##
    .* { } [ ] ( ) < > ; : , . ? + - * / % ^ & | ~ ! = # @ \\""".split(),
    key=len,
    reverse=True,
)

#: The words, tried in this order at each place in the text.
PATTERN = re.compile(
    r"""
    (?P<newline>\n)
    |(?P<space>[ \t\r\f\v]+)
    |(?P<comment>//[^\n]*|/\*.*?\*/)
    |(?P<raw>(?:u8|u|U|L)?R"(?P<delim>[^()\\\s]{0,16})\((?P<rawbody>.*?)\)(?P=delim)")
    |(?P<str>(?:u8|u|U|L)?"(?:\\.|[^"\\\n])*")
    |(?P<chr>(?:u8|u|U|L)?'(?:\\.|[^'\\\n])+')
    |(?P<num>\.?\d(?:[eEpP][+-]|[\w.']|)*)
    |(?P<id>[A-Za-z_$][\w$]*)
    |(?P<op>"""
    + "|".join(re.escape(p) for p in PUNCTUATORS)
    + r""")
    """,
    re.VERBOSE | re.DOTALL,
)

#: What each single-character escape after a backslash stands for.
ESCAPES = {
    "n": "\n",
    "t": "\t",
    "r": "\r",
    "a": "\a",
    "b": "\b",
    "f": "\f",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
    "e": "\x1b",
}

#: One escape sequence: a named one, octal, hexadecimal or a universal character name.
ESCAPE = re.compile(r"\\(?:([0-7]{1,3})|x([0-9a-fA-F]+)|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|(.))")


class Token:
    """One word: its kind, its text, where it is, and whether a line started with it.

    ``kind`` is ``id``, ``num``, ``str``, ``chr``, ``op`` or ``eof``. A string
    or character literal keeps its quotes in ``text``; :func:`unescape` gives
    what they stand for. ``hide`` is the set of macro names whose expansion
    produced this token, so that a macro is never expanded inside itself.
    """

    __slots__ = ("kind", "text", "where", "bol", "hide", "space", "note")

    def __init__(
        self,
        kind: str,
        text: str,
        where: Where,
        bol: bool = False,
        hide: frozenset[str] = frozenset(),
        space: bool = False,
    ) -> None:
        self.kind = kind
        self.text = text
        self.where = where
        self.bol = bol
        self.hide = hide
        #: Whether whitespace came before it - which ``#x`` stringising keeps.
        self.space = space
        #: The ``//`` comment that ends its line, when it is the line's last word: what
        #: ROOT makes a data member's title of, ranges for packed floats and all.
        self.note = ""

    def __repr__(self) -> str:
        return f"<{self.kind} {self.text!r} {self.where}>"

    def is_(self, *texts: str) -> bool:
        """Is this a punctuator or identifier spelled as one of ``texts``?"""
        return self.kind in ("op", "id") and self.text in texts

    def moved(self, where: Where, hide: frozenset[str]) -> Token:
        """This token again, placed where a macro was used and hidden from ``hide``."""
        return Token(self.kind, self.text, where, False, self.hide | hide, self.space)


def _joined(text: str) -> str:
    """The text with backslash-newlines removed, the lost newlines put back after the line.

    Joining lines this way keeps every later line's number right, which is
    what errors are reported against.
    """
    if "\\\n" not in text and "\\\r\n" not in text:
        return text
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    pending = ""
    lost = 0
    for line in lines:
        if line.endswith("\\"):
            pending += line[:-1]
            lost += 1
            continue
        out.append(pending + line + "\n" * lost)
        pending, lost = "", 0
    out.append(pending)
    return "\n".join(out)


def tokenize(text: str, file: str = "<macro>") -> list[Token]:
    """The tokens of ``text``, which came from ``file``, ending in an ``eof`` token."""
    tokens = list(_words(_joined(text), file))
    last = tokens[-1].where.line if tokens else 1
    tokens.append(Token("eof", "", Where(file, last)))
    return tokens


def _words(text: str, file: str) -> Iterator[Token]:
    at, line, bol, space = 0, 1, True, False
    last: Token | None = None
    while at < len(text):
        found = PATTERN.match(text, at)
        if found is None:
            raise Refusal(
                f"{text[at]!r} is not a character C++ source is written with", Where(file, line)
            )
        kind = found.lastgroup
        word = found.group()
        if kind == "newline":
            bol, last = True, None
        elif kind in ("space", "comment"):
            space = True
            if last is not None and word.startswith("//"):
                last.note = word[2:].strip()
        else:
            last = Token(_kind(kind), word, Where(file, line), bol, space=space)
            yield last
            bol = space = False
        line += word.count("\n")
        at = found.end()


def _kind(group: str | None) -> str:
    return "str" if group == "raw" else str(group)


def _escape(found: re.Match[str]) -> str:
    octal, hexa, small, big, other = found.groups()
    digits = hexa or small or big
    if octal:
        return chr(int(octal, 8))
    if digits:
        return chr(int(digits, 16) % 0x110000)
    return ESCAPES.get(other, other)


def unescape(literal: str) -> str:
    """What a string or character literal stands for, its prefix and quotes taken off."""
    raw = re.match(r'(?:u8|u|U|L)?R"([^(]*)\((.*)\)\1"$', literal, re.DOTALL)
    if raw is not None:
        return raw.group(2)
    body = re.sub(r"^(?:u8|u|U|L)?", "", literal)[1:-1]
    return ESCAPE.sub(_escape, body)
