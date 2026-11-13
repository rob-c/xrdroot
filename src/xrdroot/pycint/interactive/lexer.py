"""What was typed at the prompt, cut into the pieces cling handles one by one.

Cling reads a line, and if a brace, a bracket or a string is still open it
asks for more - ``root (cont'ed, cancel with .@) [1]`` - until the input is
whole. :func:`pending` says whether it is. A whole input is then cut at the
top level into :class:`Piece` s: each statement or declaration ending in
``;``, each block, function, namespace or class ending in its ``}``, each
preprocessor line, and, last, whatever is left without a ``;`` - which is
the expression whose value cling prints.

The tokens are C++'s only as far as this needs: strings and characters, so
that a ``;`` or ``{`` in one is not one; comments, which are dropped; words;
and punctuation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Token", "Piece", "tokens", "pending", "pieces", "ENDS"]

#: C++ as far as cutting it up needs; the first alternative that matches wins.
TOKEN = re.compile(
    r"""
    (?P<directive>(?<![^\n])[ \t]*\#(?:\\\n|[^\n])*)
    |(?P<space>[^\S\n]*\n|[^\S\n]+)
    |(?P<comment>//[^\n]*|/\*.*?\*/)
    |(?P<open>/\*.*|"(?:\\.|[^"\\\n])*(?:\\)?$|'(?:\\.|[^'\\\n])*$)
    |(?P<string>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
    |(?P<word>[A-Za-z_]\w*)
    |(?P<number>\.?\d(?:[eEpP][+-]|[\w.'])*)
    |(?P<punct>::|->\*?|<<=?|>>=?|\+\+|--|&&|\|\||[-+*/%&|^!=<>]=|\.\.\.|.)
    """,
    re.S | re.X,
)

#: What opens a bracket, and what closes one.
OPENING, CLOSING = frozenset("([{"), frozenset(")]}")

#: After a piece ends with ``}``, the words that carry it on, by the word it started with:
#: ``if (...) {} else {}``, ``do {} while (...)``, ``try {} catch (...) {}``.
CARRIED = {"if": "else", "do": "while", "try": "catch"}

#: What comes before a ``{`` that opens a body - a function's, a loop's, a ``do``'s -
#: rather than an initializer; ``""`` is a block typed by itself.
BLOCKS = frozenset({")", "]", "do", "try", "else", "const", "override", "noexcept", "final",
                    "mutable", ""})  # fmt: skip

#: What a class, struct, union or enum definition starts with: it ends at its ``;``.
RECORDS = frozenset({"class", "struct", "union", "enum", "typedef"})

#: How a piece ended: with its ``;``, with its ``}``, as a preprocessor line, or not at all.
ENDS = ("semicolon", "brace", "directive", "open")


@dataclass(frozen=True)
class Token:
    """One token: its kind (a group of :data:`TOKEN`), its text, and where it starts."""

    kind: str
    text: str
    start: int


@dataclass
class Piece:
    """A statement or declaration cut from the input, and where in the input it starts."""

    text: str
    line: int
    column: int
    end: str
    #: The piece's tokens, comments and whitespace left out: what tells a declaration.
    parts: list[str] = field(default_factory=list)


def tokens(text: str) -> list[Token]:
    """Every token of ``text``, whitespace and comments included."""
    return [Token(found.lastgroup or "", found.group(), found.start())
            for found in TOKEN.finditer(text)]  # fmt: skip


def _significant(token: Token) -> bool:
    return token.kind not in ("space", "comment")


def pending(text: str) -> bool:
    """Whether the input waits for more: a bracket, a string or a comment still open.

    A preprocessor line ending in a backslash goes on too. More closing than
    opening brackets is no reason to wait: the translator says what is wrong.
    """
    depth = 0
    for token in tokens(text):
        if token.kind == "open" or (token.kind == "directive" and token.text.endswith("\\")):
            return True
        if token.kind == "punct":
            depth += (token.text in OPENING) - (token.text in CLOSING)
    return depth > 0


class _Cutter:
    """Walks the tokens once, cutting a piece wherever one ends at the top level."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.found = [token for token in tokens(text) if _significant(token)]
        self.pieces: list[Piece] = []
        self.start: int | None = None
        #: The word the piece being cut starts with, or ``""``.
        self.first = ""
        self.depth = 0
        #: Whether the ``{`` opening at the top opens a body, which its ``}`` ends, rather
        #: than an initializer, ``v{1, 2}``, which the ``;`` after it ends.
        self.block = False
        #: Where the last token taken ends: an unfinished piece ends there, not in a comment.
        self.last = 0

    def cut(self) -> list[Piece]:
        for index, token in enumerate(self.found):
            self.take(index, token)
        if self.start is not None:
            self.close(self.last, "open")
        return self.pieces

    def take(self, index: int, token: Token) -> None:
        if token.kind == "directive" and self.start is None:
            self.start = token.start + len(token.text) - len(token.text.lstrip())
            self.close(token.start + len(token.text), "directive")
            return
        self.last = token.start + len(token.text)
        if self.start is None:
            self.start = token.start
            self.first = token.text if token.kind == "word" else ""
        if token.kind == "punct":
            self.bracket(index, token)
        if self.depth == 0 and self.ends(index, token):
            self.close(self.last, "brace" if token.text == "}" else "semicolon")

    def bracket(self, index: int, token: Token) -> None:
        """Count a bracket; at a ``{`` opening at the top, tell a body from an initializer."""
        if token.text == "{" and self.depth == 0:
            before = self.found[index - 1].text if self.start != token.start else ""
            self.block = before in BLOCKS or self.first in ("namespace", "extern")
        self.depth += (token.text in OPENING) - (token.text in CLOSING)

    def ends(self, index: int, token: Token) -> bool:
        """Whether ``token`` ends the piece it is in: a ``;``, or a ``}`` nothing carries on."""
        if token.text not in (";", "}") or token.kind != "punct":
            return False
        following = self.found[index + 1].text if index + 1 < len(self.found) else ""
        if following == CARRIED.get(self.first):
            return False
        return token.text == ";" or (following != ";" and self.block and not self.record())

    def record(self) -> bool:
        """Whether the piece is a class or enum definition, which only its ``;`` ends."""
        assert self.start is not None
        head = self.text[self.start :].split("{", 1)[0]
        return "(" not in head and bool(RECORDS & set(re.findall(r"\w+", head)))

    def close(self, stop: int, end: str) -> None:
        assert self.start is not None
        before = self.text[: self.start]
        line = before.count("\n") + 1
        column = self.start - (before.rfind("\n") + 1) + 1
        body = self.text[self.start : stop]
        parts = [token.text for token in tokens(body) if _significant(token)]
        self.pieces.append(Piece(body, line, column, end, parts))
        self.start = None


def pieces(text: str) -> list[Piece]:
    """``text``, a whole input, as the pieces cling runs in turn; empty statements dropped."""
    return [piece for piece in _Cutter(text).cut() if piece.text.strip() != ";"]
