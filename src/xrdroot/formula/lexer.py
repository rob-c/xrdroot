"""The words of an expression: numbers, names, strings and C++'s operators.

A ``TTree::Draw`` expression is C++ arithmetic, and it is cut into words the
way a C++ compiler would cut it, with two differences that come from ROOT.
A name may carry dots - ``evt.P3.Px`` is one branch, not three members - and
``::`` - ``TMath::Abs`` is one function - and may end in ``$``, which is how
ROOT spells the special names, ``Entry$`` and ``Sum$`` among them.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from .errors import FormulaError

__all__ = ["Token", "tokenize"]

#: Every word the language has, tried in this order at each place in the text.
PATTERN = re.compile(
    r"""
    (?P<space>\s+)
    |(?P<number>0[xX][0-9a-fA-F]+[uUlL]*|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[fFuUlL]*)
    |(?P<string>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
    |(?P<name>[A-Za-z_][\w$]*(?:(?:\.|::)[A-Za-z_][\w$]*)*)
    |(?P<member>\.[A-Za-z_]\w*)
    |(?P<op>&&|\|\||==|!=|<=|>=|<<|>>|[-+*/%&|^~!<>?:()\[\],@])
    """,
    re.VERBOSE,
)


class Token:
    """One word of an expression, and where in the text it starts."""

    __slots__ = ("kind", "text", "at")

    def __init__(self, kind: str, text: str, at: int) -> None:
        #: ``number``, ``string``, ``name``, ``member``, ``op`` or ``end``.
        self.kind = kind
        self.text = text
        self.at = at

    def __repr__(self) -> str:
        return f"<Token {self.kind} {self.text!r} at {self.at}>"

    def is_op(self, *texts: str) -> bool:
        """Is this one of the operators or brackets named?"""
        return self.kind == "op" and self.text in texts


def tokenize(text: str) -> list[Token]:
    """The words of ``text``, ending in an ``end`` token so the parser never runs off."""
    tokens = list(_words(text))
    tokens.append(Token("end", "", len(text)))
    return tokens


def _words(text: str) -> Iterator[Token]:
    at = 0
    while at < len(text):
        found = PATTERN.match(text, at)
        if found is None:
            raise FormulaError(
                f"{text!r} has {text[at]!r} at character {at}, which is not part of any "
                f"expression TTree::Draw reads"
            )
        kind = found.lastgroup
        assert kind is not None
        if kind != "space":
            yield Token(kind, found.group(), at)
        at = found.end()
