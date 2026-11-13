"""Whether a piece typed at the prompt declares something, or is a statement to run.

Cling wraps what is typed in a function and then moves the declarations in
it out to the top, so that ``int i = 3;`` typed once is there for every
line after it while ``i++;`` just runs. The translator reads a whole
translation unit, so here each piece goes to the top as it is when it
declares something - a variable, a function, a class, a ``using`` - and into
a function of its own when it is a statement.

Telling them apart is C++'s own rule, read off the tokens: a declaration
is a type - qualified, templated, ``const``, ``unsigned`` - then any ``*``
and ``&``, then the name declared, then ``(``, ``=``, ``{``, ``[``, ``;``
or nothing. ``TH1F h("h", "h", 10, 0, 1)`` is one, ``h.Fill(1)`` and
``f(3)`` are not; ``a * b`` is, as C++ reads it too.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lexer import Piece

__all__ = ["Kind", "kind"]

#: Words a declaration may start with before its type.
QUALIFIERS = frozenset(
    "const static constexpr inline volatile mutable thread_local register virtual explicit "
    "typename unsigned signed long short".split()
)

#: Words that make a type ``int`` by themselves: ``unsigned i = 3``.
IMPLICIT_INT = frozenset({"unsigned", "signed", "long", "short"})

#: Words that start a declaration of something other than a variable.
DECLARING = frozenset(
    "class struct union enum namespace template typedef using static_assert extern".split()
)

#: Words no declaration starts with, or names: C++'s statements and operators.
STATEMENTS = frozenset(
    "if for while do switch try return break continue goto throw delete new case default else "
    "sizeof alignof typeid static_cast dynamic_cast const_cast reinterpret_cast this nullptr "
    "true false co_return co_await co_yield".split()
)

#: What may follow the name a declaration declares - or nothing, at the end of the piece.
FOLLOWING = frozenset({"(", "=", "{", "[", ";", ",", ":", ""})

#: What may stand between a declaration's type and its name.
DECLARATOR = frozenset({"*", "&", "&&", "const", "volatile"})


@dataclass(frozen=True)
class Kind:
    """What a piece is - ``directive``, ``declaration`` or ``statement`` - and the variable
    it declares, if it declares one."""

    what: str
    name: str | None = None


def _is_name(text: str) -> bool:
    return (text[:1].isalpha() or text[:1] == "_") and text not in STATEMENTS


class _Reader:
    """The piece's tokens, read from the front as a declaration would be."""

    def __init__(self, parts: list[str]) -> None:
        self.parts = parts
        self.at = 0

    def peek(self, ahead: int = 0) -> str:
        index = self.at + ahead
        return self.parts[index] if index < len(self.parts) else ""

    def skip(self, words: frozenset[str]) -> set[str]:
        """Pass every token in ``words``, and say which were passed."""
        passed = set()
        while self.peek() in words:
            passed.add(self.peek())
            self.at += 1
        return passed

    def qualified(self) -> bool:
        """Pass a name - ``std::vector<int>::iterator`` - and say whether there was one."""
        if not _is_name(self.peek()):
            return False
        self.at += 1
        while True:
            if self.peek() == "<" and not self.template():
                return False
            if self.peek() != "::" or not _is_name(self.peek(1)):
                return True
            self.at += 2

    def template(self) -> bool:
        """Pass ``<...>``, its brackets balanced, and say whether they were."""
        depth = 0
        for index in range(self.at, len(self.parts)):
            text = self.parts[index]
            depth += (text == "<") - (text == ">") - 2 * (text == ">>")
            if depth <= 0:
                self.at = index + 1
                return depth == 0
            if text in (";", "{", "<<"):
                return False
        return False


def kind(piece: Piece) -> Kind:
    """What ``piece`` is: a preprocessor line, a declaration, or a statement to run."""
    if piece.end == "directive":
        return Kind("directive")
    first = piece.parts[0] if piece.parts else ""
    if first in DECLARING:
        return Kind("declaration")
    if not _is_name(first):
        return Kind("statement")
    return _declared(_Reader(piece.parts))


def _declared(reader: _Reader) -> Kind:
    """A declaration and its name, if the tokens read as one; else a statement."""
    implicit = bool(reader.skip(QUALIFIERS) & IMPLICIT_INT)
    if implicit and _is_name(reader.peek()) and reader.peek(1) in FOLLOWING:
        return Kind("declaration", reader.peek())
    if not reader.qualified():
        return Kind("statement")
    reader.skip(DECLARATOR)
    name = reader.peek()
    if name == "operator":
        return Kind("declaration")
    if not reader.qualified() or reader.peek() not in FOLLOWING:
        return Kind("statement")
    return Kind("declaration", name)
