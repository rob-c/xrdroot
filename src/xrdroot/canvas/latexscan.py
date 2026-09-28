"""How ``TLatex`` reads a formula: its syntax check, and the operator each piece is.

``TLatex::CheckLatexSyntax`` first turns ``#left(``...``#right)`` into the
operators they stand for and escapes, with an ``@``, every brace and
bracket that belongs to no operator, so that ``f(x)[GeV]`` is text; a
formula whose operators are not closed is an error, and ROOT draws none of
it. ``TLatex::Analyse`` then reads a piece once, left to right, for the
first operator at its top level - a script (``^{`` or ``_{``, only with
the brace), a ``#`` command, a ``}`` after which more follows - and splits
the piece there. :func:`scan` is that one reading, as ROOT does it, down
to which of two operators wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ABOVE", "GREEK", "SPECIAL", "Found", "LatexError", "check", "scan"]

#: ``TLatex``'s Greek letters, in the order of their letters in the Symbol font.
GREEK = (
    "alpha beta chi delta varepsilon phi gamma eta iota varphi kappa lambda mu nu omicron pi "
    "theta rho sigma tau upsilon varomega omega xi psi zeta Alpha Beta Chi Delta Epsilon Phi "
    "Gamma Eta Iota vartheta Kappa Lambda Mu Nu Omicron Pi Theta Rho Sigma Tau Upsilon "
    "varsigma Omega Xi Psi Zeta varUpsilon epsilon"
).split()
#: ``TLatex``'s symbols, in the order of their codes in the Symbol font from ``\\243``.
SPECIAL = (
    "leq / infty voidb club diamond heart spade leftrightarrow leftarrow uparrow rightarrow "
    "downarrow circ pm doublequote geq times propto partial bullet divide neq equiv approx "
    "3dots cbar topbar downleftarrow aleph Jgothic Rgothic voidn otimes oplus oslash cap cup "
    "supset supseteq notsubset subset subseteq in notin angle nabla oright ocopyright "
    "trademark prod surd upoint corner wedge vee Leftrightarrow Leftarrow Uparrow Rightarrow "
    "Downarrow diamond LT void1 copyright void3 sum arctop lbar arcbottom topbar void8 "
    "bottombar arcbar ltbar AA aa void06 GT int forall exists"
).split()
#: The accents, drawn over their argument.
ABOVE = "bar vec dot hat ddot acute grave check tilde slash".split()
#: The commands ROOT tries first, in its order: each name, and the brace or
#: bracket that must come right after it (none for a symbol drawn by hand).
TRIED = (
    ("splitline", "{"), ("backslash", ""), ("parallel", ""), ("lower", "[{"), ("scale", "[{"),
    ("color", "[{"), ("frac", "{"), ("sqrt", "{["), ("font", "{["), ("kern", "[{"),
    ("minus", ""), ("mbox", "[{"), ("odot", ""), ("hbar", ""), ("perp", ""), ("plus", ""),
    ("url", "[{"), ("[]", "{"), ("{}", "{"), ("||", "{"), ("()", "{"), ("Box", ""),
    ("bf", "[{"), ("it", "[{"), ("mp", ""),
)  # fmt: skip


class LatexError(ValueError):
    """A formula ``TLatex`` refuses, and draws nothing of."""


@dataclass
class Found:
    """Where ``Analyse`` found each operator in a piece: -1 where none."""

    power: int = -1
    under: int = -1
    close_curly: int = -2
    curly_curly: int = -1
    square_curly: int = -1
    #: The first ``#`` command: its name, and where its ``#`` is.
    command: tuple[str, int] = ("", -1)
    #: 1 when the scripts belong over and under an ``#int``, 2 for a ``#sum``.
    above_place: int = 0
    seen: set[str] = field(default_factory=set)


def _escaped(text: str, at: int) -> bool:
    return at > 0 and text[at - 1] == "@"


def _closing(found: Found, text: str, at: int) -> None:
    """A ``}`` closing the top level: the end of a piece, unless an argument or script follows."""
    rest = text[at + 1 :]
    if rest[:1] == "{" and found.curly_curly == -1:
        found.curly_curly = at
    if found.close_curly != -2 or not rest:
        return
    if len(rest) >= 2 and rest[1] == "{" and rest[0] in "^_":
        return
    if rest[0] != "{":
        found.close_curly = at


def _script(found: Found, text: str, at: int, level: int) -> None:
    """``^{``, ``_{`` or ``]{`` at ``at``: the first at the top level of each is kept."""
    two = text[at : at + 2]
    if two in ("^{", "_{"):
        if level == 0:
            if two == "^{" and found.power == -1:
                found.power = at
            if two == "_{" and found.under == -1:
                found.under = at
        _limits(found, text, at)
    elif two == "]{" and level == 0 and found.square_curly == -1:
        found.square_curly = at


def _limits(found: Found, text: str, at: int) -> None:
    """Scripts right after an ``#int`` or ``#sum`` go over and under it."""
    if at <= 3:
        return
    before = text[at - 4 : at]
    for place, name in ((1, "#int"), (2, "#sum")):
        if before == name:
            found.above_place = place
            if at > 4 and found.close_curly == -2:
                found.close_curly = at - 5


def _named(text: str, at: int) -> str:
    """The command at ``at``, the ``#`` read: its name, or empty for none ROOT knows."""
    after = text[at + 1 :]
    for name, opening in TRIED:
        if after.startswith(name) and (not opening or after[len(name) : len(name) + 1] in opening):
            if not opening or len(after) > len(name):
                return name
    for table in (GREEK, ABOVE):
        hit = next((name for name in table if after.startswith(name)), "")
        if hit:
            return hit
    return max((name for name in SPECIAL if after.startswith(name)), key=len, default="")


def scan(text: str) -> Found:
    """``Analyse``'s one reading of a piece: where each of its operators is."""
    found = Found()
    level = square = 0
    for at, char in enumerate(text):
        level, square = _count(found, text, at, char, level, square)
        if at + 1 < len(text):
            _script(found, text, at, level + square)
        if char == "#" and found.command[1] < 0 and level == 0 and square == 0:
            name = _named(text, at)
            if name:
                found.command = (name, at)
                if at > 0 and found.close_curly == -2:
                    found.close_curly = at - 1
    return found


def _count(found: Found, text: str, at: int, char: str, level: int, square: int) -> tuple[int, int]:
    """The depth of braces and of brackets after ``char``, each counted outside the other."""
    if char == "{" and square == 0 and not _escaped(text, at):
        level += 1
    elif char == "}" and square == 0:
        level -= 0 if _escaped(text, at) else 1
        if level == 0:
            _closing(found, text, at)
    elif char == "[" and level == 0 and not _escaped(text, at):
        square += 1
    elif char == "]" and level == 0:
        square -= 0 if _escaped(text, at) else 1
        if square < 0:
            raise LatexError('Missing "["')
    return level, square


#: ``#left`` and ``#right`` of each delimiter, and the operator a ``#left`` becomes.
LEFTS = {"#left[": "#[]{", "#left{": "#{}{", "#left|": "#||{", "#left(": "#(){"}
RIGHTS = ("#right]", "#right}", "#right|", "#right)")
#: What opens an argument in braces, closed by the next ``}``.
BRACED = (
    "{}^{", "{}_{", "^{", "_{", "#scale{", "#color{", "#url{", "#font{", "#sqrt{", "#[]{",
    "#{}{", "#||{", "#bar{", "#vec{", "#dot{", "#hat{", "#ddot{", "#acute{", "#grave{",
    "#check{", "#tilde{", "#slash{", "#bf{", "#it{", "#mbox{", "#(){",
)  # fmt: skip
#: What opens a setting in brackets, closed by ``]{`` and then ``}``.
BRACKETED = ("#scale[", "#color[", "#url[", "#font[", "#sqrt[", "#kern[", "#lower[")
#: What takes two arguments, ``}{`` between them.
PAIRED = ("#frac{", "#splitline{")


def _delimiters(text: str) -> str:
    """``#left(``...``#right)`` as the operator they are, the two counted to match."""
    lefts = sum(text.count(left) for left in LEFTS)
    rights = sum(text.count(right) for right in RIGHTS)
    if lefts != rights:
        raise LatexError('Operators "#left" and "#right" don\'t match !')
    for left, operator in LEFTS.items():
        text = text.replace(left, operator)
    for right in RIGHTS:
        text = text.replace(right, "}")
    return text


class _Checker:
    """One pass of ``CheckLatexSyntax``: every brace and bracket counted, or escaped."""

    def __init__(self) -> None:
        self.curly = self.square = self.fracs = 0
        self.bracketed = self.square_curly = self.paired = self.curly_curly = 0
        self.out: list[str] = []

    def keyword(self, text: str, at: int) -> int:
        """How many characters an opening keyword at ``at`` takes, or 0."""
        for words, kind in ((BRACED, "curly"), (BRACKETED, "square"), (PAIRED, "paired")):
            word = next((one for one in words if text.startswith(one, at)), "")
            if word:
                self.opened(kind)
                self.out.append(word)
                return len(word)
        return 0

    def opened(self, kind: str) -> None:
        if kind == "square":
            self.bracketed += 1
            self.square += 1
        elif kind == "paired":
            self.paired += 1
            self.fracs += 1
            self.curly += 1
        else:
            self.curly += 1

    def other(self, text: str, at: int) -> int:
        """One character, or a pair, that is no keyword: kept, escaped, or counted."""
        two = text[at : at + 2]
        if two == "}{" and self.fracs:
            self.fracs -= 1
            self.curly_curly += 1
        elif two == "]{" and self.square:
            self.square_curly += 1
            self.curly += 1
            self.square -= 1
        elif two[:1] == "@" and two[1:] in ("{", "}", "[", "]"):
            pass
        else:
            return self.single(text[at])
        self.out.append(two)
        return 2

    def single(self, char: str) -> int:
        if char == "}" and self.curly:
            self.curly -= 1
        elif char in "[]{}":
            self.out.append("@")
        self.out.append(char)
        return 1

    def verdict(self) -> None:
        errors = (
            (self.bracketed != self.square_curly, 'Invalid number of "]{"'),
            (self.paired != self.curly_curly, 'Error in syntax of  "#frac"'),
            (self.curly > 0, 'Missing "}"'),
            (self.square > 0, 'Missing "]"'),
        )
        for failed, said in errors:
            if failed:
                raise LatexError(said)


def check(text: str) -> str:
    """``CheckLatexSyntax``: the formula with its delimiters and loose braces made text.

    Raises :class:`LatexError`, as ROOT refuses to draw the formula, when an
    operator is left open.
    """
    text = _delimiters(text)
    checker = _Checker()
    at = 0
    while at < len(text):
        at += checker.keyword(text, at) or checker.other(text, at)
    checker.verdict()
    return "".join(checker.out)
