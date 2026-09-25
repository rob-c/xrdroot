"""ROOT's ``TLatex`` in matplotlib's mathtext.

ROOT spells mathematics with ``#`` where TeX spells it with a backslash -
``#mu^{+}#mu^{-}``, ``#sqrt{s} = 13 TeV``, ``p_{T} [GeV]`` - and, unlike TeX,
is never in or out of a math mode: every letter is upright unless
``#it`` says otherwise, and a space is a space. So the translation is of
the whole string into one mathtext expression, the words in it kept upright
with ``\\mathrm``, the ``#`` commands mapped to mathtext's, and a string with
nothing mathematical in it left as the plain text it is.

What matplotlib's mathtext has no equivalent of is translated to the
nearest thing it does: ``#splitline`` is two lines stacked as a fraction
with no bar, and ``#font``, ``#color``, ``#scale``, ``#kern`` and
``#lower`` keep their text and drop the adjustment. A ``#`` command this
does not know is kept as the word it is, rather than handed to mathtext to
fail on.
"""

from __future__ import annotations

__all__ = ["translate"]

#: The ``#`` commands that are one symbol, against mathtext's name for it.
SYMBOLS: dict[str, str] = {
    **{
        name: name
        for name in (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi pi "
            "rho sigma tau upsilon phi chi psi omega varepsilon vartheta varphi varsigma "
            "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
            "pm mp times div cdot circ bullet infty partial nabla ell hbar wp Im Re aleph "
            "approx sim simeq equiv propto neq leq geq ll gg in notin subset supset "
            "subseteq supseteq cap cup wedge vee oplus otimes odot perp parallel angle "
            "forall exists neg sum prod int oint leftarrow rightarrow leftrightarrow "
            "Leftarrow Rightarrow Leftrightarrow uparrow downarrow dagger ddagger"
        ).split()
    },
    "Alpha": "mathrm{A}",
    "Beta": "mathrm{B}",
    "Epsilon": "mathrm{E}",
    "Zeta": "mathrm{Z}",
    "Eta": "mathrm{H}",
    "Iota": "mathrm{I}",
    "Kappa": "mathrm{K}",
    "Mu": "mathrm{M}",
    "Nu": "mathrm{N}",
    "Omicron": "mathrm{O}",
    "omicron": "mathrm{o}",
    "Rho": "mathrm{P}",
    "Tau": "mathrm{T}",
    "Chi": "mathrm{X}",
    "minus": "minus",
    "void": "emptyset",
    "downleftarrow": "swarrow",
    "cbar": "bar{c}",
    "lbar": "bar{l}",
    "arcbottom": "smile",
    "arctop": "frown",
}
#: The commands that take one argument in braces and keep it.
ACCENTS: dict[str, str] = {
    "bar": "bar",
    "hat": "hat",
    "tilde": "tilde",
    "vec": "vec",
    "dot": "dot",
    "ddot": "ddot",
    "check": "check",
    "acute": "acute",
    "grave": "grave",
    "underline": "underline",
    "overline": "overline",
}
#: The commands that change the face of their text, against mathtext's.
FACES: dict[str, str] = {"it": "mathit", "bf": "mathbf", "mbox": "mathrm"}
#: The commands that take a setting in square brackets and then keep their text.
ADJUSTMENTS = frozenset({"font", "color", "scale", "kern", "lower", "sqrt"})
#: ``#left`` and ``#right``, which take the delimiter after them.
DELIMITERS = frozenset({"left", "right"})
#: Characters mathtext would read as its own, and what they are in it.
ESCAPES = {
    "%": r"\%",
    "$": r"\$",
    "\\": r"\backslash ",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
}
#: What makes a string mathematics rather than plain text.
MARKS = "#^_"
#: What an empty argument becomes: a space, which mathtext will take.
EMPTY = r"\ "


class _Reader:
    """One pass over the characters of a ``TLatex`` string."""

    __slots__ = ("text", "at")

    def __init__(self, text: str) -> None:
        self.text = text
        self.at = 0

    def peek(self) -> str:
        return self.text[self.at] if self.at < len(self.text) else ""

    def take(self) -> str:
        char = self.peek()
        self.at += 1
        return char

    def name(self) -> str:
        """The letters of a ``#`` command's name."""
        start = self.at
        while self.peek().isalpha():
            self.at += 1
        return self.text[start : self.at]

    def bracketed(self) -> str:
        """A ``[setting]``, if one is next, which is skipped over and returned."""
        end = self.text.find("]", self.at)
        if self.peek() != "[" or end < 0:
            return ""  # a bracket never closed is text, not a setting
        setting = self.text[self.at + 1 : end]
        self.at = end + 1
        return setting

    def group(self) -> str:
        """The next ``{...}``, translated - or, with no brace, the next character.

        An empty one is a space, since mathtext will not take an empty group
        where an argument has to be.
        """
        if self.peek() != "{":
            return _plain(self.take()) or EMPTY
        self.at += 1
        inner = _body(self, closing="}")
        self.at += 1  # the closing brace
        return inner or EMPTY


def _plain(run: str) -> str:
    """Characters that are text, upright, with their spaces kept."""
    if not run:
        return ""
    escaped = "".join(ESCAPES.get(char, char) for char in run).replace(" ", r"\ ")
    return r"\mathrm{" + escaped + "}"


def _command(reader: _Reader) -> str:
    """What one ``#`` command becomes, the ``#`` already read."""
    name = reader.name()
    if not name:
        return _plain(reader.take())
    if name in DELIMITERS:
        return _plain(reader.take())  # the delimiter, at the size of the text
    if name == "frac":
        top = reader.group()
        return r"\frac{" + top + "}{" + reader.group() + "}"
    if name == "splitline":
        top = reader.group()
        return r"\genfrac{}{}{0}{}{" + top + "}{" + reader.group() + "}"
    return _argument(reader, name)


def _argument(reader: _Reader, name: str) -> str:
    """A command of one argument, a symbol, or a word this does not know."""
    setting = reader.bracketed() if name in ADJUSTMENTS else ""
    if name == "sqrt":
        root = f"[{setting}]" if setting else ""
        return r"\sqrt" + root + "{" + reader.group() + "}"
    if name in ACCENTS:
        return f"\\{ACCENTS[name]}{{{reader.group()}}}"
    if name in FACES:
        return "{" + reader.group().replace(r"\mathrm{", "\\" + FACES[name] + "{") + "}"
    if name in ADJUSTMENTS:
        return "{" + reader.group() + "}"
    if name in SYMBOLS:
        return f"{{\\{SYMBOLS[name]}}}"
    return _plain(name)


def _script(reader: _Reader, mark: str) -> str:
    """A superscript or a subscript, of a group or of one character."""
    return mark + "{" + reader.group() + "}"


def _body(reader: _Reader, closing: str = "") -> str:
    """Everything up to ``closing``, or to the end, as mathtext."""
    parts: list[str] = []
    run: list[str] = []
    while reader.peek() and reader.peek() != closing:
        char = reader.take()
        if char in MARKS or char == "{":
            parts.append(_plain("".join(run)))
            run = []
            parts.append(_special(reader, char))
        else:
            run.append(char)
    parts.append(_plain("".join(run)))
    return "".join(parts)


def _special(reader: _Reader, char: str) -> str:
    """A ``#`` command, a script, or a group in braces."""
    if char == "#":
        return _command(reader)
    if char == "{":
        reader.at -= 1
        return "{" + reader.group() + "}"
    return _script(reader, char)


def translate(text: str) -> str:
    """``text`` as matplotlib draws it: ROOT's ``TLatex`` as mathtext.

    >>> translate("#mu^{+}#mu^{-} mass")
    '${\\\\mu}^{\\\\mathrm{+}}{\\\\mu}^{\\\\mathrm{-}}\\\\mathrm{\\\\ mass}$'
    >>> translate("Entries")
    'Entries'
    """
    if not any(mark in text for mark in MARKS):
        return text.replace("$", r"\$")
    body = _body(_Reader(text))
    return f"${body}$" if body else ""
