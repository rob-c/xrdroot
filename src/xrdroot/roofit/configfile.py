"""RooFit's ASCII configuration files: ``RooArgSet::writeToFile`` and ``readFromFile``.

An argument set is written as one line per member - ``f =  0.45 +/- 0.03
L(0 - 1) `` - and read back line by line, so that the values, errors,
constancy, ranges and bins a fit found can be kept in a text file, edited,
and read into another fit. A line is ``RooRealVar::writeToStream(os, false)``'s:
the value at five significant figures by its error's precision, then ``C``
for a constant, ``L(lo - hi)`` for the range, ``B(n)`` for the bins other
than a hundred, and the unit as a comment.

Reading keeps ROOT's constructs - sections (``[Name]``, and only the wanted
one is read), ``include file``, ``echo``, ``abort``, ``if``/``else``/
``endif`` over a :class:`~xrdroot.roofit.formula.RooFormula` of the set's own
values, and the deprecated ``P(lo - hi : n)`` and ``F(lo - hi : n)`` - and
ROOT's diagnostics, message for message, because a tutorial's output is its
warnings as much as its numbers.

Values are read as ``RooFit::toDouble`` reads them: ``INF`` is not a number
there either, so a variable without a range does not read back the
``L(-INF - +INF)`` line it wrote.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from ..errors import UnsupportedFeatureError
from .formatting import format_var
from .messages import ERROR, INFO, WARNING, log
from .printing import address, g

__all__ = ["read_file", "read_text", "write_file", "write_text"]

#: ``RooRealVar::_printSigDigits``: the significant figures a value is written at.
SIG_DIGITS = 5

#: The characters a token is split at, as ``RooStreamParser`` splits them: ``=`` alone at
#: a set's level, the parentheses and commas of a value's extras at an argument's. A
#: number's sign, dot and exponent stay within its token.
PUNCTUATION = "=(),[]"

#: The characters of an error's sign, which is three tokens however it is written: ``+/-``.
SIGNS = frozenset("+-/")


def _value_text(one: Any) -> str:
    """One member's value, as its own ``writeToStream(os, false)`` writes it."""
    from .categories import RooCategory
    from .variables import RooRealVar

    if isinstance(one, RooCategory):
        return str(one.getLabel())
    if not isinstance(one, RooRealVar):
        raise UnsupportedFeatureError(
            f"RooFit's ASCII configuration files hold real variables and categories; "
            f"{one.GetName()!r} is a {type(one).__name__}, which has no form in them here - "
            f"write the set's variables with writeToFile instead."
        )
    return _real_value_text(one)


def _real_value_text(var: Any) -> str:
    """``RooRealVar::writeToStream(os, false)``: value and error, constancy, range, bins, unit."""
    text = format_var(var, SIG_DIGITS, "EFA") + " "
    if var.isConstant():
        text += "C "
    text += "L(" + (g(var.getMin()) if var.hasMin() else "-INF")
    text += (" - " + g(var.getMax())) if var.hasMax() else " - +INF"
    text += ") "
    if var.getBins() != 100:
        text += f"B({var.getBins()}) "
    if var.getUnit():
        text += f"// [{var.getUnit()}]"
    return text


def write_text(collection: Any) -> str:
    """``RooArgSet::writeToStream(os, false)``: a line per member, ``name = value``."""
    return "".join(f"{one.GetName()} = {_value_text(one)}\n" for one in collection)


def write_file(collection: Any, filename: Any) -> None:
    """``RooArgSet::writeToFile``: the members as text, or ROOT's error when it opens not."""
    text = write_text(collection)
    try:
        Path(filename).write_text(text, newline="\n")
    except OSError:
        log(
            collection,
            ERROR,
            "InputArguments",
            f"RooArgSet::writeToFile({collection.GetName()}) error opening file {filename}",
        )


def read_file(
    collection: Any,
    filename: Any,
    flag_read_att: Any = None,
    section: Any = None,
    verbose: bool = False,
) -> bool:
    """``RooArgSet::readFromFile``: the file's values read into the members of that name.

    ``flag_read_att`` marks every member a line of the file set, ``section`` reads
    only that ``[section]`` of the file, and the answer is ROOT's: ``true`` when
    reading went wrong - the file did not open, a line had no ``=``, an
    ``include`` failed - and ``false`` when it did not.
    """
    try:
        text = Path(filename).read_text()
    except OSError:
        log(
            collection,
            ERROR,
            "InputArguments",
            f"RooArgSet::readFromFile({collection.GetName()}) error opening file {filename}",
        )
        return True
    return read_text(collection, text, flag_read_att, section, verbose)


def read_text(
    collection: Any,
    text: str,
    flag_read_att: Any = None,
    section: Any = None,
    verbose: bool = False,
) -> bool:
    """``RooArgSet::readFromStream``: the text's values read into the members of that name."""
    return _Reader(collection, flag_read_att, section, verbose).read(text)


# -- the text of one file ---------------------------------------------------------


def _comment_cut(line: str) -> str:
    """The line without its comment - ``//`` or ``#``, as ``RooStreamParser`` chops them."""
    cuts = [index for index in (line.find("//"), line.find("#")) if index >= 0]
    return line[: min(cuts)] if cuts else line


def _lines(text: str) -> list[str]:
    """The file's lines, a line ending in ``\\\\`` continued by the one after it."""
    lines: list[str] = []
    pending = ""
    for line in text.splitlines():
        if not line.endswith("\\\\"):
            lines.append(pending + line)
            pending = ""
            continue
        pending += line[:-2] + " "
    if pending:
        lines.append(pending)
    return lines


def _tokens(text: str) -> list[str]:
    """The line's tokens: a name or a number whole, each punctuation character alone."""
    tokens: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in " \t":
            index += 1
        elif char in PUNCTUATION:
            tokens.append(char)
            index += 1
        else:
            end = index + 1
            while end < len(text) and text[end] not in " \t" and text[end] not in PUNCTUATION:
                end += 1
            word = text[index:end]
            tokens.extend(word if set(word) <= SIGNS else [word])
            index = end
    return tokens


def _number(token: str) -> float:
    """A value's token as a number, as ``RooFit::toDouble`` reads it - which ``INF`` is not."""
    if "inf" in token.lower():
        raise ValueError("Conversion to floating point failed")
    try:
        return float(token)
    except ValueError:
        raise ValueError("Conversion to floating point failed") from None


class _Line:
    """A cursor over the tokens after ``name =``, expecting them as RooFit's parser does."""

    def __init__(self, var: Any, tokens: list[str]) -> None:
        self.var, self.tokens, self.index = var, tokens, 0

    def at_end(self) -> bool:
        return self.index >= len(self.tokens)

    def take(self) -> str:
        """The next token - the empty one past the end, as an exhausted stream answers."""
        token = self.tokens[self.index] if not self.at_end() else ""
        self.index += 1
        return token

    def expect(self, *wanted: str) -> bool:
        """``RooStreamParser::expectToken``: the next token, if it is one of ``wanted``."""
        token = self.take()
        if token in wanted:
            return True
        log(
            self.var,
            WARNING,
            "InputArguments",
            f"RooRealVar::readFromStream({self.var.GetName()}): parse error, "
            f"expected '{wanted[0]}', got '{token}'",
        )
        return False


class _RealLine:
    """The tokens after ``name =`` for one real variable, read as ``RooRealVar`` reads them.

    A token that does not fit its place ends the line, as ROOT's parser gives up on
    the rest of it, and a value seen before that point is set all the same.
    """

    #: ``RooRealVar::readFromStream``'s letters, and the method each of them is read by.
    READERS: ClassVar[dict[str, str]] = {
        "+": "_error", "C": "_constant", "L": "_range", "B": "_bins", "P": "_plot", "F": "_fit",
    }  # fmt: skip

    def __init__(self, var: Any) -> None:
        self.var: Any = var
        self.value: float | None = None
        self.constant = False

    def read(self, tokens: list[str]) -> bool:
        """The line; ``false`` - RooRealVar's answer always, a parse error being a warning."""
        self._scan(_Line(self.var, tokens))
        if self.value is not None:
            self.var.setVal(self.value)
        return False

    def _scan(self, line: _Line) -> None:
        while not line.at_end():
            token = line.take()
            handler = self.READERS.get(token)
            if handler is None:
                self.value = _number(token)
            elif not getattr(self, handler)(line):
                return

    def _constant(self, line: _Line) -> bool:
        self.var.setConstant(True)
        self.constant = True
        return True

    def _error(self, line: _Line) -> bool:
        """``+/- e``, or ``+/- (lo, hi)`` for an error that is not symmetric.

        Whichever way the sign's characters were split from their neighbours, they are
        the two tokens after ``+``: the error itself starts at the one that is neither.
        """
        if not line.expect("/", "-"):
            return False
        token = line.take()
        while token in ("/", "-"):
            token = line.take()
        if token != "(":
            self.var.setError(_number(token))
            return True
        low = _number(line.take())
        if not line.expect(","):
            return False
        high = _number(line.take())
        if not line.expect(")"):
            return False
        self.var.setAsymError(low, high)
        return True

    def _range(self, line: _Line) -> bool:
        """``L(lo - hi)``: the fit range, and - unless ``C`` came first - a freed variable."""
        if not self._paired(line, self.var.setRange):
            return False
        if not self.constant:
            self.var.setConstant(False)
        return True

    def _bins(self, line: _Line) -> bool:
        """``B(n)``: how many bins the variable has."""
        if not line.expect("("):
            return False
        token = line.take()
        if not line.expect(")"):
            return False
        self.var.setBins(int(token))
        return True

    def _plot(self, line: _Line) -> bool:
        """``P(lo - hi : n)``: read and warned about, as ROOT keeps no plot range any more."""
        if not _deprecated_range(line):
            return False
        log(
            self.var,
            WARNING,
            "Eval",
            f"RooRealVar::readFromStream({self.var.GetName()}) WARNING: plot range "
            "deprecated, removed P(...) token",
        )
        return True

    def _fit(self, line: _Line) -> bool:
        """``F(lo - hi : n)``: read and warned about, ROOT's fit range being ``L``'s now."""
        if not _deprecated_range(line):
            return False
        log(
            self.var,
            WARNING,
            "Eval",
            f"RooRealVar::readFromStream({self.var.GetName()}) WARNING: F(lo-hi:bins) token "
            "deprecated, use L(lo-hi) B(bins)",
        )
        if not self.constant:
            self.var.setConstant(False)
        return True

    def _paired(self, line: _Line, apply: Callable[[float, float], Any]) -> bool:
        """``(lo - hi)``: two numbers handed to ``apply``."""
        if not line.expect("("):
            return False
        low = _number(line.take())
        if not line.expect("-"):
            return False
        high = _number(line.take())
        if not line.expect(")"):
            return False
        apply(low, high)
        return True


def _deprecated_range(line: _Line) -> bool:
    """``(lo - hi : n)``: the shape the deprecated ``P`` and ``F`` were both written with."""
    if not line.expect("("):
        return False
    _number(line.take())  # the range's low end and its high end are read and dropped
    if not line.expect("-"):
        return False
    _number(line.take())
    if not line.expect(":"):
        return False
    line.take()  # the bins
    return line.expect(")")


def _read_real(var: Any, tokens: list[str]) -> bool:
    """``RooRealVar::readFromStream(os, false)``: the value and what follows it on the line."""
    var.removeError()
    var.removeAsymError()
    return _RealLine(var).read(tokens)


def _read_category(var: Any, tokens: list[str]) -> bool:
    """``RooCategory::readFromStream``: a state's number, or else its label."""
    token = tokens[0] if tokens else ""
    if token.isdigit() and var.hasIndex(int(token)):
        return bool(var.setIndex(int(token)))
    return bool(var.setLabel(token))


def _read_value(var: Any, tokens: list[str]) -> bool:
    """The tokens after ``name =`` read into ``var``; ``true`` when they could not be."""
    from .categories import RooCategory
    from .variables import RooRealVar

    if isinstance(var, RooCategory):
        return _read_category(var, tokens)
    if not isinstance(var, RooRealVar):
        raise UnsupportedFeatureError(
            f"RooFit's ASCII configuration files hold real variables and categories; "
            f"{var.GetName()!r} is a {type(var).__name__}, which has no form in them here - "
            f"read the set's variables with readFromFile instead."
        )
    return _read_real(var, tokens)


# -- the constructs of one file ---------------------------------------------------


class _Reader:
    """One ``RooArgSet::readFromStream``: the section wanted, the conditionals, the errors."""

    def __init__(self, collection: Any, flag_read_att: Any, section: Any, verbose: bool) -> None:
        self.collection = collection
        self.flag = "" if flag_read_att is None else str(flag_read_att)
        self.section = "" if section is None else f"[{section}]"
        self.in_section = section is None
        self.verbose = bool(verbose)
        self.failed = False
        self.conditions = [True]
        self.any_true = [False]
        self.last_was_else = False

    def read(self, text: str) -> bool:
        """The file; ``true`` when it ended badly - an error, an abort, an unclosed ``if``."""
        for line in _lines(text):
            if self._line(line):
                return True
        if len(self.conditions) != 1:
            self._say("missing 'endif'")
            return True
        return self.failed

    def _say(self, text: str, level: int = ERROR) -> None:
        name = self.collection.GetName()
        log(self.collection, level, "InputArguments", f"RooArgSet::readFromStream({name}): {text}")

    def _line(self, line: str) -> bool:
        """One line; ``true`` when reading stops - an abort, or an error that ends it."""
        text = _comment_cut(line).strip()
        if not text:
            return False
        head = _tokens(text)[0]
        rest = text[len(head) :].strip()
        if head == "include":
            return self._include(rest)
        if head == "[":
            self._header(text)
            return False
        if not self.in_section:
            return False
        return self._inside(head, rest)

    def _header(self, text: str) -> None:
        """``[Name]``: from here on the wanted section's lines are read, and only them."""
        end = text.find("]")
        name = " ".join(text[1 : end if end >= 0 else len(text)].split())
        if self.section:
            self.in_section = f"[{name}]" == self.section

    def _inside(self, head: str, rest: str) -> bool:
        if head == "if":
            self._push(rest)
            return False
        if head == "else":
            return self._else(rest)
        if head == "endif":
            return self._endif()
        if not self.conditions[-1]:
            return False
        return self._statement(head, rest)

    def _statement(self, head: str, rest: str) -> bool:
        if head == "echo":
            self._say(f">> {rest}")
            return False
        if head == "abort":
            self._say("USER ABORT")
            return True
        return self._assign([head, *_tokens(rest)])

    def _assign(self, tokens: list[str]) -> bool:
        """``name = value``: one member's line."""
        arg = self.collection.find(tokens[0])
        if arg is None or arg.getAttribute("Dynamic"):
            if self.verbose:
                self._say(f"argument {tokens[0]} not in list, ignored")
            return False
        if len(tokens) < 2 or tokens[1] != "=":
            self._say(f"missing '=' sign: {address(arg)}")
            self.failed = True
            return False
        error = _read_value(arg, tokens[2:])
        if not error and self.flag:
            arg.setAttribute(self.flag, True)
        self.failed = self.failed or error
        return False

    # -- conditionals ---------------------------------------------------------------

    def _push(self, expression: str) -> None:
        """``if (expr)``: its condition pushed - beside the last one, if it was ``else if``."""
        from .formula import RooFormula

        status = bool(RooFormula(expression, self.collection).evaluate({}))
        if self.last_was_else:
            self.any_true[-1] = self.any_true[-1] or status
            self.last_was_else = False
        else:
            self.conditions.append(status)
            self.any_true.append(status)
        self.conditions[-1] = status

    def _else(self, rest: str) -> bool:
        """``else``, or ``else if``: the branch taken when nothing before it was."""
        if len(self.conditions) == 1:
            self._say("unmatched 'else'")
        if not rest:
            self.conditions[-1] = not self.any_true[-1]
            return False
        tokens = _tokens(rest)
        if tokens[0] != "if":
            self._say(f"syntax error: 'else {tokens[0]}'")
            self.failed = True
            return False
        if self.any_true[-1]:
            self.conditions[-1] = False
            return False
        self.last_was_else = True
        self._push(rest[len(tokens[0]) :].strip())
        return False

    def _endif(self) -> bool:
        if len(self.conditions) == 1:
            self._say("unmatched 'endif'")
            return True
        self.conditions.pop()
        self.any_true.pop()
        return False

    def _include(self, filename: str) -> bool:
        """``include file``: another file's lines here - the wanted section's, or all of them."""
        if not filename:
            self._say("no filename found after include statement")
            return True
        try:
            text = Path(filename).read_text()
        except OSError:
            self._say(f"cannot open include file {filename}")
            return True
        self._say(f"processing include file {filename}", INFO)
        wanted = None if self.in_section else self.section[1:-1]
        return _Reader(self.collection, self.flag, wanted, self.verbose).read(text)
