"""``TPRegexp`` and ``TPMERegexp``: ROOT's regular expressions, over Python's ``re``.

A macro's patterns are Perl's - ``\\b``, ``\\w``, ``\\d``, ``$1`` in a replacement - which
``re`` reads the same way, so what is ROOT's here is the shape around it: a match's full
text and every group as a ``TObjArray`` of ``TObjString``, a substitution that rewrites the
``TString`` it was given, and ``TPMERegexp``'s own walk over a string - one match per call,
printed in ROOT's layout, split the way PCRE splits, and substituted with the case
controls a Perl replacement has.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from .collections import TObjArray, TObjString
from .objects import TObject
from .strings import TString, _text

__all__ = ["TPMERegexp", "TPRegexp"]


#: ``$1``..``$9`` and ``\\0`` in a replacement, as Perl writes them: a group of the match.
_REFERENCE = re.compile(r"\$(\d+)|\\(\d+)")


def _flags(options: Any) -> int:
    """The ``re`` flags ROOT's option letters ask for."""
    asked = str(options)
    flags = 0
    if "i" in asked:
        flags |= re.IGNORECASE
    if "m" in asked:
        flags |= re.MULTILINE
    if "s" in asked:
        flags |= re.DOTALL
    if "x" in asked:
        flags |= re.VERBOSE
    return flags


def _group(match: Any, at: int) -> str:
    """The text of group ``at`` of a match, empty where the group did not take part."""
    try:
        found = match.group(at)
    except IndexError:
        return ""
    return "" if found is None else str(found)


def _written(match: Any, replacement: str) -> str:
    """``replacement`` with the groups of ``match`` in place of their ``$n``."""
    return _REFERENCE.sub(lambda ref: _group(match, int(ref.group(1) or ref.group(2))), replacement)


class TPRegexp(TObject):
    """``TPRegexp``: one pattern, matched against a ``TString``, and substituted in it."""

    def __init__(self, pattern: Any = "", options: Any = "") -> None:
        super().__init__()
        self._pattern = str(pattern)
        self._options = str(options)
        self._re = re.compile(self._pattern, _flags(options))

    def __repr__(self) -> str:
        return f"<TPRegexp {self._pattern!r}>"

    def GetPattern(self) -> str:
        """``GetPattern``: the expression as it was written."""
        return self._pattern

    def GetOptions(self) -> str:
        """``GetOptions``: the modifier letters it was made with."""
        return self._options

    def First(self, text: Any) -> str:
        """``First``: the text of the first match in ``text``, or nothing - what
        ``TString::operator()(const TPRegexp&)`` answers with."""
        found = self._re.search(str(text))
        return "" if found is None else found.group(0)

    def MatchB(self, text: Any) -> bool:
        """``MatchB``: is there a match anywhere in ``text``?"""
        return self._re.search(str(text)) is not None

    def MatchS(self, text: Any) -> TObjArray:
        """``MatchS``: the match and its groups, the whole of it first, as ``TObjString``s."""
        found = self._re.search(str(text))
        made = TObjArray()
        if found is not None:
            for at in range(0, (found.re.groups or 0) + 1):
                made.Add(TObjString(_group(found, at)))
        return made

    def Match(self, text: Any, start: int = 0) -> int:
        """``Match``: where the first match at or after ``start`` is, or ``-1``."""
        found = self._re.search(str(text), max(int(start), 0))
        return -1 if found is None else found.start()

    def Substitute(self, text: Any, replacement: Any, options: Any = "") -> int:
        """``Substitute``: write ``replacement`` over each match in ``text``; how many were made.

        ``text`` is a ``TString``, which is changed where it is; ``$1`` and its kin in the
        replacement are the groups, and the ``g`` option substitutes every match, not the first.
        """
        held = _text(text)
        written = str(replacement)
        every = "g" in str(options) or "g" in self._options  # the call's, or the object's own
        count = 0

        def once(found: Any) -> str:
            nonlocal count
            count += 1
            return _written(found, written)

        made, _how_many = self._re.subn(once, held, count=0 if every else 1)
        if isinstance(text, TString):
            text._s = made
        return count


class TPMERegexp(TPRegexp):
    """``TPMERegexp``: ``TPRegexp`` as Perl reads it - a walk over the matches, a split."""

    def __init__(self, pattern: Any = "", options: Any = "") -> None:
        super().__init__(pattern, options)
        self._text = ""
        self._at = 0
        self._groups: list[str] = []

    # -- matching ---------------------------------------------------------------------------

    def Match(self, text: Any, start: int = 0) -> bool:
        """``Match``: the next match in ``text``, remembered for ``Print`` and ``Split``.

        With the ``g`` option each call continues after the match the last one found, which
        is how ROOT walks the matches of a string; without it every call starts at the
        beginning.
        """
        self._text = _text(text)
        from_at = self._at if "g" in self._options else max(int(start), 0)
        found = self._re.search(self._text, from_at)
        if found is None:
            self._groups = []
            return False
        self._at = found.end() if found.end() > found.start() else found.end() + 1
        self._groups = [_group(found, at) for at in range(0, (found.re.groups or 0) + 1)]
        return True

    def NMatches(self) -> int:
        """``NMatches``: how many strings the last match found - the whole and its groups."""
        return len(self._groups)

    def GetMatch(self, at: int) -> str:
        """``GetMatch``: the text of group ``at`` of the last match (``0`` is the whole)."""
        return self._groups[at] if at < len(self._groups) else ""

    def Print(self, option: Any = "") -> None:
        """``Print``: ROOT's own layout of the pattern - and, for ``Print("all")``, of the
        string and every group the last match found."""
        said = [f"Regexp='{self._pattern}', Opts='{self._options}'"]
        if str(option) == "all":
            said.append(f"  last string='{self._text}'")
            said.append(f"  number of matches = {len(self._groups)}")
            said.extend(f"  {at} - {text}" for at, text in enumerate(self._groups))
        sys.stdout.write("\n".join(said) + "\n")

    # -- splitting and substituting ----------------------------------------------------------

    def Split(self, text: Any, maxfields: int = 0) -> int:
        """``Split``: ``text`` cut at each match, the fields kept as this object's matches.

        ``maxfields`` is ROOT's: above zero, at most that many fields with the rest of the
        string in the last; zero, every field but the empty ones a trailing match leaves;
        below zero, those too.
        """
        held = _text(text)
        fields: list[str] = []
        last = 0
        for found in self._re.finditer(held):
            if found.start() == found.end() == last:
                continue  # a match of nothing where the last one ended names no field
            fields.append(held[last : found.start()])
            fields.extend(_group(found, at) for at in range(1, (found.re.groups or 0) + 1))
            last = found.end()
            if maxfields > 0 and len(fields) >= maxfields - 1:
                break
        fields.append(held[last:])
        if maxfields == 0:
            while fields and fields[-1] == "":
                fields.pop()
        self._text, self._groups = held, fields
        return len(fields)

    def Substitute(self, text: Any, replacement: Any, options: Any = "") -> int:
        """``Substitute``: each match replaced, with Perl's ``\\u`` and ``\\U...\\E`` cases."""
        held = _text(text)
        written = str(replacement)
        every = "g" in str(options) or "g" in self._options  # the call's, or the object's own
        count = 0

        def once(found: Any) -> str:
            nonlocal count
            count += 1
            return _cased(_written(found, written))

        made, _how_many = self._re.subn(once, held, count=0 if every else 1)
        if isinstance(text, TString):
            text._s = made
        return count


#: What Perl's case controls ask for: the mode every letter takes, and the next letter's.
_CASE_CONTROLS = {"u": ("", "upper"), "l": ("", "lower"), "U": ("upper", ""), "L": ("lower", "")}


def _cased(text: str) -> str:
    """``text`` under Perl's case controls: ``\\u`` lifts the next letter, ``\\l`` lowers it,
    ``\\U`` and ``\\L`` every letter until ``\\E``."""
    out: list[str] = []
    mode = once = ""
    at = 0
    while at < len(text):
        if _is_control(text, at):
            mode, once = _CASE_CONTROLS.get(text[at + 1], ("", ""))
            at += 2
            continue
        letter, once = _applied(text[at], mode, once)
        out.append(letter)
        at += 1
    return "".join(out)


def _is_control(text: str, at: int) -> bool:
    """Is there one of Perl's case controls at ``at``?"""
    return at + 1 < len(text) and text[at] == "\\" and text[at + 1] in "ulULE"


def _applied(letter: str, mode: str, once: str) -> tuple[str, str]:
    """``letter`` cased as ``once`` then ``mode`` ask, and ``once`` spent."""
    kind = once or mode
    if not kind:
        return letter, ""
    return (letter.upper() if kind == "upper" else letter.lower()), ""
