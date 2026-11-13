"""The gmock matchers tests hand ``EXPECT_THAT``: ``HasSubstr`` above all.

A matcher answers whether a value matches and describes what it wants, in
gmock's words, for the failure: ``Expected: has substring "fit"``. These
are the matchers whose meaning is plain on strings and numbers; any other
of gmock's is refused by name where it is used.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from typing import Any

from ..streams import ostringstream
from .values import equal, show

__all__ = [
    "Matcher",
    "MakeMatcher",
    "HasSubstr",
    "StartsWith",
    "EndsWith",
    "ContainsRegex",
    "MatchesRegex",
    "StrEq",
    "StrNe",
    "Eq",
    "Ne",
    "Lt",
    "Le",
    "Gt",
    "Ge",
    "IsEmpty",
    "Not",
    "AllOf",
    "AnyOf",
]


class Matcher:
    """A test of a value, and gmock's description of what passes it."""

    def __init__(self, test: Callable[[Any], bool], description: str) -> None:
        self.test = test
        self.description = description
        #: What the last value tested failed by, as the matcher explained it: may be empty.
        self.explanation = ""

    def matches(self, value: Any) -> bool:
        return bool(self.test(value))

    def describe(self) -> str:
        return self.description


def MakeMatcher(test: Callable[[Any, Any], Any], name: str, description: Any,
                *values: Any) -> Matcher:  # fmt: skip
    """A matcher ``MATCHER_P2(name, p0, p1, description) { ... }`` defines.

    ``test`` is its body, given the value and the ``result_listener`` it may
    explain a mismatch to. A matcher with no description is described, as
    gmock describes it, by its name in words and its parameters.
    """
    words = str(description) or " ".join([*_words(name), ", ".join(map(show, values))]).strip()
    made = Matcher(lambda value: _explained(made, test, value), words)
    return made


def _explained(matcher: Matcher, test: Callable[[Any, Any], Any], value: Any) -> bool:
    listener = ostringstream()
    matched = bool(test(value, listener))
    matcher.explanation = listener.str()
    return matched


def _words(name: str) -> list[str]:
    """``RelativeNear`` as gmock writes it in a description: ``relative near``."""
    return [word.lower() for word in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])", name)]


def _text(test: Callable[[str, str], bool], words: str) -> Callable[[Any], Matcher]:
    """A matcher of strings, made from a test of the value and the matcher's argument."""

    def made(argument: Any) -> Matcher:
        wanted = str(argument)
        return Matcher(lambda value: test(str(value), wanted), f"{words} {show(wanted)}")

    return made


HasSubstr = _text(lambda value, wanted: wanted in value, "has substring")
StartsWith = _text(str.startswith, "starts with")
EndsWith = _text(str.endswith, "ends with")
ContainsRegex = _text(lambda v, w: re.search(w, v) is not None, "contains regular expression")
MatchesRegex = _text(lambda v, w: re.fullmatch(w, v) is not None, "matches regular expression")
StrEq = _text(operator.eq, "is equal to")
StrNe = _text(operator.ne, "isn't equal to")


def _relation(test: Callable[[Any, Any], bool], words: str) -> Callable[[Any], Matcher]:
    def made(argument: Any) -> Matcher:
        return Matcher(lambda value: test(value, argument), f"{words} {show(argument)}")

    return made


Eq = _relation(equal, "is equal to")
Ne = _relation(lambda value, wanted: not equal(value, wanted), "isn't equal to")
Lt = _relation(operator.lt, "is <")
Le = _relation(operator.le, "is <=")
Gt = _relation(operator.gt, "is >")
Ge = _relation(operator.ge, "is >=")


def IsEmpty() -> Matcher:
    return Matcher(lambda value: len(value) == 0, "is empty")


def Not(inner: Matcher) -> Matcher:
    return Matcher(lambda value: not inner.matches(value), f"not ({inner.describe()})")


def AllOf(*inner: Matcher) -> Matcher:
    words = ") and (".join(each.describe() for each in inner)
    return Matcher(lambda value: all(each.matches(value) for each in inner), f"({words})")


def AnyOf(*inner: Matcher) -> Matcher:
    words = ") or (".join(each.describe() for each in inner)
    return Matcher(lambda value: any(each.matches(value) for each in inner), f"({words})")
