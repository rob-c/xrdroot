"""The assertions: ``EXPECT_EQ`` and the rest, as the checks ``gtest/gtest.h``'s macros call.

Each check is handed the values, the C++ text of each (``#a``) and the
place, and answers whether the assertion holds. When it does not, the
failure is recorded against the test running and its note begun - worded
as GoogleTest words it - and the check answers ``False``, which runs the
macro's ``else``: the ``<< message`` after it, and for an ``ASSERT_`` the
``return`` out of the function.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

from .exceptions import caught, internal
from .state import cxx, flush, report, skip
from .values import almost_equal, equal, show, text_of

__all__ = [
    "IsTrue",
    "IsFalse",
    "Compare",
    "Near",
    "Throws",
    "NoThrow",
    "AnyThrow",
    "That",
    "Fail",
    "Skip",
    "Death",
]


def IsTrue(value: Any, text: str, file: str, line: int, fatal: bool) -> bool:
    """``EXPECT_TRUE(c)``."""
    return _truth(bool(value), True, text, file, line, fatal)


def IsFalse(value: Any, text: str, file: str, line: int, fatal: bool) -> bool:
    """``EXPECT_FALSE(c)``."""
    return _truth(bool(value), False, text, file, line, fatal)


def _truth(value: bool, wanted: bool, text: str, file: str, line: int, fatal: bool) -> bool:
    flush()
    if value is wanted:
        return True
    said = f"Value of: {text}\n  Actual: {show(value)}\nExpected: {show(wanted)}"
    return report(file, line, said, fatal)


def _equality(left: str, right: str, first: str, second: str, extra: str = "") -> str:
    """GoogleTest's ``Expected equality of these values:``, each side's value if not its text."""
    lines = ["Expected equality of these values:"]
    for text, value in ((left, first), (right, second)):
        lines.append(f"  {text}")
        if value != text:
            lines.append(f"    Which is: {value}")
    return "\n".join(lines) + extra


def _equal(left: str, right: str, a: Any, b: Any) -> str:
    return _equality(left, right, show(a), show(b))


def _strings(left: str, right: str, a: Any, b: Any) -> str:
    return _equality(left, right, show(text_of(a)), show(text_of(b)))


def _caseless(left: str, right: str, a: Any, b: Any) -> str:
    return _strings(left, right, a, b) + "\nIgnoring case"


def _precise(digits: int) -> Callable[[str, str, Any, Any], str]:
    """The ``FLOAT_EQ``/``DOUBLE_EQ`` failure: each value to the digits that tell them apart."""

    def said(left: str, right: str, a: Any, b: Any) -> str:
        return _equality(left, right, f"{float(a):.{digits}g}", f"{float(b):.{digits}g}")

    return said


def _relation(symbol: str, suffix: str = "") -> Callable[[str, str, Any, Any], str]:
    """The failure of ``EXPECT_LT`` and kin: ``Expected: (a) < (b), actual: 2 vs 1``."""

    def said(left: str, right: str, a: Any, b: Any) -> str:
        return f"Expected: ({left}) {symbol} ({right}){suffix}, actual: {show(a)} vs {show(b)}"

    return said


def _streq(a: Any, b: Any) -> bool:
    return text_of(a) == text_of(b)


def _strcaseeq(a: Any, b: Any) -> bool:
    first, second = text_of(a), text_of(b)
    if first is None or second is None:
        return first is second
    return first.lower() == second.lower()


def _strings_relation(symbol: str, suffix: str = "") -> Callable[[str, str, Any, Any], str]:
    relation = _relation(symbol, suffix)

    def said(left: str, right: str, a: Any, b: Any) -> str:
        return relation(left, right, text_of(a), text_of(b))

    return said


#: Each comparison: whether it holds of two values, and its failure's words.
COMPARISONS: dict[str, tuple[Callable[[Any, Any], bool], Callable[[str, str, Any, Any], str]]] = {
    "EQ": (equal, _equal),
    "NE": (lambda a, b: not equal(a, b), _relation("!=")),
    "LT": (operator.lt, _relation("<")),
    "LE": (operator.le, _relation("<=")),
    "GT": (operator.gt, _relation(">")),
    "GE": (operator.ge, _relation(">=")),
    "STREQ": (_streq, _strings),
    "STRNE": (lambda a, b: not _streq(a, b), _strings_relation("!=")),
    "STRCASEEQ": (_strcaseeq, _caseless),
    "STRCASENE": (lambda a, b: not _strcaseeq(a, b), _strings_relation("!=", " (ignoring case)")),
    "FLOAT_EQ": (lambda a, b: almost_equal(a, b, True), _precise(9)),
    "DOUBLE_EQ": (lambda a, b: almost_equal(a, b, False), _precise(17)),
}


def Compare(kind: str, a: Any, b: Any, left: str, right: str, file: str, line: int,
            fatal: bool) -> bool:  # fmt: skip
    """``EXPECT_<kind>(a, b)``: ``EQ``, ``LT``, ``STREQ``, ``DOUBLE_EQ`` and the rest."""
    flush()
    holds, words = COMPARISONS[kind]
    if bool(holds(a, b)):
        return True
    return report(file, line, words(left, right, a, b), fatal)


def Near(a: Any, b: Any, error: Any, left: str, right: str, bound: str, file: str, line: int,
         fatal: bool) -> bool:  # fmt: skip
    """``EXPECT_NEAR(a, b, error)``: ``|a - b| <= error``, NaN never near anything."""
    flush()
    difference = abs(float(a) - float(b))
    if difference <= float(error):
        return True
    said = (
        f"The difference between {left} and {right} is {cxx(difference)}, which exceeds "
        f"{bound}, where\n{left} evaluates to {cxx(float(a))},\n{right} evaluates to "
        f"{cxx(float(b))}, and\n{bound} evaluates to {cxx(float(error))}."
    )
    return report(file, line, said, fatal)


def _thrown(statement: Callable[[], Any]) -> BaseException | None:
    """What running ``statement`` throws - ``None`` for nothing - letting xrdroot's gaps through."""
    try:
        statement()
    except BaseException as why:
        if internal(why):
            raise
        return why
    return None


def Throws(statement: Callable[[], Any], kind: str, text: str, file: str, line: int,
           fatal: bool) -> bool:  # fmt: skip
    """``EXPECT_THROW(statement, kind)``: the statement throws what a ``catch (kind &)`` takes."""
    flush()
    why = _thrown(statement)
    if why is not None and caught(why, kind):
        return True
    actual = "it throws nothing." if why is None else (
        f'it throws {type(why).__name__} with description "{why}".'
    )
    said = f"Expected: {text} throws an exception of type {kind}.\n  Actual: {actual}"
    return report(file, line, said, fatal)


def NoThrow(statement: Callable[[], Any], text: str, file: str, line: int, fatal: bool) -> bool:
    """``EXPECT_NO_THROW(statement)``."""
    flush()
    why = _thrown(statement)
    if why is None:
        return True
    said = (
        f"Expected: {text} doesn't throw an exception.\n"
        f'  Actual: it throws {type(why).__name__} with description "{why}".'
    )
    return report(file, line, said, fatal)


def AnyThrow(statement: Callable[[], Any], text: str, file: str, line: int, fatal: bool) -> bool:
    """``EXPECT_ANY_THROW(statement)``."""
    flush()
    if _thrown(statement) is not None:
        return True
    said = f"Expected: {text} throws an exception.\n  Actual: it doesn't."
    return report(file, line, said, fatal)


def That(value: Any, matcher: Any, text: str, spelled: str, file: str, line: int,
         fatal: bool) -> bool:  # fmt: skip
    """``EXPECT_THAT(value, matcher)``: a gmock matcher, or a plain value it must equal."""
    from .matchers import Eq, Matcher

    flush()
    rule = matcher if isinstance(matcher, Matcher) else Eq(matcher)
    if rule.matches(value):
        return True
    why = f", {rule.explanation}" if rule.explanation else ""
    said = f"Value of: {text}\nExpected: {rule.describe()}\n  Actual: {show(value)}{why}"
    return report(file, line, said, fatal)


def Fail(file: str, line: int, fatal: bool) -> bool:
    """``FAIL()`` and ``ADD_FAILURE()``."""
    return report(file, line, "Failed", fatal)


def Skip(file: str, line: int) -> bool:
    """``GTEST_SKIP()``: the test ends, skipped, with the message streamed after it."""
    return skip(file, line)


def Death(text: str, file: str, line: int) -> bool:
    """``EXPECT_DEATH(statement, regex)``: not run, and the test skipped, saying why.

    GoogleTest runs the statement in a child process and watches it die;
    xrdroot runs every test in one process, which the statement would end.
    """
    why = (
        f"The death test of {text} is not run: xrdroot runs every test in one process, "
        "which a statement meant to end its process would end."
    )
    return skip(file, line, why)
