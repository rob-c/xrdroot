"""ROOT's ``TestSupport``: checking the diagnostics a statement prints, ``ROOT_EXPECT_ERROR``.

ROOT's tests install an error handler that sees each ``Error(where, ...)``
as it is made. xrdroot's diagnostics are lines on standard error - ``Error
in <TFile::TFile>: file x.root does not exist`` - written from wherever the
library meets the trouble, so while a :class:`CheckDiagsRAII` lives,
standard error is read line by line: a diagnostic one of the live checks
expects is taken, and is not printed, as ROOT's handler takes it; any other
line goes through. A warning or worse that no check expects is a failure,
as in ROOT; an unexpected ``Info`` is printed and let be.

ROOT's check ends with its scope, and the translation has none: one made
by ``ROOT_EXPECT_ERROR`` ends with the statement it wraps, and one the test
declares ends with the test.
"""

from __future__ import annotations

import io
import re
import sys
from dataclasses import dataclass
from typing import Any, TextIO

from ...errors import ROOTError, UnsupportedFeatureError
from ...pycxx.runtime import gtest
from ...pycxx.runtime.gtest import Namespace
from ...pycxx.runtime.gtest.exceptions import adopt
from ...pycxx.runtime.gtest.runner import AFTER_TEST
from ...pycxx.runtime.gtest.state import report
from .fileraii import FileRaii

__all__ = ["CheckDiagsRAII", "ExpectDiag", "ExpectNoDiag", "close_all", "LEVELS", "TestSupport"]

#: A ROOT diagnostic as a line: ``Error in <TFile::TFile>: file x.root does not exist``.
DIAGNOSTIC = re.compile(r"^(Print|Info|Warning|Error|Break|SysError|Fatal) in <(.*?)>: (.*)$")

#: The levels of ROOT's messages, ``kInfo`` to ``kFatal``, by the word a line starts with.
LEVELS = {
    "Print": 0,
    "Info": 1000,
    "Warning": 2000,
    "Error": 3000,
    "Break": 4000,
    "SysError": 5000,
    "Fatal": 6000,
}
#: The words, by level.
WORDS = {level: word for word, level in LEVELS.items()}
#: The least level a diagnostic must have for nobody expecting it to fail the test.
WARNING = 2000


@dataclass
class Diagnostic:
    """One diagnostic a check expects: its level, where it is made, and what it says."""

    level: int
    where: str
    message: str
    full: bool = True
    optional: bool = False
    seen: int = 0

    def matches(self, level: int, where: str, message: str) -> bool:
        if level != self.level or where != self.where:
            return False
        return message == self.message if self.full else self.message in message

    def __str__(self) -> str:
        return f"{WORDS.get(self.level, self.level)} in <{self.where}>: {self.message}"


#: The checks alive, innermost last; standard error is read while there is one.
ACTIVE: list[CheckDiagsRAII] = []


class CheckDiagsRAII:
    """``ROOT::TestSupport::CheckDiagsRAII``: the diagnostics a stretch of a test must print."""

    def __init__(
        self, level: Any = None, where: Any = "", message: Any = "", full: Any = True
    ) -> None:
        self.expected: list[Diagnostic] = []
        self.unexpected: list[str] = []
        self.closed = False
        if level is not None:
            self.requiredDiag(level, where, message, full)
        _listen(self)

    def requiredDiag(self, level: Any, where: Any, message: Any, full: Any = True) -> None:
        """A diagnostic that must be printed before the check ends."""
        self.expected.append(Diagnostic(int(level), str(where), str(message), bool(full)))

    def optionalDiag(self, level: Any, where: Any, message: Any, full: Any = True) -> None:
        """A diagnostic that may be printed, and is taken if it is."""
        self.expected.append(Diagnostic(int(level), str(where), str(message), bool(full), True))

    def take(self, level: int, where: str, message: str) -> bool:
        """Is this diagnostic one the check expects? Counted as seen if so."""
        for diagnostic in self.expected:
            if diagnostic.matches(level, where, message):
                diagnostic.seen += 1
                return True
        return False

    def close(self, file: Any = "unknown file", line: Any = None) -> None:
        """The check's end: a failure for each diagnostic missing and each one unexpected."""
        if self.closed:
            return
        self.closed = True
        _stop(self)
        intro = "ROOT::TestSupport::CheckDiagsRAII: "
        for diagnostic in self.expected:
            if not diagnostic.optional and not diagnostic.seen:
                report(file, line, f"{intro}Expected diagnostic message missing:\n{diagnostic}", 0)
        for line_text in self.unexpected:
            report(file, line, f"{intro}Unexpected diagnostic message received:\n{line_text}", 0)


class _Reader(io.TextIOBase):
    """Standard error while a check lives: each whole line offered to the checks first."""

    def __init__(self, target: TextIO) -> None:
        super().__init__()
        self.target = target
        self.partial = ""

    def write(self, text: str) -> int:
        *lines, self.partial = (self.partial + text).split("\n")
        for line in lines:
            if not _offered(line):
                self.target.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        self.target.flush()

    def finish(self) -> TextIO:
        """What was standard error before, with any unended line written to it."""
        if self.partial:
            self.target.write(self.partial)
        return self.target


def _offered(line: str) -> bool:
    """Was ``line`` a diagnostic a live check took? An unexpected warning is noted as one."""
    found = DIAGNOSTIC.match(line)
    if found is None or not ACTIVE:  # not ROOT's, or read after every check has ended
        return False
    level, where, message = LEVELS[found[1]], found[2], found[3]
    if any(check.take(level, where, message) for check in reversed(ACTIVE)):
        return True
    if level >= WARNING:
        ACTIVE[-1].unexpected.append(line)
    return False


def _listen(check: CheckDiagsRAII) -> None:
    if not isinstance(sys.stderr, _Reader):
        sys.stderr = _Reader(sys.stderr)
    ACTIVE.append(check)


def _stop(check: CheckDiagsRAII) -> None:
    ACTIVE.remove(check)
    if not ACTIVE and isinstance(sys.stderr, _Reader):
        sys.stderr = sys.stderr.finish()


def close_all() -> None:
    """End every check still alive, innermost first: the test they were made in has ended."""
    for check in reversed(list(ACTIVE)):
        check.close()


def ExpectDiag(statement: Any, level: Any, where: Any, message: Any, full: Any, file: Any,
               line: Any) -> None:  # fmt: skip
    """``ROOT_EXPECT_ERROR(statement, where, message)`` and kin: the statement prints it."""
    check = CheckDiagsRAII(level, where, message, full)
    try:
        statement()
    finally:
        check.close(file, line)


def ExpectNoDiag(statement: Any, file: Any, line: Any) -> None:
    """``ROOT_EXPECT_NODIAG(statement)``: the statement prints no warning or worse."""
    check = CheckDiagsRAII()
    try:
        statement()
    finally:
        check.close(file, line)


#: ``ROOT::TestSupport``: the check of the diagnostics a test prints, and the guard of a
#: scratch file (:mod:`.fileraii`).
TestSupport = Namespace("ROOT::TestSupport",
                        {"CheckDiagsRAII": CheckDiagsRAII, "FileRaii": FileRaii})


def _adopted() -> None:
    """ROOT's unit tests as GoogleTest runs them here: ``ROOT_EXPECT_ERROR``'s checks in
    ``testing::internal``, every check ended with its test, and ROOT's exceptions told apart -
    xrdroot's refusal a gap no test expects, ROOT's errors what ``std::runtime_error`` and
    ``ROOT::RException`` catch."""
    gtest.provide({"ExpectDiag": ExpectDiag, "ExpectNoDiag": ExpectNoDiag})
    AFTER_TEST.append(close_all)
    adopt(
        gaps=(UnsupportedFeatureError,),
        errors=(ROOTError,),
        caught={"ROOT::RException": "library"},
        refused=UnsupportedFeatureError,
    )


_adopted()
