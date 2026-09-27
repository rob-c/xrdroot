"""What became of a tutorial, in one word, and why, in one line that groups well.

The six outcomes:

``PASS``
    xrdroot ran it and said and wrote what ROOT did.
``DIFF``
    it ran, and some output differs - the first difference is the reason.
``FAIL``
    xrdroot crashed or timed out: the exception's type and message.
``UNSUPPORTED``
    xrdroot refused by name - ``ROOT has X; xrdroot.pyroot does not yet``,
    an ``UnsupportedFeatureError``, a ``NotImplementedError``, the translator
    turning a construct down.
``SKIP``
    ROOT itself does not run it here: a GUI, the network, a build option or a
    Python package this machine lacks, or no test in ROOT's CMakeLists.
``ORACLE-FAIL``
    ROOT ran it and failed by its own CI's rules, so there is nothing to match.

Reasons are made to be counted: the parts of a message that differ between
tutorials hitting the same gap (numbers, paths, line numbers) are masked, and
the common shapes - a missing ROOT name, a missing method - are rewritten as
``missing ROOT.TLorentzVector.Boost``, so the report can rank what to build
next by how many tutorials each gap blocks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NamedTuple

from .runner import RunResult

__all__ = [
    "STATUSES",
    "Verdict",
    "failure",
    "oracle_failure",
    "reason_of",
    "last_exception",
    "plain",
    "NO_RUN",
]

#: Every status, in the order reports list them.
STATUSES = ("PASS", "DIFF", "FAIL", "UNSUPPORTED", "SKIP", "ORACLE-FAIL")

#: The reason given when the xrdroot under test has no ``run`` subcommand.
NO_RUN = "not runnable: xrdroot has no `run`"

#: The longest reason kept, so a runaway message does not become a category.
REASON_LENGTH = 160


class Verdict(NamedTuple):
    status: str
    reason: str = ""


#: The final line of a Python traceback: a dotted exception name, perhaps a message.
_EXCEPTION = re.compile(r"^(?P<type>[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)|"
                        r"[A-Za-z_][\w.]*\.[A-Z]\w*)(?::\s?(?P<message>.*))?$")  # fmt: skip

#: The one-line refusal ``xrdroot <command>: <why>`` the CLI prints instead of a traceback.
_CLI_REFUSAL = re.compile(r"^xrdroot(?: \w+)?: (?P<message>.+)$")

#: The pyroot namespace's own refusal of a name it does not have yet.
_MISSING_NAME = re.compile(r"ROOT has (?P<name>[\w.:]+); xrdroot\.pyroot does not yet")

#: Python's missing-attribute messages, on an instance, a class or a module.
_MISSING_ATTRIBUTE = re.compile(
    r"^(?:'(?P<obj>[\w.]+)' object|type object '(?P<cls>[\w.]+)'|module '(?P<mod>[\w.]+)')"
    r" has no attribute '(?P<attr>\w+)'"
)

#: Exception names, or words in a refusal, that mean "refused by name".
_REFUSING = re.compile(r"Unsupported|NotImplemented|Refus|NotSupported|does not yet|not supported"
                       r"|cannot translate|not yet", re.I)  # fmt: skip


#: A terminal escape sequence: the colours Python 3.13 and later put in a traceback
#: when ``FORCE_COLOR`` or a terminal asks for them.
_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def plain(text: str) -> str:
    """``text`` without its terminal colours, so a coloured traceback reads as a plain one."""
    return _ESCAPE.sub("", text)


def last_exception(stderr: str) -> tuple[str, str] | None:
    """The exception a traceback ends with, as its type and its message - coloured or not."""
    lines = plain(stderr).splitlines()
    if not any(line.startswith("Traceback (most recent call last)") for line in lines):
        return None
    for line in reversed(lines):
        match = _EXCEPTION.match(line.strip())
        if match:
            return match.group("type"), match.group("message") or ""
    return None


#: ``from ROOT import X`` or ``import ROOT.X`` finding nothing, as Python words it.
_MISSING_IMPORT = re.compile(
    r"cannot import name '(?P<name>\w+)' from '(?:ROOT|xrdroot\.pyroot)'"
    r"|No module named '(?:ROOT|xrdroot\.pyroot)\.(?P<name2>[\w.]+)'"
)


def _masked(message: str) -> str:
    """The message with what varies between tutorials hitting one gap masked.

    Paths, line numbers and numbers go, and so do the C++ handed to
    ``ProcessLine`` or ``Declare``, which is each tutorial's own.
    """
    message = re.sub(r"\((['\"]).*?\1\)", "(...)", message)
    message = re.sub(r"(/[\w.@+-]+)+/?", "<path>", message)
    message = re.sub(r"\bline \d+\b", "line <n>", message)
    message = re.sub(r"(?<![\w.])[-+]?\d+(\.\d+)?(e[-+]?\d+)?\b", "<n>", message)
    return message.strip()[:REASON_LENGTH]


def _attribute(message: str) -> str | None:
    match = _MISSING_ATTRIBUTE.match(message)
    if not match:
        return None
    owner = match.group("obj") or match.group("cls") or match.group("mod")
    owner = re.sub(r"^(cppyy\.gbl|xrdroot\.pyroot(\.\w+)*|ROOT)(\.|$)", "", owner)
    return f"missing ROOT.{owner + '.' if owner else ''}{match.group('attr')}"


def reason_of(kind: str, message: str) -> Verdict:
    """An exception as a status and a reason that groups with its kind."""
    short = kind.rsplit(".", 1)[-1]
    name = _MISSING_NAME.search(message)
    if name:
        return Verdict("UNSUPPORTED", f"missing ROOT.{name.group('name')}")
    if short == "AttributeError":
        attribute = _attribute(message)
        if attribute:
            return Verdict("FAIL", attribute)
    if short in ("ModuleNotFoundError", "ImportError"):
        imported = _MISSING_IMPORT.search(message)
        if imported:
            name = imported.group("name") or imported.group("name2")
            return Verdict("FAIL", f"missing ROOT.{name}")
        return Verdict("FAIL", f"{short}: {_masked(message)}")
    status = "UNSUPPORTED" if _REFUSING.search(short) or _REFUSING.search(message) else "FAIL"
    return Verdict(status, f"{short}: {_masked(message)}" if message else short)


def _refusal_line(stderr: str) -> str | None:
    for line in reversed(plain(stderr).splitlines()):
        match = _CLI_REFUSAL.match(line.strip())
        if match:
            return match.group("message")
    return None


def _fallback(result: RunResult) -> Verdict:
    tail = [line.strip() for line in plain(result.stderr).splitlines() if line.strip()]
    said = _masked(tail[-1]) if tail else "no message"
    return Verdict("FAIL", f"exit {result.exit_code}: {said}")


def failure(result: RunResult) -> Verdict:
    """Why an xrdroot run did not finish as it should, from its exit and its stderr."""
    if result.timed_out:
        return Verdict("FAIL", "timeout")
    if result.exit_code is None:
        return Verdict("FAIL", f"did not start: {_masked(result.stderr)}")
    if result.exit_code < 0:
        return Verdict("FAIL", f"killed by signal {-result.exit_code}")
    found = last_exception(result.stderr)
    if found is not None:
        return reason_of(*found)
    refused = _refusal_line(result.stderr)
    if refused is not None:
        return reason_of("", refused) if _MISSING_NAME.search(refused) else _refused(refused)
    return _fallback(result)


def _refused(message: str) -> Verdict:
    status = "UNSUPPORTED" if _REFUSING.search(message) else "FAIL"
    return Verdict(status, _masked(message))


def matched_failure(text: str, patterns: Sequence[str]) -> str | None:
    """The first output line one of CI's failure patterns matches - CTest's FAILREGEX."""
    for pattern in patterns:
        regex = re.compile(pattern)
        for line in text.splitlines():
            if regex.search(line):
                return line.strip()
    return None


def oracle_failure(result: RunResult, passrc: int, failregex: Sequence[str]) -> str | None:
    """Why ROOT's run fails its CI's own rules, or None if it passes them."""
    if result.timed_out:
        return "ROOT timed out"
    if result.exit_code != passrc:
        return f"ROOT exited {result.exit_code}, CI expects {passrc}"
    line = matched_failure(result.stdout + "\n" + result.stderr, failregex)
    if line is not None:
        return f"ROOT printed {_masked(line)!r}"
    return None
