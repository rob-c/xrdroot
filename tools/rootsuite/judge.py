"""Judging one run the way ctest and ``RootTestDriver.cmake`` judge it.

The order is the driver's: a test that checks only standard error (an
``assert*`` macro) fails on any of it; then the exit code must be
``PASSRC`` (else 0); then an ``OUTCNVCMD`` command (run on the file holding
the output) and an ``OUTCNV`` script (fed it) rewrite the output; then the
output, and the error stream with ``ERRREF``, are diffed against their
references. ctest then applies ``FAIL_REGULAR_EXPRESSION`` (a match fails),
``PASS_REGULAR_EXPRESSION`` (which replaces the exit code: a match is the
pass) and ``WILL_FAIL`` (which turns pass and fail around).

Before any of that, two outcomes are xrdroot's rather than the test's: a
timeout, and a refusal - ``xrdroot run`` exiting 2 with its one-line
``xrdroot run: ...``, or an ``UnsupportedFeatureError`` traceback - which is
UNSUPPORTED, so that what xrdroot says it does not do is told apart from
what it gets wrong.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import refdiff
from .model import Test
from .process import Result

__all__ = ["Verdict", "judge", "refusal", "first_failure", "STATUSES"]

#: Every status a test can end with, in the order reports list them.
STATUSES = ("PASS", "FAIL", "DIFF", "UNSUPPORTED", "OUT-OF-SCOPE", "SKIP", "TIMEOUT")

#: ``xrdroot <command>: <why>``: a subcommand's refusal, printed as one line before exit 2.
_REFUSAL = re.compile(r"^xrdroot [a-z]+: (.+)$", re.M)
#: A traceback ending in xrdroot's own word for "not supported".
_UNSUPPORTED = re.compile(r"^(?:[\w.]+\.)?(?:UnsupportedFeatureError|Refusal): (.+)$", re.M)
#: argparse's complaint: the command line, not the test, is what ``xrdroot`` did not take.
_USAGE = re.compile(r"^xrdroot: error: (.+)$", re.M)
#: ``[Errno 2] No such file or directory: ...``: an ``OSError`` the CLI printed as its refusal.
_OS_ERROR = re.compile(r"\[Errno \d+\]")
#: ROOT's own Python modules that tests import beside ``ROOT``, which xrdroot does not provide,
#: and what each is.
ROOT_PYTHON_MODULES = {
    "cmdLineUtils": "the module ROOT's rootls, rootcp, rootmv and rootrm are written over, "
                    "whose functions the test calls itself; xrdroot implements those tools as "
                    "subcommands, not that module",
}  # fmt: skip
#: Python failing to import a module.
_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '([\w.]+)'")
#: ``file.C:7:1: error: ...``: an error ``xrdroot root`` reports as cling would.
_DIAGNOSTIC = re.compile(r"^(\S+): error: (.+)$", re.M)
#: What xrdroot says of what it does not do: its refusal, a ROOT name it does not have
#: yet, or C++ its translator does not turn into Python.
_SAYS_UNSUPPORTED = re.compile(
    r"^(?:[\w.]+\.)?(?:UnsupportedFeatureError|Refusal): |xrdroot\.pyroot does not yet|"
    r"this translator")


@dataclass
class Verdict:
    """A test's status, the sentence saying why, and any diff that shows it.

    ``cases`` counts the cases inside a gtest program or unittest file, and
    ``failures`` says how each failed case of a gtest program failed.
    """

    status: str
    reason: str = ""
    diff: str = ""
    cases: dict[str, int] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if value}


def _said(text: str) -> list[str]:
    """Errors in which xrdroot says it lacks something, or a test imports ROOT's own Python."""
    said = [f"{where}: {what}" for where, what in _DIAGNOSTIC.findall(text)
            if _SAYS_UNSUPPORTED.search(what)]
    return said + [f"It imports ROOT's Python module {name}, {ROOT_PYTHON_MODULES[name]}."
                   for name in _NO_MODULE.findall(text) if name in ROOT_PYTHON_MODULES]


def refusal(result: Result) -> str:
    """xrdroot's refusal in a run's output, or ``""``: exit 2 with its line, the error raised,
    or a cling-style error saying xrdroot does not do something."""
    text = result.stdout + "\n" + result.stderr
    found = _UNSUPPORTED.search(text)
    if found:
        return found.group(1).strip()
    said = _said(text)
    if said:
        return said[0]
    found = _REFUSAL.search(text)
    if result.exit_code != 2 or found is None or _USAGE.search(text):
        return ""
    message = found.group(1).strip()
    # A missing or unreadable file is the run failing, not xrdroot declining a feature.
    return "" if _OS_ERROR.search(message) else message


def first_failure(result: Result) -> str:
    """The line most likely to say why a run failed: a gtest failure, else the last error line."""
    text = result.stdout + "\n" + result.stderr
    usage = _USAGE.search(text)
    if usage:
        return f"xrdroot did not take the command line: {usage.group(1)}"
    failed = re.search(r"^\[  FAILED  \] (\S+)", text, re.M)
    if failed:
        return f"gtest case {failed.group(1)} failed"
    lines = [line.strip() for line in (result.stderr or result.stdout).splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def judge(test: Test, result: Result, root: Path, cwd: Path,
          post: Result | None = None) -> Verdict:
    """The status ctest would give ``test`` for ``result``, xrdroot's refusals told apart.

    ``post`` is how the ``POSTCMD`` ran, which the driver checks after the
    output is converted and before it is compared.
    """
    if result.timed_out:
        return Verdict("TIMEOUT", f"still running after {result.runtime:.0f} s")
    refused = refusal(result)
    if refused:
        return Verdict("UNSUPPORTED", refused)
    verdict = _driver(test, result, root, cwd, post)
    verdict = _regexes(test, result, verdict)
    return _inverted(verdict) if test.will_fail else verdict


def _exit(result: Result, expected: int) -> str:
    code = result.exit_code
    if code == expected:
        return ""
    if code is not None and code < 0:
        return f"killed by signal {-code}: {first_failure(result)}"
    return f"exit {code}, expected {expected}: {first_failure(result)}"


def _driver(test: Test, result: Result, root: Path, cwd: Path, post: Result | None) -> Verdict:
    unexpected = refdiff.unexpected_errors(result.stderr) if test.check_err and not test.check_out \
        else []
    if unexpected:
        return Verdict("FAIL", f"unexpected error output: {unexpected[0][:300]}")
    wrong = _exit(result, test.passrc if test.passrc is not None else 0)
    if wrong:
        return Verdict("FAIL", wrong)
    return _references(test, result, root, cwd, post)


def _references(test: Test, result: Result, root: Path, cwd: Path,
                post: Result | None) -> Verdict:
    """The output, converted by any ``OUTCNV``, diffed against ``OUTREF`` and ``ERRREF``.

    A ``POSTCMD`` that failed fails the test first, as in the driver. A test
    that does not check its standard output - an ``assert*`` macro - has the
    driver write nothing to the file its ``OUTREF`` is compared with.
    """
    output, errors = _streams(test, result, root, cwd)
    if post is not None and post.exit_code != 0:
        return Verdict("FAIL", f"the post-command {_exit(post, 0)}")
    for reference, text, stream in ((test.outref, output, "output"),
                                    (test.errref, errors, "error output")):
        diff = _compare(test, (root / reference).read_text(errors="replace"), text) \
            if reference else ""
        if diff:
            return Verdict("DIFF", f"standard {stream} differs from {reference}", diff)
    return Verdict("PASS")


def _streams(test: Test, result: Result, root: Path, cwd: Path) -> tuple[str, str]:
    """The output and error streams as compared: converted by ``OUTCNVCMD`` and ``OUTCNV``."""
    output, errors = (result.stdout if test.check_out else ""), result.stderr
    if test.outcnvcmd:
        output = _converted_by(test.outcnvcmd, output, cwd)
    if test.outcnv:
        output = _converted(root / test.outcnv, output, cwd)
        errors = _converted(root / test.outcnv, errors, cwd) if test.errref else errors
    return output, errors


def _compare(test: Test, reference: str, output: str) -> str:
    return (refdiff.custom_diff if test.custom_diff else refdiff.exact)(reference, output)


def _converted(script: Path, text: str, cwd: Path) -> str:
    """``OUTCNV``: the output piped through ``sh <script>``, as the driver does."""
    try:
        done = subprocess.run(["sh", str(script)], input=text.encode(), cwd=str(cwd),
                              capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return text
    return done.stdout.decode("utf-8", "replace")


def _converted_by(command: tuple[str, ...], text: str, cwd: Path) -> str:
    """``OUTCNVCMD``: the command run on the file holding the output, its own output kept."""
    with tempfile.NamedTemporaryFile("w", suffix=".log", dir=cwd, delete=False) as stream:
        stream.write(text)
    try:
        done = subprocess.run([*command, stream.name], cwd=str(cwd), capture_output=True,
                              timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return text
    finally:
        Path(stream.name).unlink(missing_ok=True)
    return done.stdout.decode("utf-8", "replace")


def _regexes(test: Test, result: Result, verdict: Verdict) -> Verdict:
    """ctest's regular expressions over everything the test printed."""
    text = result.stdout + result.stderr
    for pattern in test.failregex:
        found = _search(pattern, text)
        if found:
            return Verdict("FAIL", f"output matches the FAILREGEX {pattern!r}: {found[:200]}")
    if test.passregex:
        if any(_search(pattern, text) for pattern in test.passregex):
            return Verdict("PASS", verdict.reason)
        return Verdict("FAIL", f"output matches no PASSREGEX {list(test.passregex)!r}")
    return verdict


def _search(pattern: str, text: str) -> str:
    try:
        found = re.search(pattern, text, re.M)
    except re.error:
        found = re.search(re.escape(pattern), text)
    return found.group(0) or pattern if found else ""


def _inverted(verdict: Verdict) -> Verdict:
    """``WILLFAIL``: a failure (or difference) passes, and a pass fails."""
    if verdict.status == "PASS":
        return Verdict("FAIL", "passed, where CMake declares it WILLFAIL")
    return Verdict("PASS", f"failed as WILLFAIL expects: {verdict.reason}")
