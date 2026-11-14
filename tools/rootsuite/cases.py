"""The cases inside one test: pytest's junit XML for a unittest file, gtest's summary lines.

ctest judges a whole file or program; the cases inside say how far from
passing it is. A unittest file is judged here from its junit XML - passed,
failed, errored and skipped cases, and the first failure's message - and a
whole file whose every failure is xrdroot refusing is UNSUPPORTED rather
than FAIL.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ElementTree
from pathlib import Path

from .judge import Verdict, refusal
from .model import Test
from .process import Result

__all__ = ["junit", "gtest_counts", "gtest_failures", "judge_pyunittest"]

#: pytest's exit status when it collected nothing to run.
NO_TESTS = 5


def junit(path: Path) -> tuple[dict[str, int], list[str]]:
    """Case counts from a junit XML file, and each failure's or error's first line."""
    try:
        tree = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError):
        return {}, []
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    messages: list[str] = []
    for case in tree.iter("testcase"):
        outcome = _outcome(case)
        counts[outcome] += 1
        problem = case.find("failure") if outcome == "failed" else case.find("error")
        if problem is not None:
            messages.append(_message(case, problem))
    return counts, messages


def _outcome(case: ElementTree.Element) -> str:
    for tag, outcome in (("failure", "failed"), ("error", "errors"), ("skipped", "skipped")):
        if case.find(tag) is not None:
            return outcome
    return "passed"


def _message(case: ElementTree.Element, problem: ElementTree.Element) -> str:
    text = (problem.get("message") or problem.text or "").strip().splitlines()
    return f"{case.get('name', '?')}: {text[0][:300] if text else '?'}"


def gtest_counts(output: str) -> dict[str, int]:
    """gtest's closing summary: how many cases passed and failed."""
    counts = {}
    for label, key in (("PASSED", "passed"), ("FAILED", "failed")):
        found = re.search(rf"^\[  {label}  \] (\d+) tests?", output, re.M)
        if found:
            counts[key] = int(found.group(1))
    return counts


#: One case's block of gtest output: its ``[ RUN      ]`` line to the next one.
_CASE = re.compile(r"^\[ RUN      \] (\S+)\n(.*?)(?=^\[ RUN      \] |\Z)", re.M | re.S)
#: The line a failed assertion or an exception starts with: ``file.cxx:12: Failure``.
_FAILURE = re.compile(r"^(?:.*: )?Failure$")


def gtest_failures(output: str) -> list[str]:
    """Each failed gtest case and the first line saying why: ``Suite.Case: message``.

    The program's verdict names only its first failed case; these say how
    each case failed, so the causes inside programs can be counted.
    """
    failures = []
    for name, body in _CASE.findall(output):
        if re.search(rf"^\[  FAILED  \] {re.escape(name)}\b", body, re.M):
            failures.append(f"{name}: {_why(body.splitlines())[:300]}")
    return failures


def _why(lines: list[str]) -> str:
    """The line after a case's first ``Failure`` (and the next, if that one ends in a colon),
    else its first line that is not gtest's."""
    said = [line for line in lines if line.strip() and not line.startswith("[")]
    for at, line in enumerate(lines[:-1]):
        if _FAILURE.match(line) and lines[at + 1].strip():
            first = lines[at + 1].strip()
            # ``Expected equality of these values:`` says little without what was compared
            more = lines[at + 2].strip() if first.endswith(":") and at + 2 < len(lines) else ""
            return f"{first} {more}".strip()
    return said[0].strip() if said else "?"


def judge_pyunittest(test: Test, result: Result, junit_path: Path) -> Verdict:
    """A unittest file: PASS when pytest passes all of it, UNSUPPORTED when only refusals fail."""
    counts, messages = junit(junit_path)
    if result.timed_out:
        return Verdict("TIMEOUT", f"still running after {result.runtime:.0f} s", cases=counts)
    verdict = _verdict(result, counts, messages)
    verdict.cases = counts
    if test.will_fail and verdict.status in ("PASS", "FAIL"):
        verdict.status = "FAIL" if verdict.status == "PASS" else "PASS"
    return verdict


def _verdict(result: Result, counts: dict[str, int], messages: list[str]) -> Verdict:
    if result.exit_code == 0:
        return Verdict("PASS")
    if result.exit_code == NO_TESTS:
        return Verdict("FAIL", "pytest collected no tests from the file")
    unsupported = ("UnsupportedFeatureError", "Refusal")
    if messages and all(any(word in m for word in unsupported) for m in messages):
        return Verdict("UNSUPPORTED", messages[0])
    if not messages:
        return Verdict("UNSUPPORTED", refusal(result)) if refusal(result) else Verdict(
            "FAIL", f"pytest exited {result.exit_code}: {_last_line(result)}")
    total = sum(counts.values())
    bad = counts.get("failed", 0) + counts.get("errors", 0)
    return Verdict("FAIL", f"{bad} of {total} cases failed; first {messages[0]}")


def _last_line(result: Result) -> str:
    lines = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""
