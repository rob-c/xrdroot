"""Comparing output with a ``.ref`` file the way roottest's ``custom_diff.py`` does.

roottest's driver diffs with ``roottest/scripts/custom_diff.py``: both sides
lose the lines that differ between runs for reasons nobody tests
(``Processing x.C...``, ACLiC's ``creating shared library``, blank lines),
pointers and ``input_line_N`` are canonicalised, and then the two are equal
when they are equal with every space removed. Only when they differ is a
unified diff of the filtered (space-kept) lines made, to show.

This reimplements that filter - rather than importing it from ROOT's tree,
which this harness only reads - including its one quirk: ``^[:space:]*$``
is a Python character class of ``:``, ``s``, ``p``, ``a``, ``c`` and ``e``,
so it drops empty lines and lines made only of those letters, but not lines
of spaces. Tests outside roottest use plain ``diff``: :func:`exact`.
"""

from __future__ import annotations

import difflib
import re

__all__ = ["filtered", "custom_diff", "exact", "unexpected_errors"]

#: Lines custom_diff.py drops wherever they are.
_SKIPPED = [
    re.compile(pattern)
    for pattern in (
        r"^Processing ",
        r"^Info in <\w+::ACLiC>: creating shared library",
        r"^In file included from input_line",
        r"^[:space:]*$",
        r"^Info in <TROOT>: Object auto registration",
        r"^<frozen importlib\._bootstrap>:\d+: RuntimeWarning: The global interpreter lock "
        r"\(GIL\) has been enabled",
    )
]

#: Rewrites custom_diff.py applies to every line it keeps.
_SUBSTITUTIONS = [
    (re.compile(r"[ ]@0x[a-fA-F0-9]+"), ""),
    (re.compile(r"std::__[0-9]+::"), "std::"),
    (re.compile(r"^(\S*/|)([^:/]+)[-:0-9]*(?=: error:)"), r"\2"),
    (re.compile(r"(input_line_|ROOT_prompt_)[0-9]+"), r"\1--"),
]

#: Standard-error lines the driver forgives a test that checks only its error stream.
_EXPECTED_ERRORS = (
    re.compile(r"^Info in <.+::ACLiC>: creating shared library.+"),
    re.compile(r"^Info in <TROOT>: Object auto registration.*"),
)


def filtered(lines: list[str], ignore_spaces: bool) -> list[str]:
    """custom_diff.py's ``LineFilter.filter``: skipped lines gone, the rest canonicalised."""
    kept = []
    for line in lines:
        if any(pattern.match(line) for pattern in _SKIPPED):
            continue
        for pattern, replacement in _SUBSTITUTIONS:
            line = pattern.sub(replacement, line)
        kept.append(line.replace(" ", "") if ignore_spaces else line)
    return kept


def custom_diff(reference: str, output: str) -> str:
    """The diff custom_diff.py would print - empty when the two count as the same."""
    ref, out = reference.splitlines(keepends=True), output.splitlines(keepends=True)
    if filtered(ref, True) == filtered(out, True):  # custom_diff.py's empty context_diff
        return ""
    lines = difflib.unified_diff(filtered(ref, False), filtered(out, False), "reference", "output")
    return "".join(lines)


def exact(reference: str, output: str) -> str:
    """Plain ``diff``: any difference at all, shown unified."""
    if reference == output:
        return ""
    ref, out = reference.splitlines(keepends=True), output.splitlines(keepends=True)
    return "".join(difflib.unified_diff(ref, out, "reference", "output"))


def unexpected_errors(stderr: str) -> list[str]:
    """What an ``assert*`` macro wrote to standard error beyond what the driver forgives."""
    lines = stderr.strip().split("\n") if stderr.strip() else []
    return [line for line in lines if not any(p.match(line) for p in _EXPECTED_ERRORS)]
