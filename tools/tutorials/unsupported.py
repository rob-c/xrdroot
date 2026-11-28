"""What a run refused, grouped by cause: the worklist for getting the refusals to zero.

A refusal says xrdroot does not carry the thing at all; a wrong answer says it carries it and
is wrong. Only the first is a gap that a missing name closes, so this module turns a run's
``results.json`` into those gaps, biggest first, each with the tutorials it blocks.

    python -m tools.tutorials unsupported RUN_DIR/results.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The status a refusal is recorded under, and the only one this report counts.
REFUSED = "UNSUPPORTED"

#: The line that carries the reason: the exception the run ended on, or a sentence the
#: library refused with. The frames around it name files and source lines instead.
_RAISED = re.compile(r"\b(?:AttributeError|UnsupportedFeatureError|NotImplementedError): ")
_MARKERS = ("does not yet", "missing ROOT", "not supported", "cannot translate")

#: What a cause is compared by: the shapes that differ from run to run are numbers and
#: addresses, and two refusals of one thing must group together.
_NOISE = (
    (re.compile(r"0x[0-9a-fA-F]+"), "0x<addr>"),
    (re.compile(r"\bline \d+"), "line <n>"),
    (re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?"), "<n>"),
)


def _first(lines: list[str], matches: Any) -> str:
    """The first line ``matches`` accepts, or the empty string when none does."""
    return next((line for line in lines if matches(line)), "")


def _marked(line: str) -> bool:
    return any(word in line for word in _MARKERS)


def cause(record: Mapping[str, Any]) -> str:
    """The line that says why a tutorial was refused, with its numbers masked.

    A refusal reaches the run as a traceback, whose frames name files and source lines; the
    reason is the exception line, or the sentence the library refused in. Taken in that order
    so that two refusals of one thing group together.
    """
    text = "\n".join(record.get("details") or ()) or str(record.get("reason") or "")
    lines = [_masked(line) for line in text.splitlines() if line.strip()]
    chosen = _first(lines, _RAISED.search) or _first(lines, _marked)
    return (chosen or (lines[0] if lines else str(record.get("status") or "")))[:160]


def _masked(line: str) -> str:
    """``line`` without the numbers and addresses that change from run to run."""
    for pattern, replacement in _NOISE:
        line = pattern.sub(replacement, line)
    return line.strip()


def causes(results: Mapping[str, Any]) -> list[tuple[str, list[str]]]:
    """Every refusal, its tutorials, biggest group first - ties alphabetical so a run repeats."""
    groups: dict[str, list[str]] = defaultdict(list)
    for record in results.get("tutorials") or ():
        if record.get("status") == REFUSED:
            groups[cause(record)].append(str(record.get("path", "")))
    return sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))


def report(results: Mapping[str, Any]) -> str:
    """The refusals as a markdown table, so the campaign has one list to work down."""
    found = causes(results)
    total = sum(len(paths) for _, paths in found)
    lines = [
        f"# {total} refusals in {len(found)} causes",
        "",
        "| tutorials | cause | first tutorial |",
        "|---|---|---|",
    ]
    lines += [f"| {len(paths)} | `{why}` | {sorted(paths)[0]} |" for why, paths in found]
    return "\n".join(lines) + "\n"


def read(path: Path) -> dict[str, Any]:
    """A run's ``results.json``, which is where every number above comes from."""
    return json.loads(Path(path).read_text())


def run(args: Any) -> int:
    """Print the report for the run whose ``results.json`` the command was given."""
    print(report(read(args.results)), end="")
    return 0
