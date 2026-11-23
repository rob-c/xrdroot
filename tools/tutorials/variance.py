"""Measuring whether ROOT's own run of a tutorial is the same twice.

A ``.py`` tutorial that hands RooFit a Python ``set`` - ``RooDataSet("ds", "ds", {x, y},
Import=tree)`` - has as many outputs as the set has orders, and PyROOT picks one by where
the process happens to put the objects: measured on ``rf102_dataimport.py``, ROOT prints
``1) x = 0.0174204`` seven times in ten and ``1) x = 9.46654`` three times, and the PNG it
draws differs with them. Nothing this tree does can make such a tutorial the same twice, so
the harness cannot promise it: a tutorial whose own oracle is not reproducible is recorded
as ``FLAKY`` (``harness.FLAKY``) and the ratchet treats it as neither a gain nor a loss.

This module is the measurement behind that record: repeat ROOT's run, compare each with the
first, and say in one sentence how they differ.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .catalog import Tutorial
from .compare import normalise
from .environment import Oracle
from .harness import FLAKY, Settings, oracle_command, oracle_environment
from .runner import RunResult, Sandbox, run_in_sandbox

__all__ = ["distinct", "distinctive_line", "same_output", "tutorial_named", "varying"]

#: How much of a contrasting line the recorded sentence quotes.
QUOTE = 60


def same_output(one: RunResult, other: RunResult) -> bool:
    """Whether two runs of one tutorial produced the same thing, in whatever order.

    A ``.py`` tutorial's ``print`` and RooFit's message stream are two streams into one
    file, so even two unchanged runs interleave them differently; what a draw *is* is which
    lines it has, so the streams are compared as sorted normalised lines, and the outputs'
    checksums, so a run that drew a different picture is never the same draw.
    """
    if one.exit_code != other.exit_code or one.files != other.files:
        return False
    theirs = sorted(normalise(other.stdout)), sorted(normalise(other.stderr))
    mine = sorted(normalise(one.stdout)), sorted(normalise(one.stderr))
    return mine == theirs


def distinctive_line(one: RunResult, other: RunResult) -> str:
    """How one run's output differs from another's, in one sentence.

    A draw differs from another by *which lines* it has, so the sentence quotes the first
    line one has and the other does not, and the first the other has and this one does not -
    for a tutorial whose order varies, that is the observable it lists where the other names
    the other one.
    """
    mine, theirs = normalise(one.stdout), normalise(other.stdout)
    extra = Counter(mine) - Counter(theirs)
    missing = Counter(theirs) - Counter(mine)
    if not extra and not missing:
        return "a different file"
    head = next(iter(extra), None)
    absent = next(iter(missing), None)
    said = f"{head.strip()[:QUOTE]!r}" if head is not None else "no line of its own"
    return said + (f" where the other run has {absent.strip()[:QUOTE]!r}" if absent else "")


def _one(tutorial: Tutorial, settings: Settings, oracle: Oracle, keep: Path) -> RunResult:
    sandbox = Sandbox((settings.tutorials / tutorial.path).parent, (), keep / "files")
    env = oracle_environment(tutorial, oracle, settings.tutorials)
    return run_in_sandbox(sandbox, oracle_command(tutorial, oracle), env, settings.timeout)


def distinct(
    tutorial: Tutorial, settings: Settings, oracle: Oracle, runs: int, work: Path
) -> list[RunResult]:
    """ROOT's own runs of one tutorial, one per distinct draw, in the order they came."""
    found: list[RunResult] = []
    for number in range(1, max(1, runs) + 1):
        result = _one(tutorial, settings, oracle, work / tutorial.path / str(number))
        if not any(same_output(result, one) for one in found):
            found.append(result)
    return found


def varying(
    tutorial: Tutorial, settings: Settings, oracle: Oracle, runs: int, work: Path
) -> list[dict[str, Any]]:
    """The rows a tutorial's own variance is recorded as, or none when ROOT is repeatable.

    One row per draw after the first: what it has that the first does not, and what the first
    has that it does not - the sentence a reader needs to see that this tutorial is not the
    tree's to make repeatable.
    """
    found = distinct(tutorial, settings, oracle, runs, work)
    first, others = (found[0], found[1:]) if found else (None, [])
    if first is None or not others:
        return []
    return [
        {"path": tutorial.path, "draw": distinctive_line(one, first), "runs": runs}
        for one in others
    ]


def recorded(index: Path = FLAKY) -> dict[str, list[dict[str, Any]]]:
    """The tutorials already recorded as varying, by path."""
    try:
        found: dict[str, list[dict[str, Any]]] = json.loads(index.read_text()).get("varying", {})
    except (OSError, ValueError):
        return {}
    return found


def keep(rows: Sequence[dict[str, Any]], index: Path = FLAKY) -> int:
    """Add ``rows`` to the record, one per path, and say how many are new."""
    known = recorded(index)
    added = 0
    for row in rows:
        entries = known.setdefault(str(row["path"]), [])
        if any(one["draw"] == row["draw"] for one in entries):
            continue
        entries.append(dict(row))
        added += 1
    index.write_text(json.dumps({"varying": known}, indent=1) + "\n")
    return added


def tutorial_named(catalogue: dict[str, Tutorial], wanted: str) -> Tutorial:
    """The catalogue's tutorial called ``wanted`` - its path, or a unique end of it."""
    if wanted in catalogue:
        return catalogue[wanted]
    matches = [one for path, one in catalogue.items() if path.endswith(wanted)]
    if len(matches) != 1:
        raise KeyError(f"{wanted!r} names {len(matches)} tutorials; give a path the catalogue has")
    return matches[0]


def paths(catalogue: dict[str, Tutorial], wanted: Sequence[str]) -> list[Tutorial]:
    """The tutorials ``wanted`` names, each resolved against the catalogue."""
    return [tutorial_named(catalogue, one) for one in wanted]
