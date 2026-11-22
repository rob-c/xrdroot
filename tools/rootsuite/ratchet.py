"""A test that passed is a promise: no later run may lose it, even by leaving it out.

``baseline.json`` maps every test id of a full run to its status. A run is
checked against it by :func:`regressions`: every baseline PASS that is not
PASS now - or is missing - is a regression. A TIMEOUT can be the machine
rather than xrdroot, so :func:`recheck` reruns the timed-out tests alone,
one at a time, and only an actual PASS there excuses them.

The promise runs one way only: a PASS is worth recording the moment a run
sees it, so :func:`merge_baseline` raises the rows a subset run passed and
leaves every other row alone, while :func:`write_baseline` - the full run -
replaces the file outright.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .judge import STATUSES

__all__ = ["BASELINE", "FULL_KINDS", "judged", "merge_baseline", "regressions",
           "statuses", "write_baseline", "timeouts"]

#: Kept in the repository so checking a run never silently starts with no promises.
BASELINE = Path(__file__).with_name("baseline.json")

#: The kinds a baseline must cover: everything that runs without ROOT itself.
FULL_KINDS = ("gtest", "roottest", "pyunittest", "test")


def statuses(path: Path) -> dict[str, str]:
    """The status each test id has in the baseline at ``path``, as a committed run recorded it."""
    found: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return {str(name): str(status) for name, status in found.items()}


def judged(doc: Mapping[str, Any]) -> set[str]:
    """The kinds a run covers, which are the only ones its ratchet may judge.

    A run of one kind - ``--kind roottest`` while working on the tree layer - says
    nothing about the others, so their baseline PASSes are not "missing": they
    were not asked for. A document that does not say which kinds it ran is taken
    to cover every kind the baseline holds.
    """
    kinds = set(doc.get("meta", {}).get("kinds") or FULL_KINDS)
    return set(FULL_KINDS) if not kinds else kinds


def regressions(baseline: Mapping[str, str], doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each baseline PASS of a kind this run covers that it did not pass, with what it got.

    A run of one kind - ``--kind roottest`` while working on the tree layer - says
    nothing about the others, so their baseline PASSes are not "missing": they
    were not asked for. An id with no kind in it cannot be attributed to one, and
    is judged as it always was.
    """
    if not baseline or any(status not in STATUSES for status in baseline.values()):
        raise ValueError("The rootsuite baseline must map test ids to known statuses.")
    records = {record["id"]: record for record in doc["tests"]}
    kinds = judged(doc)
    failed = []
    for test_id, status in sorted(baseline.items()):
        kind, sep, _ = test_id.partition("/")
        if status != "PASS" or (sep and kind not in kinds):
            continue
        record = records.get(test_id, {"id": test_id, "status": "MISSING",
                                       "reason": "not in the results"})
        if record["status"] != "PASS":
            failed.append(dict(record))
    return failed


def timeouts(failed: list[dict[str, Any]]) -> list[str]:
    """The regressions that timed out, which a rerun alone may excuse."""
    return [record["id"] for record in failed if record["status"] == "TIMEOUT"]


def _trusted_records(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The run's records, refused when a status is missing or one we cannot read.

    Both ways of writing a baseline promise the same thing about a row - that
    the test passed - so both must refuse a run whose verdicts mean nothing.
    """
    records = doc["tests"]
    if not records or any(record["status"] not in STATUSES for record in records):
        raise ValueError("A baseline requires nonempty results with known statuses.")
    return records


def _write(statuses: Mapping[str, str], destination: Path) -> None:
    """One row per line, sorted, so a merge shows up as a diff of the rows it changed."""
    destination.write_text(json.dumps(dict(statuses), indent=1, sort_keys=True) + "\n")


def write_baseline(doc: Mapping[str, Any], destination: Path) -> None:
    """Every test's status from a full run of the oracle-free kinds."""
    meta = doc["meta"]
    kinds = set(meta.get("kinds") or FULL_KINDS)
    if meta.get("only") or not set(FULL_KINDS) <= kinds:
        raise ValueError("A baseline requires a full run of the gtest, roottest, pyunittest "
                         "and test kinds.")
    records = _trusted_records(doc)
    _write({record["id"]: record["status"] for record in records}, destination)


def merge_baseline(doc: Mapping[str, Any], destination: Path) -> None:
    """Record the rows a run saw pass, leaving every other row exactly as it was.

    A PASS a run saw is a promise from then on, and a round that gains one cannot
    wait for a many-hour run of all four kinds to write it down. The converse is
    not true: a FAIL, DIFF, UNSUPPORTED or TIMEOUT in this run says nothing about
    whether the test ever passed - the baseline's FAIL may be an old run's, the
    UNSUPPORTED a refusal the code no longer makes, the TIMEOUT this machine -
    so a row that is not PASS now is left alone rather than overwritten.

    That keeps the ratchet as strict as it was: :func:`regressions` still fails a
    baseline PASS the next run loses or does not run, whoever wrote the row.
    """
    records = _trusted_records(doc)
    statuses = json.loads(destination.read_text()) if destination.exists() else {}
    statuses.update({record["id"]: "PASS" for record in records if record["status"] == "PASS"})
    _write(statuses, destination)
