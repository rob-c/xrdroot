"""``results.json`` and ``summary.md``: a run's records, and the tables people read.

The JSON is the record - every test's status, reason, time and log - and
the Markdown is drawn from it alone, so ``report`` can redraw it after the
fact. Pass rates are given twice: over every test, and over the tests in
scope (OUT-OF-SCOPE left out), because the first is the distance from
ROOT's whole suite and the second the distance from what xrdroot means to do.

A gtest program or unittest file passes only when every case in it does, so
the cases are counted too, and causes are ranked by how many tests each one
alone keeps from passing: a gtest program's causes are its failed cases'.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .judge import STATUSES
from .model import KINDS

__all__ = ["SCHEMA", "document", "summary", "write", "cause"]

#: The results document's format; ``report`` and ``ratchet`` refuse any other.
SCHEMA = "rootsuite/1"

#: The statuses whose reasons are failure causes worth counting.
_FAILING = ("FAIL", "DIFF", "UNSUPPORTED", "TIMEOUT")
_ORDER = [*KINDS.values(), "other"]
#: Keys a row's counter holds beside the statuses: the cases inside its tests.
_CASES_PASSED, _CASES_TOTAL = " cases passed", " cases"


def document(records: Sequence[Mapping[str, Any]], meta: Mapping[str, Any]) -> dict[str, Any]:
    """The results: counts by kind and status, then every record."""
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        kind = counts.setdefault(record["kind"], dict.fromkeys(STATUSES, 0))
        kind[record["status"]] = kind.get(record["status"], 0) + 1
    return {"schema": SCHEMA, "meta": dict(meta), "counts": counts, "tests": list(records)}


def cause(reason: str) -> str:
    """A reason with what varies between tests - paths, line numbers, counts - made generic.

    Quoted names stay: ``module 'ROOT' has no attribute 'TRef'`` is a cause
    worth counting apart from ``... 'TRefArray'``.
    """
    text = re.sub(r"(?<![\w'\"])\S*/\S+", "<path>", reason)
    text = re.sub(r"\b\d+\b", "N", text)
    return text[:160]


def _rates(statuses: Counter[str]) -> tuple[str, str]:
    total = sum(statuses[status] for status in STATUSES)
    in_scope = total - statuses["OUT-OF-SCOPE"]
    passed = statuses["PASS"]
    overall = f"{100 * passed / total:.1f}%" if total else "-"
    return overall, (f"{100 * passed / in_scope:.1f}%" if in_scope else "-")


def _cases(statuses: Counter[str]) -> str:
    """``passed/total`` of the cases inside the row's tests, or ``-`` when none were counted."""
    total = statuses[_CASES_TOTAL]
    return f"{statuses[_CASES_PASSED]}/{total}" if total else "-"


def _table(rows: Iterable[tuple[str, Counter[str]]], first: str) -> list[str]:
    head = [first, "tests", *STATUSES, "pass rate", "in scope", "cases passed"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, statuses in rows:
        tests = sum(statuses[status] for status in STATUSES)
        cells = [name, str(tests), *(str(statuses[s]) for s in STATUSES), *_rates(statuses),
                 _cases(statuses)]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _by(records: Iterable[Mapping[str, Any]], key: str) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = {}
    for record in records:
        row = grouped.setdefault(record[key], Counter())
        row[record["status"]] += 1
        cases = record.get("cases") or {}
        row[_CASES_PASSED] += cases.get("passed", 0)
        row[_CASES_TOTAL] += sum(cases.values())
    return grouped


def _causes_of(record: Mapping[str, Any]) -> set[str]:
    """What keeps a test from passing: its failed cases' causes, else its reason's."""
    failures = record.get("failures") or []
    if failures:
        return {cause(failure.partition(": ")[2]) for failure in failures}
    return {cause(record.get("reason", ""))}


def _unblockers(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Causes ranked by how many tests each alone blocks, then by how many it appears in."""
    alone: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    for record in records:
        if record["status"] not in _FAILING:
            continue
        causes = _causes_of(record)
        seen.update(causes)
        if len(causes) == 1:
            alone.update(causes)
    ranked = sorted(seen, key=lambda text: (-alone[text], -seen[text], text))[:15]
    return [f"{alone[text]:5d} {seen[text]:5d}  {text}" for text in ranked]


def _kinds(records: Sequence[Mapping[str, Any]]) -> list[str]:
    present = {record["kind"] for record in records}
    return [kind for kind in _ORDER if kind in present] + sorted(present - set(_ORDER))


def _kind_section(kind: str, records: Sequence[Mapping[str, Any]]) -> list[str]:
    mine = [record for record in records if record["kind"] == kind]
    lines = [f"## {kind}", "", *_table(sorted(_by(mine, "area").items()), "area"), ""]
    top = Counter(cause(r.get("reason", "")) for r in mine if r["status"] in _FAILING)
    if top:
        causes = [f"{count:5d}  {text}" for text, count in top.most_common(10)]
        lines += ["Top failure causes:", "", "```", *causes, "```", ""]
    if any(record.get("failures") for record in mine):
        lines += ["Causes by the tests each alone blocks, then the tests it is in (a gtest "
                  "program's causes are its failed cases'):", "", "```", *_unblockers(mine),
                  "```", ""]
    return lines


def summary(doc: Mapping[str, Any]) -> str:
    """The Markdown summary: by kind, by area within each kind, causes, scope, slowest."""
    records: list[Mapping[str, Any]] = doc["tests"]
    meta = doc.get("meta", {})
    lines = ["# ROOT test suite under xrdroot", ""]
    lines += [f"- {key}: {meta[key]}" for key in sorted(meta) if meta[key] not in ("", None, [])]
    by_kind = _by(records, "kind")
    lines += ["", *_table([(k, by_kind[k]) for k in _kinds(records)], "kind"), ""]
    for kind in _kinds(records):
        lines += _kind_section(kind, records)
    lines += _scope_section(records) + _slowest_section(records)
    return "\n".join(lines) + "\n"


def _scope_section(records: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons = Counter(r.get("reason", "") for r in records if r["status"] == "OUT-OF-SCOPE")
    lines = ["## Out of scope", ""]
    return lines + [f"- {count}: {reason}" for reason, count in reasons.most_common()] + [""]


def _slowest_section(records: Sequence[Mapping[str, Any]]) -> list[str]:
    timed = sorted((r for r in records if r.get("runtime")), key=lambda r: -r["runtime"])[:15]
    lines = ["## Slowest", ""]
    return lines + [f"- {r['runtime']:.1f} s  {r['id']}  ({r['status']})" for r in timed] + [""]


def write(doc: Mapping[str, Any], out: Path) -> list[Path]:
    """``results.json`` and ``summary.md`` under ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    results, markdown = out / "results.json", out / "summary.md"
    results.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    markdown.write_text(summary(doc), encoding="utf-8")
    return [results, markdown]
