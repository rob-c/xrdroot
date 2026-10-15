"""Comparable wall times, including the original runtime of cached oracle runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def measured(record: Mapping[str, Any]) -> dict[str, float | None]:
    times = {}
    for side in ("oracle", "xrdroot"):
        run = record.get(side) or {}
        value = run.get("runtime")
        times[side] = value if isinstance(value, (int, float)) and math.isfinite(value) else None
    root, mine = times["oracle"], times["xrdroot"]
    ratio = mine / root if root is not None and root > 0 and mine is not None else None
    return {"root_seconds": root, "xrdroot_seconds": mine, "ratio": ratio}


def rows(
    records: Sequence[Mapping[str, Any]], previous: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    before = (previous or {}).get("tutorials", {})
    found = []
    for record in records:
        times = measured(record)
        old = before.get(record["path"], {}).get("ratio")
        ratio = times["ratio"]
        change = ratio / old if old and ratio is not None else None
        found.append({"path": record["path"], **times, "ratio_change": change})
    return sorted(found, key=lambda row: (-(row["xrdroot_seconds"] or 0), row["path"]))


def snapshot(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generated": doc["generated"],
        "meta": doc["meta"],
        "tutorials": {record["path"]: measured(record) for record in doc["tutorials"]},
    }


def display(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def cells(row: Mapping[str, Any]) -> list[str]:
    return [str(row["path"]), *(display(row[key]) for key in (
        "root_seconds", "xrdroot_seconds", "ratio", "ratio_change"
    ))]


#: The same measurements appear in the Markdown and HTML reports.
HEADER = ("tutorial", "ROOT s", "xrdroot s", "xrdroot/ROOT", "ratio/previous")
