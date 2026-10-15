"""A passing tutorial remains a promise, even when a later run omits it."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import timing
from .classify import STATUSES
from .compare import Tolerance
from .environment import find_oracle, root_config
from .harness import DEFAULT_CACHE, Harness, Settings

#: Kept in the repository so checking a run never silently starts with no promises.
BASELINE = Path(__file__).with_name("baseline.json")


def regressions(baseline: Mapping[str, str], doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not baseline or any(status not in STATUSES for status in baseline.values()):
        raise ValueError("The tutorial baseline must contain paths with known statuses.")
    records = {record["path"]: record for record in doc["tutorials"]}
    failed = []
    for path, status in sorted(baseline.items()):
        if status != "PASS":
            continue
        record = records.get(path, {"path": path, "status": "MISSING", "reason": "not in results"})
        if record["status"] != "PASS":
            failed.append(dict(record))
    return failed


def write_baseline(doc: Mapping[str, Any], destination: Path) -> None:
    meta = doc["meta"]
    _require_full_run(meta)
    records = doc["tutorials"]
    if not records or any(record["status"] not in STATUSES for record in records):
        raise ValueError("A baseline requires nonempty results with known statuses.")
    statuses = {record["path"]: record["status"] for record in records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(statuses, indent=1, sort_keys=True) + "\n")
    destination.with_name(destination.stem + "-timings.json").write_text(
        json.dumps(timing.snapshot(doc), indent=1, sort_keys=True) + "\n"
    )


def _require_full_run(meta: Mapping[str, Any]) -> None:
    partial = meta.get("only") or meta.get("oracle_only") or meta.get("no_oracle")
    if partial or not meta.get("root_version"):
        raise ValueError("A baseline requires a full run compared against ROOT.")


def recheck(doc: Mapping[str, Any], failed: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """Only an actual isolated PASS excuses a timeout from the concurrent run."""
    from .catalog import catalog
    from .environment import cmake_variables, targets
    from .report import document, write

    paths = _timeouts(failed)
    if not paths:
        return {}
    meta = doc["meta"]
    oracle = _oracle(meta)
    if oracle is None or oracle.version != meta.get("root_version"):
        raise ValueError("Timeout rechecks require the same ROOT oracle as the original run.")
    tutorials = Path(meta["tutorials_dir"])
    found = catalog(tutorials, cmake_variables(oracle.features, tutorials, oracle.python or ""),
                    targets(oracle.features), oracle.has_module)
    settings = _settings(meta, paths, out)
    harness = Harness(settings, found, oracle)
    if harness.xrdroot_commit() != meta.get("xrdroot_commit"):
        raise ValueError("Timeout rechecks require the original xrdroot commit.")
    records = harness.run()
    write(document(records, {**meta, "jobs": 1, "only": paths,
                             "xrdroot_commit": harness.xrdroot_commit()}), out)
    return {record["path"]: record for record in records}


def _timeouts(failed: list[dict[str, Any]]) -> list[str]:
    return [record["path"] for record in failed if (record.get("xrdroot") or {}).get("timed_out")]


def _oracle(meta: Mapping[str, Any]) -> Any:
    prefix = Path(meta["oracle_prefix"]) if meta.get("oracle_prefix") else None
    return find_oracle(root_config(prefix), meta.get("root_python"))


def _settings(meta: Mapping[str, Any], paths: list[str], out: Path) -> Settings:
    return Settings(
        tutorials=Path(meta["tutorials_dir"]),
        cache=Path(meta.get("cache", DEFAULT_CACHE)).expanduser(),
        out=out, jobs=1, timeout=min(meta.get("timeout", 300), 300), only=tuple(paths),
        xrdroot_python=meta["xrdroot_python"], xrdroot_src=Path(meta["xrdroot_src"]),
        tolerance=Tolerance(**meta.get("tolerance", {})), network=meta.get("network", False),
    )
