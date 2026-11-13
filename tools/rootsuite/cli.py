"""``python -m tools.rootsuite``: the harness's command line.

python -m tools.rootsuite list --root ROOT [--kind K] [--only GLOB] [--tests]
python -m tools.rootsuite run --root ROOT [--kind K] [--only GLOB] [--jobs N] [--timeout S]
                              [--out DIR]
python -m tools.rootsuite report [--results DIR/results.json]
python -m tools.rootsuite ratchet RESULTS.json [--baseline FILE] [--rerun-timeouts]
python -m tools.rootsuite baseline RESULTS.json [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import ratchet, report, scope
from .discover import DEFAULT_FEATURES, discover
from .model import CI_COUNTS, Test
from .runner import Runner, Settings, select

__all__ = ["main", "parser", "DEFAULT_OUT", "KINDS"]

#: The repository this harness is in, whose ``src`` is the xrdroot measured by default.
REPOSITORY = Path(__file__).resolve().parents[2]

#: Where results go unless told otherwise: outside the repository, always.
DEFAULT_OUT = Path(os.environ.get("XRDROOT_ROOTSUITE_OUT",
                                  Path(tempfile.gettempdir()) / "xrdroot-rootsuite"))

#: The kinds ``--kind`` takes: the oracle-free ones, ctest programs of no family, tutorials.
KINDS = ("gtest", "roottest", "pyunittest", "test", "other", "tutorial")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="python -m tools.rootsuite",
        description="Discover ROOT's ctest suite from its CMake and run it against xrdroot.",
    )
    commands = top.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="the tests ROOT's CMake registers, and counts")
    _selection(listing)
    listing.add_argument("--tests", action="store_true", help="one line per test")
    run = commands.add_parser("run", help="run the tests against xrdroot and write the reports")
    _selection(run)
    run.add_argument("--jobs", "-j", type=int, default=6, help="parallel workers (6)")
    run.add_argument("--timeout", type=float, default=300.0, help="seconds per test (300)")
    run.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"results ({DEFAULT_OUT})")
    run.add_argument("--python", default=sys.executable, help="the Python that runs xrdroot")
    run.add_argument("--xrdroot-src", type=Path, default=REPOSITORY / "src",
                     help="put on PYTHONPATH (this repository's src)")
    again = commands.add_parser("report", help="redraw summary.md from results.json")
    again.add_argument("--results", type=Path, default=DEFAULT_OUT / "results.json")
    check = commands.add_parser("ratchet", help="fail if a test in the baseline stops passing")
    check.add_argument("results", type=Path)
    check.add_argument("--baseline", type=Path, default=ratchet.BASELINE)
    check.add_argument("--rerun-timeouts", action="store_true", help="recheck timeouts alone")
    baseline = commands.add_parser("baseline", help="record every status from a full run")
    baseline.add_argument("results", type=Path)
    baseline.add_argument("--output", type=Path, default=ratchet.BASELINE)
    return top


def _selection(command: argparse.ArgumentParser) -> None:
    command.add_argument("--root", type=Path, required=True, help="ROOT's source tree")
    command.add_argument("--kind", action="append", choices=KINDS, default=[],
                         help="only tests of this kind (repeatable)")
    command.add_argument("--only", action="append", default=[], help="only ids matching a glob")
    command.add_argument("--features", help="build options to assume, space-separated")
    command.add_argument("--scope", type=Path, default=scope.SCOPE, help="the scope labels")


def _features(args: argparse.Namespace) -> frozenset[str]:
    return frozenset(args.features.replace(",", " ").split()) if args.features else DEFAULT_FEATURES


def _tests(args: argparse.Namespace) -> list[Test]:
    kinds = [kind for kind in args.kind if kind != "tutorial"]
    if args.kind and not kinds:
        return []
    return select(discover(args.root, _features(args)), kinds, args.only)


def _list(args: argparse.Namespace) -> int:
    tests = _tests(args)
    labels = scope.load(args.scope)
    if args.tests:
        for test in tests:
            label = scope.label_for(test, labels)
            note = f"OUT-OF-SCOPE: {label.reason}" if label else test.skip or test.runner
            print(f"{test.id}\t{test.ctest}\t{note}")
    counts = Counter(test.kind for test in tests)
    if not args.kind or "tutorial" in args.kind:
        counts["tutorial"] = _tutorial_count(args)
    in_scope = sum(1 for test in tests if scope.label_for(test, labels) is None)
    _print_counts(counts, in_scope)
    return 0


def _print_counts(counts: Counter[str], in_scope: int) -> None:
    """Each kind's count beside the CI job's, then the total and how many are in scope."""
    for kind in KINDS:
        ci = CI_COUNTS.get(kind)
        print(f"{kind:12} {counts.get(kind, 0):5d}" + (f"   CI: {ci}" if ci else ""))
    print(f"{'total':12} {sum(counts.values()):5d}   in scope (not tutorials): {in_scope}")


def _tutorial_count(args: argparse.Namespace) -> int:
    """The tutorials ROOT's ``tutorials/CMakeLists.txt`` makes tests of, by the tutorial harness."""
    from tools.tutorials.catalog import catalog
    from tools.tutorials.environment import cmake_variables, targets

    tutorials = (args.root / "tutorials").resolve()
    if not tutorials.is_dir():
        return 0
    features = _features(args)
    found = catalog(tutorials, cmake_variables(features, tutorials), targets(features),
                    lambda name: True)
    return sum(1 for tutorial in found.values() if tutorial.test and not tutorial.vetoed)


def _run(args: argparse.Namespace) -> int:
    status = 0
    if "tutorial" in args.kind:
        status = _run_tutorials(args)
    tests = _tests(args)
    if not tests:
        return status
    settings = Settings(root=args.root.resolve(), out=args.out.resolve(), python=args.python,
                        xrdroot_src=args.xrdroot_src, jobs=args.jobs, timeout=args.timeout,
                        kinds=tuple(args.kind), only=tuple(args.only), scope=args.scope)
    start = time.monotonic()
    records = Runner(settings).run(tests)
    doc = report.document(records, _meta(args, settings, time.monotonic() - start))
    for path in report.write(doc, settings.out):
        print(path)
    print(json.dumps(doc["counts"]))
    return status


def _meta(args: argparse.Namespace, settings: Settings, seconds: float) -> dict[str, Any]:
    return {
        "root": str(settings.root), "python": settings.python,
        "xrdroot_src": str(settings.xrdroot_src), "xrdroot_commit": _commit(settings.xrdroot_src),
        "kinds": list(settings.kinds), "only": list(settings.only), "jobs": settings.jobs,
        "timeout": settings.timeout, "features": sorted(_features(args)),
        "wall_seconds": round(seconds, 1), "out": str(settings.out),
    }


def _commit(src: Path | None) -> str:
    if src is None:
        return ""
    try:
        done = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip()


def _run_tutorials(args: argparse.Namespace) -> int:
    """``--kind tutorial``: the tutorial harness, which needs ROOT itself as its oracle."""
    from tools.tutorials.cli import main as tutorials

    argv = ["run", "--tutorials", str(args.root / "tutorials"), "--jobs", str(args.jobs),
            "--timeout", str(args.timeout), "--out", str(args.out / "tutorials"),
            "--python", args.python, "--xrdroot-src", str(args.xrdroot_src)]
    argv += [item for prefix in args.only for item in ("--only", prefix)]
    return tutorials(argv)


def _load(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(path.read_text())
    if doc.get("schema") != report.SCHEMA:
        raise ValueError(f"{path}: schema {doc.get('schema')!r}, not {report.SCHEMA!r}")
    return doc


def _report(args: argparse.Namespace) -> int:
    doc = _load(args.results)
    doc = {**doc, **report.document(doc["tests"], doc["meta"])}
    for path in report.write(doc, args.results.parent):
        print(path)
    return 0


def _ratchet(args: argparse.Namespace) -> int:
    doc = _load(args.results)
    failed = ratchet.regressions(json.loads(args.baseline.read_text()), doc)
    excused = _recheck(doc, ratchet.timeouts(failed), args.results.parent) \
        if args.rerun_timeouts else set()
    failures = 0
    for record in failed:
        if record["id"] in excused:
            print(f"{record['id']}: timed out, but passed when rerun alone")
            continue
        failures += 1
        print(f"{record['id']}: PASS -> {record['status']}: {record.get('reason', '')}")
    print(f"{failures} regressions")
    return int(bool(failures))


def _recheck(doc: dict[str, Any], ids: list[str], parent: Path) -> set[str]:
    """Rerun each timed-out test alone, with the original run's settings; keep those that pass."""
    if not ids:
        return set()
    meta = doc["meta"]
    root = Path(meta["root"])
    settings = Settings(root=root, out=parent / "timeout-rechecks", python=meta["python"],
                        xrdroot_src=Path(meta["xrdroot_src"]), jobs=1,
                        timeout=min(float(meta.get("timeout", 300)), 300.0), only=tuple(ids))
    tests = select(discover(root, meta.get("features") or DEFAULT_FEATURES), (), ids)
    records = Runner(settings).run(tests)
    return {record["id"] for record in records if record["status"] == "PASS"}


def _baseline(args: argparse.Namespace) -> int:
    ratchet.write_baseline(_load(args.results), args.output)
    print(args.output)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {"list": _list, "run": _run, "report": _report, "ratchet": _ratchet,
                "baseline": _baseline}
    try:
        return commands[args.command](args)
    except (OSError, ValueError) as why:
        print(str(why), file=sys.stderr)
        return 2
