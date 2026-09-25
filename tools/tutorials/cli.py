"""``python -m tools.tutorials``: the harness's command line.

python -m tools.tutorials run --tutorials DIR [--only hist/ --jobs N --timeout S]
                              [--refresh-oracle] [--oracle-only | --no-oracle]
python -m tools.tutorials report [--results DIR/results.json]
python -m tools.tutorials list --tutorials DIR [--only ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import report
from .catalog import catalog
from .compare import Tolerance
from .environment import cmake_variables, find_oracle, root_config, targets
from .harness import DEFAULT_CACHE, Harness, Settings, selected

__all__ = ["main", "parser"]

#: The repository this harness is in, whose ``src`` is the xrdroot measured by default.
REPOSITORY = Path(__file__).resolve().parents[2]


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(
        prog="python -m tools.tutorials",
        description="Run ROOT's tutorials under ROOT and under xrdroot, and compare.",
    )
    commands = top.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run the tutorials and write the reports")
    _common(run)
    run.add_argument("--jobs", "-j", type=int, default=4, help="parallel runs (4)")
    run.add_argument("--timeout", type=float, default=300.0, help="seconds per run (300)")
    run.add_argument("--refresh-oracle", action="store_true", help="rerun ROOT, ignoring the cache")
    run.add_argument("--network", action="store_true", help="run tutorials that need the network")
    run.add_argument("--oracle-only", action="store_true", help="run ROOT only: fill the cache")
    run.add_argument("--no-oracle", action="store_true", help="run xrdroot only, uncompared")
    run.add_argument("--out", type=Path, help="where results go (CACHE/results)")
    run.add_argument("--python", default=sys.executable, help="the Python that runs xrdroot")
    run.add_argument(
        "--xrdroot-src", type=Path, default=REPOSITORY / "src", help="put on PYTHONPATH"
    )
    run.add_argument("--rtol", type=float, default=1e-6, help="relative tolerance for numbers")
    run.add_argument("--atol", type=float, default=1e-12, help="absolute tolerance for numbers")
    run.add_argument("--image-threshold", type=float, default=0.95, help="least image similarity")
    listing = commands.add_parser("list", help="the tutorials and how ROOT's CMakeLists runs them")
    _common(listing)
    again = commands.add_parser("report", help="redraw summary.md and report.html from results")
    again.add_argument(
        "--results", type=Path, default=DEFAULT_CACHE.expanduser() / "results" / "results.json"
    )
    return top


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--tutorials", type=Path, required=True, help="ROOT's tutorials directory")
    command.add_argument("--only", action="append", default=[], help="a directory or file prefix")
    command.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="the oracle cache")
    command.add_argument(
        "--oracle-prefix",
        type=Path,
        help="the ROOT installation to compare with ($XRDROOT_ORACLE_PREFIX, "
        "~/.local/root-oracle, else root-config on PATH)",
    )
    command.add_argument("--root-python", help="the Python whose PyROOT runs .py tutorials")
    command.add_argument(
        "--features",
        help="ROOT build options to assume, space-separated, instead of root-config's "
        "(with no ROOT here: which tutorials a ROOT built so would run)",
    )


def _catalogue(args: argparse.Namespace) -> tuple[dict[str, Any], Any]:
    oracle = find_oracle(root_config(args.oracle_prefix), args.root_python)
    tutorials = args.tutorials.resolve()
    features = oracle.features if oracle is not None else frozenset()
    if args.features is not None:
        features = frozenset(args.features.replace(",", " ").split())
    python = (oracle.python if oracle is not None else None) or ""
    module = oracle.has_module if oracle is not None else (lambda name: False)
    found = catalog(
        tutorials, cmake_variables(features, tutorials, python), targets(features), module
    )
    return found, oracle


def _run(args: argparse.Namespace) -> int:
    found, oracle = _catalogue(args)
    if oracle is None:
        print("no ROOT found: xrdroot is run uncompared", file=sys.stderr)
    settings = Settings(
        tutorials=args.tutorials.resolve(),
        cache=args.cache.expanduser(),
        out=args.out,
        jobs=args.jobs,
        timeout=args.timeout,
        refresh_oracle=args.refresh_oracle,
        network=args.network,
        xrdroot_python=args.python,
        xrdroot_src=args.xrdroot_src,
        tolerance=Tolerance(args.rtol, args.atol, args.image_threshold),
        only=tuple(args.only),
        run_oracle=not args.no_oracle,
        run_xrdroot=not args.oracle_only,
    )
    harness = Harness(settings, found, oracle)
    records = harness.run()
    meta = {
        "root_version": oracle.version if oracle else None,
        "root_features": sorted(oracle.features) if oracle else [],
        "root_python": oracle.python if oracle else None,
        "tutorials_dir": str(settings.tutorials),
        "xrdroot_python": settings.xrdroot_python,
        "xrdroot_src": str(settings.xrdroot_src),
        "xrdroot_commit": harness.xrdroot_commit(),
        "xrdroot_has_run": harness.has_run,
        "xrdroot_has_pyroot": harness.has_pyroot,
        "only": list(settings.only),
        "jobs": settings.jobs,
        "timeout": settings.timeout,
        "tolerance": asdict(settings.tolerance),
        "oracle_only": args.oracle_only,
    }
    doc = report.document(records, meta)
    for path in report.write(doc, settings.results):
        print(path)
    print(json.dumps(doc["counts"]))
    return 0


def _list(args: argparse.Namespace) -> int:
    found, _ = _catalogue(args)
    for path in selected(sorted(found), args.only):
        tutorial = found[path]
        why = tutorial.vetoed or f"rc={tutorial.passrc} deps={','.join(tutorial.depends) or '-'}"
        print(f"{path}\t{tutorial.test or '-'}\t{why}")
    return 0


def _report(args: argparse.Namespace) -> int:
    doc = json.loads(args.results.read_text())
    if doc.get("schema") != report.SCHEMA:
        print(
            f"{args.results}: schema {doc.get('schema')!r}, not {report.SCHEMA!r}", file=sys.stderr
        )
        return 2
    for path in report.write(doc, args.results.parent):
        print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {"run": _run, "list": _list, "report": _report}
    return commands[args.command](args)
