"""Enforce the component graph: every import must follow a declared edge."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import Report, graph, inspect
from .isolation import isolate
from .model import ContractError, Policy, load_policy


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Require zero undeclared imports.")
    drawing = commands.add_parser("graph", help="Show ownership counts and the declared DAG.")
    drawing.add_argument("--dot", action="store_true")
    isolation = commands.add_parser("isolate", help="Import behind fresh component boundaries.")
    isolation.add_argument("components", nargs="+")
    isolation.add_argument(
        "--tests",
        nargs="+",
        default=[],
        metavar="SCOPE",
        help="Run declared component test scopes with their fixture cutoff.",
    )
    return result


def check(policy: Policy) -> int:
    report = inspect(policy)
    _show_report(report)
    return int(bool(report.violations))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        policy = load_policy(args.root.resolve())
        if args.command == "check":
            return check(policy)
        if args.command == "graph":
            print(graph(policy, inspect(policy), args.dot))
            return 0
        return isolate(policy, args.components, args.tests)
    except (ContractError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"Component contract: {error}")
        return 2


def _show_report(report: Report) -> None:
    for item in report.violations:
        print(f"{item.path}:{item.line}: {item.component} -> {item.target} [{item.rule}]")
    for item in report.reviewed_private:
        print(f"{item.path}:{item.line}: {item.component} -> {item.target} "
              "[external-private-reviewed]")
    print(f"{len(report.modules)} source modules; {report.import_count} imports; "
          f"{len(report.violations)} undeclared imports.")

