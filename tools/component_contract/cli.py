"""Expose inventories and enforcement without automatically approving existing debt."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .analysis import Report, graph, inspect, inventory, ratchet
from .isolation import isolate
from .model import ContractError, Policy, load_policy


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    commands = result.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Reject new debt and stale baseline allowances.")
    check.add_argument("--strict", action="store_true", help="Require zero dependency debt.")
    check.add_argument("--inventory", type=Path, help="Write a NEW, unreviewed source inventory.")
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


def check(policy: Policy, args: argparse.Namespace) -> int:
    report = inspect(policy)
    _show_report(report)
    if args.inventory is not None:
        _write_inventory(policy, report, args.inventory)
        return 0
    if args.strict:
        return int(bool(report.violations))
    added, removed = ratchet(policy, report, policy.root / policy.baseline)
    _show_changes(added, removed)
    return int(bool(added or removed))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        policy = load_policy(args.root.resolve())
        if args.command == "check":
            return check(policy, args)
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
          f"{len(report.violations)} existing violations.")


def _write_inventory(policy: Policy, report: Report, path: Path) -> None:
    with path.open("x") as stream:
        json.dump(inventory(policy, report), stream, indent=2)
        stream.write("\n")
    print("Inventory written for review; no debt allowance was approved.")


def _show_changes(added: Counter[str], removed: Counter[str]) -> None:
    for label, records in (("NEW", added), ("RESOLVED: shrink the reviewed inventory", removed)):
        for record, count in sorted(records.items()):
            print(f"{label} ({count}): {record}")
