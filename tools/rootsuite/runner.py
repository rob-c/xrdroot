"""Running the selected tests: in working copies of their directories, in ctest's order.

ctest runs a directory's tests in its build directory, in parallel unless
``DEPENDS`` or fixtures order them, and a test reads what earlier ones
wrote there. Here each source directory a test uses is copied once under
``OUT/work`` (subdirectories with no CMakeLists of their own - data - come
with it), so the ROOT sources are only ever read; tests linked by
``DEPENDS`` or fixtures form one unit, run in order by one worker; units
run in parallel. Each test's command, output, diff and junit XML are kept
under ``OUT/logs/<id>/``.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import cases, judge, plan, scope
from .discover import fixture_providers
from .model import Test, area_of
from .process import Result, execute

__all__ = ["Settings", "select", "Runner", "units"]


@dataclass(frozen=True)
class Settings:
    """How a run goes: what is run with what, how wide, how long, where results go."""

    root: Path
    out: Path
    python: str = sys.executable
    xrdroot_src: Path | None = None
    jobs: int = 6
    timeout: float = 300.0
    kinds: tuple[str, ...] = ()
    only: tuple[str, ...] = ()
    scope: Path = scope.SCOPE
    progress: Callable[[str], None] = field(default=lambda line: print(line, file=sys.stderr),
                                            compare=False)


def select(
    tests: Iterable[Test], kinds: Sequence[str] = (), only: Sequence[str] = ()
) -> list[Test]:
    """The tests of the given kinds whose ids match any ``--only`` glob (all, when none given)."""
    return [
        test for test in tests
        if (not kinds or test.kind in kinds)
        and (not only or any(fnmatch.fnmatchcase(test.id, pattern) for pattern in only))
    ]


def units(tests: Sequence[Test]) -> list[list[Test]]:
    """Tests joined by ``DEPENDS`` or fixtures, each group in an order that respects them."""
    by_name = {test.ctest: test for test in tests}
    providers = fixture_providers(tests)
    before = {test.id: _upstream(test, by_name, providers) for test in tests}
    return [_ordered(group, before) for group in _components(tests, before)]


def _components(tests: Sequence[Test], before: dict[str, list[Test]]) -> list[list[Test]]:
    """The tests split where no dependency joins them (union-find), in their given order."""
    index = {test.id: at for at, test in enumerate(tests)}
    parent = list(range(len(tests)))

    def find(at: int) -> int:
        while parent[at] != at:
            parent[at] = parent[parent[at]]
            at = parent[at]
        return at

    for test in tests:
        for other in before[test.id]:
            parent[find(index[test.id])] = find(index[other.id])
    groups: dict[int, list[Test]] = {}
    for test in tests:
        groups.setdefault(find(index[test.id]), []).append(test)
    return list(groups.values())


def _upstream(test: Test, by_name: dict[str, Test], providers: dict[str, list[Test]]) -> list[Test]:
    found = [by_name[name] for name in test.depends if name in by_name]
    found += [p for f in test.fixtures_required for p in providers.get(f, []) if p != test]
    return found


def _ordered(group: list[Test], before: dict[str, list[Test]]) -> list[Test]:
    """A topological order, registration order breaking ties; cycles are cut where met."""
    done: list[Test] = []
    placed: set[str] = set()

    def visit(test: Test, path: set[str]) -> None:
        if test.id in placed or test.id in path:
            return
        for other in before[test.id]:
            visit(other, path | {test.id})
        placed.add(test.id)
        done.append(test)

    for test in group:
        visit(test, set())
    return done


class Runner:
    """One run over a selection: plans, working copies, workers, verdicts, records."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.labels = scope.load(settings.scope)
        self.work = settings.out / "work"
        self.logs = settings.out / "logs"
        self.context = plan.Context(settings.root, self.work, settings.python,
                                    settings.xrdroot_src)
        self.records: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.modules: dict[str, bool] = {}
        self.total = 0

    def run(self, tests: Sequence[Test]) -> list[dict[str, Any]]:
        """Every test judged or labelled, in the order given."""
        self.total = len(tests)
        planned: list[tuple[Test, plan.Invocation]] = []
        for test in tests:
            invocation = self._planned(test)
            if invocation is not None:
                planned.append((test, invocation))
        prepare(self.settings.root, self.work, _directories(t for t, _ in planned))
        invocations = {test.id: invocation for test, invocation in planned}
        with ThreadPoolExecutor(max_workers=max(1, self.settings.jobs)) as pool:
            for unit in units([test for test, _ in planned]):
                pool.submit(self._unit, unit, invocations)
        return [self.records[test.id] for test in tests]

    def _planned(self, test: Test) -> plan.Invocation | None:
        label = scope.label_for(test, self.labels)
        if label is not None:
            self._record(test, judge.Verdict("OUT-OF-SCOPE", label.reason))
            return None
        invocation = plan.plan(test, self.context, self.logs / test.id)
        missing = [name for name in test.python_deps if not self._has_module(name)]
        if isinstance(invocation, str) or missing:
            why = invocation if isinstance(invocation, str) else (
                f"needs the Python package {missing[0]}, which the xrdroot Python lacks")
            self._record(test, judge.Verdict("SKIP", why))
            return None
        return invocation

    def _has_module(self, name: str) -> bool:
        if name not in self.modules:
            code = f"import importlib.util as u, sys; sys.exit(u.find_spec({name!r}) is None)"
            done = subprocess.run([self.settings.python, "-c", code], capture_output=True,
                                  env=self.environment({}), timeout=120)
            self.modules[name] = done.returncode == 0
        return self.modules[name]

    def environment(self, extra: dict[str, str]) -> dict[str, str]:
        """The test's environment: this one, with the xrdroot under test first on the path."""
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        if self.settings.xrdroot_src is not None:
            env["PYTHONPATH"] = str(self.settings.xrdroot_src)
        env.update(MPLBACKEND="Agg", **extra)
        return env

    def _unit(self, unit: list[Test], invocations: dict[str, plan.Invocation]) -> None:
        for test in unit:
            try:
                self._one(test, invocations[test.id])
            except Exception as why:  # one broken test must not lose the run's other records
                self._record(test, judge.Verdict("FAIL", f"harness error: {why!r}"))

    def _one(self, test: Test, invocation: plan.Invocation) -> None:
        logs = self.logs / test.id
        logs.mkdir(parents=True, exist_ok=True)
        invocation.cwd.mkdir(parents=True, exist_ok=True)
        timeout = min(self.settings.timeout, test.timeout or self.settings.timeout)
        result = execute(invocation.argv, invocation.cwd, self.environment(invocation.env),
                         timeout, invocation.merge, invocation.stdin)
        if test.runner == "pyunittest" and invocation.junit is not None:
            verdict = cases.judge_pyunittest(test, result, invocation.junit)
        else:
            verdict = judge.judge(test, result, self.settings.root, invocation.cwd)
        if test.runner == "gtest":
            verdict.cases = cases.gtest_counts(result.stdout + result.stderr)
        write_log(logs, invocation, result, verdict)
        self._record(test, verdict, result, logs)

    def _record(self, test: Test, verdict: judge.Verdict, result: Result | None = None,
                logs: Path | None = None) -> None:
        record: dict[str, Any] = {"id": test.id, "ctest": test.ctest, "kind": test.kind,
                                  "area": area_of(test.id), **verdict.to_json()}
        record.pop("diff", None)
        if "reason" in record:  # paths into the working copy say nothing a reader needs
            record["reason"] = record["reason"].replace(f"{self.work}/", "")
        if result is not None:
            record.update(runtime=result.runtime, exit_code=result.exit_code,
                          command=" ".join(result.command))
        if logs is not None:
            record["log"] = str(logs.relative_to(self.settings.out))
        with self.lock:
            self.records[test.id] = record
            line = f"[{len(self.records)}/{self.total}] {verdict.status:12} {test.id}"
        self.settings.progress(line + (f" ({result.runtime:.1f} s)" if result else ""))


def write_log(logs: Path, invocation: plan.Invocation, result: Result,
              verdict: judge.Verdict) -> None:
    """The run as one readable file, and its diff beside it when there is one."""
    header = [f"$ cd {invocation.cwd}", "$ " + " ".join(invocation.argv),
              f"exit: {result.exit_code}  timed out: {result.timed_out}  {result.runtime} s",
              f"status: {verdict.status}  {verdict.reason}"]
    body = ["-- stdout --", result.stdout, "-- stderr --", result.stderr]
    (logs / "run.log").write_text("\n".join(header + body), encoding="utf-8")
    if verdict.diff:
        (logs / "output.diff").write_text(verdict.diff, encoding="utf-8")


def _directories(tests: Iterable[Test]) -> list[str]:
    found: set[str] = set()
    for test in tests:
        found.update(d for d in (test.directory, test.working_dir) if d)
        found.update(str(Path(name).parent) for name in test.files)
    return sorted(found)


def prepare(root: Path, work: Path, directories: Iterable[str]) -> None:
    """Copy each source directory once: its files, and subdirectories holding no tests of their own.

    A copy, never a link, so that whatever a test writes or rewrites stays in
    the working copy; ``OUT/work`` mirrors the source tree, so relative paths
    between test directories still hold.
    """
    for relative in directories:
        source, target = root / relative, work / relative
        if (target / ".rootsuite-copied").exists() or not source.is_dir():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for entry in sorted(source.iterdir()):
            _copy(entry, target / entry.name)
        (target / ".rootsuite-copied").touch()


def _copy(entry: Path, target: Path) -> None:
    if entry.is_dir() and not entry.is_symlink():
        if not (entry / "CMakeLists.txt").exists():
            shutil.copytree(entry, target, symlinks=True, dirs_exist_ok=True)
    elif entry.is_file() and not target.exists():
        shutil.copy2(entry, target)
