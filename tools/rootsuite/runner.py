"""Running the selected tests: in working copies of their directories, in ctest's order.

ctest runs a directory's tests in its build directory, in parallel unless
``DEPENDS`` or fixtures order them, and a test reads what earlier ones
wrote there. Here each source directory a test uses is copied once under
``OUT/work`` (subdirectories with no CMakeLists of their own - data - come
with it), so the ROOT sources are only ever read; the copy is given the
build tree's ``bin/``, ``tutorials/`` and ``hsimple.root``
(:mod:`~tools.rootsuite.buildtree`). Tests linked by ``DEPENDS`` or
fixtures form one unit, run in order by one worker; units run in parallel.

ctest's fixture rules hold: selecting a test selects the tests that set up
the fixtures it requires, and a test whose fixture was not set up - its
setup test did not pass - is not run. It takes its setup's status (a
refusal stays a refusal, a failure is a failure) and the reason at the root
of the chain. ``DEPENDS`` only orders: a dependent runs whatever happened.

A test's builds that belong to the build tree - a library several tests
link - run once, whichever test needs them first. Each test's commands,
output, diff and junit XML are kept under ``OUT/logs/<id>/``.
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

from . import buildtree, cases, judge, plan, scope
from .discover import fixture_providers
from .model import Test, area_of
from .process import Result, execute

__all__ = ["Settings", "select", "Runner", "units", "BLOCKED"]

#: The start of the reason a test was not run for want of a fixture; the root cause follows.
BLOCKED = "A fixture it requires was not set up: "


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
    """The tests of the given kinds whose ids match any ``--only`` glob (all, when none given),
    with every test setting up a fixture they require, as ctest adds them."""
    every = list(tests)
    chosen = _with_setups(every, {test.id for test in every if _chosen(test, kinds, only)})
    return [test for test in every if test.id in chosen]


def _with_setups(every: list[Test], chosen: set[str]) -> set[str]:
    """The chosen tests and, transitively, the tests setting up fixtures they require."""
    providers = fixture_providers(every)
    pending = [test for test in every if test.id in chosen]
    while pending:
        required = pending.pop().fixtures_required
        for provider in (p for f in required for p in providers.get(f, [])):
            if provider.id not in chosen:
                chosen.add(provider.id)
                pending.append(provider)
    return chosen


def _chosen(test: Test, kinds: Sequence[str], only: Sequence[str]) -> bool:
    return (not kinds or test.kind in kinds) and (
        not only or any(fnmatch.fnmatchcase(test.id, pattern) for pattern in only))


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


def blocked(record: dict[str, Any]) -> judge.Verdict:
    """The verdict of a test whose fixture's setup ended as ``record`` did, at the root cause."""
    status = record["status"]
    status = status if status in ("UNSUPPORTED", "SKIP", "OUT-OF-SCOPE") else "FAIL"
    reason = record.get("reason", "") or f"{record['id']} ended {record['status']}"
    return judge.Verdict(status, reason if reason.startswith(BLOCKED) else BLOCKED + reason)


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
        self.available: frozenset[str] | None = None
        self.built: dict[tuple[str, ...], Result] = {}
        self.building: dict[tuple[str, ...], threading.Event] = {}
        self.providers: dict[str, list[Test]] = {}
        self.total = 0

    def run(self, tests: Sequence[Test]) -> list[dict[str, Any]]:
        """Every test judged or labelled, in the order given."""
        self.total = len(tests)
        self.providers = fixture_providers(tests)
        self.available = buildtree.subcommands(self.settings.python, self.environment({}))
        invocations = self._prepared(tests)
        with ThreadPoolExecutor(max_workers=max(1, self.settings.jobs)) as pool:
            for unit in units([test for test in tests if test.id in invocations]):
                pool.submit(self._unit, unit, invocations)
        return [self.records[test.id] for test in tests]

    def _prepared(self, tests: Sequence[Test]) -> dict[str, plan.Invocation]:
        """Each test that will run, planned, with its directories copied and the build tree made;
        the others are recorded as they are."""
        planned = [(test, made) for test in tests if (made := self._planned(test)) is not None]
        prepare(self.settings.root, self.work, _directories(t for t, _ in planned))
        if planned:
            self._build_tree()
        return {test.id: invocation for test, invocation in planned}

    def _build_tree(self) -> None:
        """``bin/``, ``tutorials/`` and ``hsimple.root``, as ROOT's build tree has them."""
        buildtree.launchers(self.work, self.settings.python, self.available)
        directory = buildtree.tutorials(self.settings.root, self.work)
        made = buildtree.hsimple(directory, self.settings.python, self.environment({})) \
            if directory is not None else None
        if made is not None:
            logs = self.logs / "build" / "hsimple"
            logs.mkdir(parents=True, exist_ok=True)
            write_log(logs, plan.Invocation(tuple(made.command), directory), made,
                      judge.Verdict("PASS" if made.exit_code == 0 else "FAIL"))

    def _planned(self, test: Test) -> plan.Invocation | None:
        label = scope.label_for(test, self.labels)
        if label is not None:
            self._record(test, judge.Verdict("OUT-OF-SCOPE", label.reason))
            return None
        invocation = plan.plan(test, self.context, self.logs / test.id)
        if isinstance(invocation, plan.Unrunnable):
            self._record(test, judge.Verdict(invocation.status, invocation.reason))
            return None
        lacking = buildtree.missing(invocation.commands, self.available)
        if lacking:
            self._record(test, judge.Verdict("UNSUPPORTED", lacking))
            return None
        absent = [name for name in test.python_deps if not self._has_module(name)]
        if absent:
            why = f"needs the Python package {absent[0]}, which the xrdroot Python lacks"
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
        """The test's environment: the xrdroot under test first on the path, and, as the driver
        sets them after the test's own, the build tree as ``ROOTSYS`` and its ``bin`` first on
        ``PATH`` - even a ``PATH`` the test sets."""
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        if self.settings.xrdroot_src is not None:
            env["PYTHONPATH"] = str(self.settings.xrdroot_src)
        env.update(MPLBACKEND="Agg", **extra)
        env.update(ROOTSYS=str(self.work),
                   PATH=os.pathsep.join([str(self.work / "bin"), env.get("PATH", "")]))
        return env

    def _unit(self, unit: list[Test], invocations: dict[str, plan.Invocation]) -> None:
        for test in unit:
            try:
                unset = self._unset_fixture(test)
                if unset is not None:
                    self._record(test, blocked(unset))
                else:
                    self._one(test, invocations[test.id])
            except Exception as why:  # one broken test must not lose the run's other records
                self._record(test, judge.Verdict("FAIL", f"harness error: {why!r}"))

    def _unset_fixture(self, test: Test) -> dict[str, Any] | None:
        """The record of a setup test that did not pass, of a fixture ``test`` requires."""
        for fixture in test.fixtures_required:
            for provider in self.providers.get(fixture, []):
                record = self.records.get(provider.id) if provider.id != test.id else None
                if record is not None and record["status"] != "PASS":
                    return record
        return None

    def _one(self, test: Test, invocation: plan.Invocation) -> None:
        logs = self.logs / test.id
        logs.mkdir(parents=True, exist_ok=True)
        invocation.cwd.mkdir(parents=True, exist_ok=True)
        _copy_to_build(invocation.copy, self.work / test.directory)
        timeout = min(self.settings.timeout, test.timeout or self.settings.timeout)
        steps: list[tuple[plan.Step, Result]] = []
        for step in invocation.before:
            done = self._step(step, invocation.env, timeout)
            steps.append((step, done))
            if done.timed_out or done.exit_code != 0:
                verdict = step_verdict(step, done)
                write_log(logs, invocation, done, verdict, steps)
                self._record(test, verdict, done, logs)
                return
        result = execute(invocation.argv, invocation.cwd, self.environment(invocation.env),
                         timeout, invocation.merge, invocation.stdin)
        post = [self._step(step, invocation.env, timeout) for step in invocation.after]
        steps += list(zip(invocation.after, post, strict=True))
        verdict = self._judged(test, invocation, result, post[0] if post else None)
        write_log(logs, invocation, result, verdict, steps)
        self._record(test, verdict, result, logs)

    def _judged(self, test: Test, invocation: plan.Invocation, result: Result,
                post: Result | None) -> judge.Verdict:
        if test.runner == "pyunittest" and invocation.junit is not None:
            verdict = cases.judge_pyunittest(test, result, invocation.junit)
        else:
            verdict = judge.judge(test, result, self.settings.root, invocation.cwd, post)
        if test.runner == "gtest":
            verdict.cases = cases.gtest_counts(result.stdout + result.stderr)
            verdict.failures = cases.gtest_failures(result.stdout + result.stderr)
        return verdict

    def _step(self, step: plan.Step, env: dict[str, str], timeout: float) -> Result:
        """A step run, or - a build of the build tree's - run once and its result shared."""
        if not step.shared:
            return self._execute(step, env, timeout)
        key = (str(step.cwd), *step.argv)
        with self.lock:
            event = self.building.get(key)
            first = event is None
            if event is None:
                event = self.building[key] = threading.Event()
        if first:
            try:
                self.built[key] = self._execute(step, env, timeout)
            finally:
                event.set()
        event.wait()
        return self.built.get(key) or Result(list(step.argv), None, False, 0.0, "",
                                             "the shared build did not finish")

    def _execute(self, step: plan.Step, env: dict[str, str], timeout: float) -> Result:
        step.cwd.mkdir(parents=True, exist_ok=True)
        return execute(step.argv, step.cwd, self.environment(env), timeout)

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


def step_verdict(step: plan.Step, result: Result) -> judge.Verdict:
    """A test whose build or pre-command failed: refused, timed out, or failed, and which."""
    if result.timed_out:
        return judge.Verdict("TIMEOUT", f"{step.what} was still running after "
                                        f"{result.runtime:.0f} s")
    refused = judge.refusal(result)
    if refused:
        return judge.Verdict("UNSUPPORTED", refused)
    return judge.Verdict("FAIL", f"{step.what} exited {result.exit_code}: "
                                 f"{judge.first_failure(result)}")


def write_log(logs: Path, invocation: plan.Invocation, result: Result, verdict: judge.Verdict,
              steps: Sequence[tuple[plan.Step, Result]] = ()) -> None:
    """The run as one readable file, each step around it first, and its diff beside it."""
    lines: list[str] = []
    for step, done in steps:
        lines += [f"$ cd {step.cwd}  # {step.what}", "$ " + " ".join(step.argv),
                  f"exit: {done.exit_code}", done.stdout, done.stderr]
    header = [f"$ cd {invocation.cwd}", "$ " + " ".join(invocation.argv),
              f"exit: {result.exit_code}  timed out: {result.timed_out}  {result.runtime} s",
              f"status: {verdict.status}  {verdict.reason}"]
    body = ["-- stdout --", result.stdout, "-- stderr --", result.stderr]
    (logs / "run.log").write_text("\n".join(lines + header + body), encoding="utf-8")
    if verdict.diff:
        (logs / "output.diff").write_text(verdict.diff, encoding="utf-8")


def _copy_to_build(files: Iterable[Path], directory: Path) -> None:
    """``COPY_TO_BUILDDIR``: each file into the test's build directory, unless already there."""
    for source in files:
        target = directory / source.name
        if source.is_file() and source.resolve() != target.resolve():
            shutil.copy2(source, target)


def _directories(tests: Iterable[Test]) -> list[str]:
    """The source directories the tests use, below ROOT's tree."""
    found: set[str] = set()
    for test in tests:
        found.update((test.directory, test.working_dir))
        found.update(str(Path(name).parent) for name in (*test.files, *test.copy))
        found.update(build[0] for build in test.builds)
    return sorted(name for name in found if name and name != "." and not name.startswith("/"))


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
