"""Run the tutorials under ROOT and under xrdroot, and judge each pair of runs.

ROOT's runs are the slow half and do not change while ROOT does not, so
each is cached on disk (``~/.cache/xrdroot-tutorials/oracle/<ROOT
version>/<key>/``) under a key hashing the tutorial, every file beside it,
how CI runs it, and its dependencies' keys - an edit to any of those, or a
new ROOT, is a new key. xrdroot's runs are never cached: they are what is
being measured.

Tutorials run in waves, each wave's dependencies all in earlier ones, and
each wave through a pool of workers. A dependent is given the outputs ROOT's
run of its dependency wrote - even on the xrdroot side - so that each
tutorial is judged on its own, not on whether ``hsimple.C`` ran first.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from . import classify
from .catalog import Tutorial, closure, levels
from .compare import Tolerance, compare_files, compare_streams, extra_files
from .environment import Oracle
from .runner import RunResult, Sandbox, run_in_sandbox

__all__ = ["Settings", "Outcome", "Harness", "DEFAULT_CACHE", "oracle_key", "selected"]

#: Where ROOT's cached runs and the latest results live, unless told otherwise.
DEFAULT_CACHE = Path(os.environ.get("XRDROOT_TUTORIALS_CACHE", "~/.cache/xrdroot-tutorials"))

#: Bumped whenever how an oracle run is made changes, so old cache entries stop matching.
ORACLE_SCHEMA = "oracle/1"

#: The reason given for every tutorial ROOT ran when xrdroot was not asked to.
ORACLE_ONLY = "xrdroot not run (--oracle-only)"

#: Environment entries of CI's that name its build tree, replaced by the harness's own.
BUILD_TREE_VARIABLES = ("PYTHONPATH", "ROOT_INCLUDE_PATH")

#: The xrdroot ``run`` subcommand's presence, asked of the xrdroot under test.
_HAS_RUN = "import sys, xrdroot.cli as c; sys.exit(0 if 'run' in c.COMMANDS else 3)"
_HAS_PYROOT = "import importlib.util as u, sys; sys.exit(0 if u.find_spec('xrdroot.pyroot') else 3)"
#: What runs a ``.py`` tutorial against xrdroot.pyroot when there is no ``run`` yet.
_PY_SHIM = (
    "import sys, runpy, xrdroot.pyroot as R; sys.modules['ROOT'] = R; "
    "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
)


@dataclass(frozen=True)
class Settings:
    """How a run of the harness is to go."""

    tutorials: Path
    cache: Path = DEFAULT_CACHE
    out: Path | None = None
    jobs: int = 4
    timeout: float = 300.0
    refresh_oracle: bool = False
    network: bool = False
    xrdroot_python: str = sys.executable
    xrdroot_src: Path | None = None
    tolerance: Tolerance = field(default_factory=Tolerance)
    only: tuple[str, ...] = ()
    run_oracle: bool = True
    run_xrdroot: bool = True

    @property
    def results(self) -> Path:
        return self.out if self.out is not None else self.cache / "results"


@dataclass
class Outcome:
    """One side's run of one tutorial, and where its outputs were kept."""

    result: RunResult
    files: Path
    cached: bool = False
    key: str = ""


def selected(paths: Iterable[str], only: Sequence[str]) -> list[str]:
    """The paths under any of the ``--only`` prefixes (a directory, or a file); all if none."""
    prefixes = [prefix.strip("/") for prefix in only if prefix.strip("/")]
    if not prefixes:
        return list(paths)
    return [p for p in paths if any(p == pre or p.startswith(pre + "/") for pre in prefixes)]


# --- keys ------------------------------------------------------------------

_directory_sums: dict[Path, str] = {}
_sums_lock = threading.Lock()


def directory_sum(directory: Path) -> str:
    """A checksum of the files beside a tutorial - its inputs - and the names of its subdirs."""
    with _sums_lock:
        if directory in _directory_sums:
            return _directory_sums[directory]
    digest = hashlib.sha256()
    for entry in sorted(directory.iterdir()):
        digest.update(entry.name.encode())
        if entry.is_file():
            digest.update(hashlib.sha256(entry.read_bytes()).digest())
    found = digest.hexdigest()
    with _sums_lock:
        _directory_sums[directory] = found
    return found


def oracle_key(
    tutorial: Tutorial, tutorials: Path, version: str, dependency_keys: Sequence[str]
) -> str:
    """The cache key of ROOT's run: what is run, how, with what, by which ROOT."""
    source = tutorials / tutorial.path
    digest = hashlib.sha256()
    parts = [ORACLE_SCHEMA, version, tutorial.path, " ".join(tutorial.flags), tutorial.suffix]
    parts += [*tutorial.environment, directory_sum(source.parent), *dependency_keys]
    for part in parts:
        digest.update(part.encode() + b"\0")
    return digest.hexdigest()


# --- commands --------------------------------------------------------------


def oracle_command(tutorial: Tutorial, oracle: Oracle) -> list[str]:
    name = PurePosixPath(tutorial.path).name
    if tutorial.language == "py":
        return [oracle.python or "python3", name]
    flags = list(tutorial.flags) or ["-b", "-q"]
    return [oracle.root, *flags, *([] if "-l" in flags else ["-l"]), name + tutorial.suffix]


def ci_environment(tutorial: Tutorial) -> dict[str, str]:
    """This process's environment with CI's for the tutorial laid over it."""
    env = {key: value for key, value in os.environ.items() if key not in BUILD_TREE_VARIABLES}
    for entry in tutorial.environment:
        key, _, value = entry.partition("=")
        if key and key not in BUILD_TREE_VARIABLES:
            env[key] = value
    env.setdefault("MPLBACKEND", "AGG")
    env.setdefault("ROOT_BATCH", "1")
    return env


def oracle_environment(tutorial: Tutorial, oracle: Oracle, tutorials: Path) -> dict[str, str]:
    env = ci_environment(tutorial)
    env["PYTHONPATH"] = oracle.pythonpath
    env["ROOT_INCLUDE_PATH"] = os.pathsep.join(["{workdir}", str(tutorials / "io" / "tree")])
    env["PATH"] = os.pathsep.join([str(Path(oracle.root).parent), env.get("PATH", "")])
    return env


# --- the harness -----------------------------------------------------------


class Harness:
    """One run of the harness over a catalogue: ROOT's side, xrdroot's, the verdicts."""

    def __init__(
        self,
        settings: Settings,
        tutorials: Mapping[str, Tutorial],
        oracle: Oracle | None,
        log: Callable[[str], None] = lambda line: print(line, file=sys.stderr, flush=True),
    ) -> None:
        self.settings = settings
        self.tutorials = dict(tutorials)
        self.oracle = oracle
        self.log = log
        self.oracles: dict[str, Outcome] = {}
        self.xrdroots: dict[str, Outcome] = {}
        self.skips: dict[str, str] = {}
        self._lock = threading.Lock()
        self._done = 0
        self._total = 0
        self.has_run = self._probe(_HAS_RUN)
        self.has_pyroot = self._probe(_HAS_PYROOT)

    # probing

    def _xrdroot_env(self, tutorial: Tutorial | None = None) -> dict[str, str]:
        env = ci_environment(tutorial) if tutorial is not None else dict(os.environ)
        if self.settings.xrdroot_src is not None:
            env["PYTHONPATH"] = str(self.settings.xrdroot_src)
        # ROOT answers gROOT->GetTutorialDir() with the tutorials it was
        # installed with, which a hundred-odd tutorials read their data from;
        # xrdroot has none of its own, so it is told where the ones being run
        # are, and both sides read the same files.
        env.setdefault("ROOT_TUTORIAL_DIR", str(self.settings.tutorials))
        return env

    def _probe(self, code: str) -> bool:
        try:
            done = subprocess.run(
                [self.settings.xrdroot_python, "-c", code],
                env=self._xrdroot_env(),
                capture_output=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return done.returncode == 0

    def xrdroot_commit(self) -> str:
        src = self.settings.xrdroot_src
        if src is None:
            return ""
        try:
            done = subprocess.run(
                ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return done.stdout.strip()

    # what runs

    def skip_reason(self, tutorial: Tutorial) -> str | None:
        """Why ROOT itself does not run this tutorial here, or None."""
        if tutorial.vetoed is not None:
            return tutorial.vetoed
        if "needs_network" in tutorial.labels and not self.settings.network:
            return "needs the network"
        return self._oracle_lacks(tutorial) if self.oracle is not None else None

    def _oracle_lacks(self, tutorial: Tutorial) -> str | None:
        """What this ROOT's Python lacks for the tutorial: PyROOT, or a package."""
        assert self.oracle is not None
        if tutorial.language == "py" and self.oracle.python is None:
            return "no Python with PyROOT here"
        missing = [name for name in tutorial.python_deps if not self.oracle.has_module(name)]
        return f"needs Python package {', '.join(missing)}" if missing else None

    def xrdroot_command(self, tutorial: Tutorial) -> list[str] | None:
        """How xrdroot runs the tutorial - or None when the xrdroot under test cannot."""
        name = PurePosixPath(tutorial.path).name
        python = self.settings.xrdroot_python
        if self.has_run:  # ROOT's -n, no logon macros, is the one flag a run can differ by
            quiet = ["-n"] if tutorial.language == "cxx" and "-n" in tutorial.flags else []
            return [python, "-m", "xrdroot", "run", *quiet, name]
        if tutorial.language == "py" and self.has_pyroot:
            return [python, "-c", _PY_SHIM, name]
        return None

    # running

    def _progress(self, side: str, path: str, result: RunResult, cached: bool) -> None:
        with self._lock:
            self._done += 1
            done, total = self._done, self._total
        how = "cached" if cached else f"{result.runtime:.1f}s"
        status = "timeout" if result.timed_out else f"exit {result.exit_code}"
        self.log(f"[{done}/{total}] {side} {path}: {status} ({how})")

    def _inputs(self, path: str, side: dict[str, Outcome]) -> tuple[Path, ...]:
        before = closure(self.tutorials, [path])[:-1]
        return tuple(side[d].files for d in before if d in side)

    def _oracle_one(self, path: str) -> None:
        assert self.oracle is not None
        tutorial = self.tutorials[path]
        before = closure(self.tutorials, [path])[:-1]
        keys = [self.oracles[d].key for d in before if d in self.oracles]
        key = oracle_key(tutorial, self.settings.tutorials, self.oracle.version, keys)
        entry = self.settings.cache / "oracle" / self.oracle.version / key
        if (entry / "result.json").is_file() and not self.settings.refresh_oracle:
            result = RunResult.load(entry / "result.json")
            self.oracles[path] = Outcome(result, entry / "files", True, key)
            self._progress("ROOT", path, result, True)
            return
        sandbox = Sandbox(
            (self.settings.tutorials / path).parent,
            self._inputs(path, self.oracles),
            entry / "files",
        )
        env = oracle_environment(tutorial, self.oracle, self.settings.tutorials)
        result = run_in_sandbox(
            sandbox, oracle_command(tutorial, self.oracle), env, self._timeout(tutorial)
        )
        result.save(entry / "result.json")
        self.oracles[path] = Outcome(result, entry / "files", False, key)
        self._progress("ROOT", path, result, False)

    def _timeout(self, tutorial: Tutorial) -> float:
        return self.settings.timeout * (3 if "longtest" in tutorial.labels else 1)

    def _xrdroot_one(self, path: str) -> None:
        tutorial = self.tutorials[path]
        command = self.xrdroot_command(tutorial)
        if command is None:
            return
        keep = self.settings.results / "xrdroot" / path
        inputs = self._inputs(path, self.oracles) or self._inputs(path, self.xrdroots)
        sandbox = Sandbox((self.settings.tutorials / path).parent, inputs, keep / "files")
        result = run_in_sandbox(
            sandbox, command, self._xrdroot_env(tutorial), self._timeout(tutorial)
        )
        result.save(keep / "result.json")
        self.xrdroots[path] = Outcome(result, keep / "files")
        self._progress("xrdroot", path, result, False)

    def _waves(self, paths: Sequence[str], work: Callable[[str], None]) -> None:
        with self._lock:
            self._done, self._total = 0, len(paths)
        wanted = set(paths)
        with ThreadPoolExecutor(max_workers=max(1, self.settings.jobs)) as pool:
            for wave in levels(self.tutorials, paths):
                list(pool.map(work, [path for path in wave if path in wanted]))

    def run(self) -> list[dict[str, Any]]:
        """Both sides' runs of every selected tutorial, and a verdict on each."""
        chosen = selected(sorted(self.tutorials), self.settings.only)
        needed = self._unskipped(closure(self.tutorials, chosen))
        if self.oracle is not None and self.settings.run_oracle:
            self._waves(needed, self._oracle_one)
        wanted = set(chosen)
        runnable = [p for p in needed if p in wanted and self._worth_running(p)]
        if self.settings.run_xrdroot:
            self._waves(runnable, self._xrdroot_one)
        return [self.verdict(path) for path in chosen]

    def _unskipped(self, paths: Iterable[str]) -> list[str]:
        """The paths ROOT runs here; the others' reasons are kept in :attr:`skips`."""
        kept = []
        for path in paths:
            reason = self.skip_reason(self.tutorials[path])
            if reason is None:
                kept.append(path)
            else:
                self.skips[path] = reason
        return kept

    def _worth_running(self, path: str) -> bool:
        outcome = self.oracles.get(path)
        if outcome is None:
            return self.oracle is None or not self.settings.run_oracle
        tutorial = self.tutorials[path]
        return classify.oracle_failure(outcome.result, tutorial.passrc, tutorial.failregex) is None

    # judging

    def verdict(self, path: str) -> dict[str, Any]:
        tutorial = self.tutorials[path]
        oracle, mine = self.oracles.get(path), self.xrdroots.get(path)
        record: dict[str, Any] = {
            "path": path,
            "area": tutorial.area,
            "language": tutorial.language,
            "test": tutorial.test,
            "labels": [label for label in tutorial.labels if label != "tutorial"],
            "depends": list(tutorial.depends),
            "oracle": _side(oracle, passrc=tutorial.passrc),
            "xrdroot": _side(mine),
            "details": [],
            "images": [],
            "extra_files": [],
        }
        status, reason = self._judged(tutorial, oracle, mine, record)
        record["status"], record["reason"] = status, reason
        return record

    def _judged(
        self,
        tutorial: Tutorial,
        oracle: Outcome | None,
        mine: Outcome | None,
        record: dict[str, Any],
    ) -> tuple[str, str]:
        if tutorial.path in self.skips:
            return "SKIP", self.skips[tutorial.path]
        if oracle is not None:
            failed = classify.oracle_failure(oracle.result, tutorial.passrc, tutorial.failregex)
            if failed is not None:
                record["details"].append(_tail(oracle.result.stderr or oracle.result.stdout))
                return "ORACLE-FAIL", failed
        if mine is None:
            if not self.settings.run_xrdroot:
                return "SKIP", ORACLE_ONLY
            if self.xrdroot_command(tutorial) is None:
                return "FAIL", classify.NO_RUN
            return "ORACLE-FAIL", "no ROOT run to compare with"
        crashed = self._crash(tutorial, oracle, mine.result)
        if crashed is not None:
            record["details"].append(_tail(mine.result.stderr))
            return crashed
        if oracle is None:
            return "ORACLE-FAIL", "xrdroot ran; no ROOT run to compare with"
        return self._compared(tutorial, oracle, mine, record)

    def _crash(
        self, tutorial: Tutorial, oracle: Outcome | None, mine: RunResult
    ) -> tuple[str, str] | None:
        expected = oracle.result.exit_code if oracle is not None else tutorial.passrc
        broke = mine.timed_out or mine.exit_code is None or mine.exit_code < 0
        traced = classify.last_exception(mine.stderr) is not None
        if broke or traced or mine.exit_code not in (expected, 0):
            return tuple(classify.failure(mine))  # type: ignore[return-value]
        line = classify.matched_failure(mine.stdout + "\n" + mine.stderr, tutorial.failregex)
        if line is not None:
            return "FAIL", f"printed {classify._masked(line)!r}"
        return None

    def _compared(
        self, tutorial: Tutorial, oracle: Outcome, mine: Outcome, record: dict[str, Any]
    ) -> tuple[str, str]:
        found: list[tuple[str, str]] = []
        theirs, ours = oracle.result, mine.result
        if theirs.exit_code != ours.exit_code:
            found.append(("exit code", f"exit code {ours.exit_code}, ROOT {theirs.exit_code}"))
        paths = (self._paths(theirs), self._paths(ours))
        python = tutorial.language == "py"
        stdout = compare_streams(theirs.stdout, ours.stdout, paths, self.settings.tolerance, python)
        if stdout is not None:
            found.append(("stdout", f"stdout {stdout}"))
        for compared in compare_files(
            theirs.files, ours.files, (oracle.files, mine.files), self.settings.tolerance
        ):
            if compared.score is not None:
                record["images"].append({"file": compared.name, "score": compared.score})
            found.extend((_diff_kind(compared.kind, line), line) for line in compared.differences)
        record["extra_files"] = extra_files(theirs.files, ours.files)
        record["details"].extend(line for _, line in found)
        if not found:
            return "PASS", ""
        return "DIFF", found[0][0]

    def _paths(self, result: RunResult) -> dict[str, str]:
        """The directories a run's output names, each masked as what it is.

        ROOT's ``gROOT->GetTutorialDir()`` is its installation's own
        ``tutorials`` - not the checkout the harness was given, which is what
        xrdroot answers - so both are masked the same: as the tutorials.
        """
        paths = {result.workdir: "<workdir>", str(self.settings.tutorials): "<tutorials>"}
        if self.oracle is not None and self.oracle.rootsys:
            paths[self.oracle.rootsys] = "<rootsys>"
            paths[str(Path(self.oracle.rootsys) / "tutorials")] = "<tutorials>"
        return paths


def _diff_kind(kind: str, line: str) -> str:
    if "ROOT wrote it, xrdroot did not" in line:
        return f"missing output ({kind})"
    return {"root": "ROOT file differs", "image": "image differs"}.get(kind, f"{kind} file differs")


def _side(outcome: Outcome | None, **extra: Any) -> dict[str, Any] | None:
    if outcome is None:
        return None
    result = outcome.result
    return {
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "runtime": result.runtime,
        "cached": outcome.cached,
        "files": sorted(result.files),
        **extra,
    }


def _tail(text: str, lines: int = 12) -> str:
    kept = [line for line in text.splitlines() if line.strip()]
    return "\n".join(kept[-lines:])
