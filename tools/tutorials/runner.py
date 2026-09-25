"""Running one tutorial, in a directory of its own, and keeping what it made.

ROOT's CI runs every tutorial in one build directory, where each finds what
earlier ones wrote. Here each gets a fresh temporary directory instead,
holding a copy of the files beside it in the source tree (the macros it
``.L``s, the ``.dat`` it reads), links to the directories beside it, and the
files its dependencies produced - so a run is repeatable and runs can go in
parallel. The run is timed and bounded by a timeout that kills its whole
process group; then every file that is new or changed is its output, and is
copied out with its checksum before the directory is removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RunResult",
    "execute",
    "prepare",
    "snapshot",
    "produced",
    "collect",
    "run_in_sandbox",
    "IGNORED_OUTPUTS",
]

#: At most this much of a stream is kept: the head and the tail, the middle elided.
STREAM_LIMIT = 200_000

#: Outputs larger than this are checksummed but not copied out.
FILE_LIMIT = 200 * 1024 * 1024

#: Files a run leaves that are the interpreter's, not the tutorial's: ACLiC's
#: libraries and dictionaries, Python's caches, cling's history.
IGNORED_OUTPUTS = (
    ".so", ".d", ".pcm", ".dylib", ".pyc", ".o", "_ACLiC_dict.cxx", "_ACLiC_linkdef.h",
    ".root_hist", ".rootrc",
)  # fmt: skip

#: Directory names whose contents are never outputs.
IGNORED_DIRS = ("__pycache__", ".ipynb_checkpoints")


@dataclass
class RunResult:
    """What one run did: its exit, its streams, how long it took, what it wrote."""

    command: list[str]
    exit_code: int | None
    timed_out: bool
    runtime: float
    stdout: str
    stderr: str
    files: dict[str, str] = field(default_factory=dict)
    workdir: str = ""
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RunResult:
        names = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in names})

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> RunResult:
        return cls.from_json(json.loads(Path(path).read_text()))


def _clipped(text: str) -> str:
    if len(text) <= STREAM_LIMIT:
        return text
    half = STREAM_LIMIT // 2
    return f"{text[:half]}\n[... {len(text) - STREAM_LIMIT} characters elided ...]\n{text[-half:]}"


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def execute(command: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float) -> RunResult:
    """Run a command to its end or its timeout, whichever comes first."""
    start = time.monotonic()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as why:
        return RunResult(list(command), None, False, 0.0, "", f"cannot start: {why}")
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(process)
        out, err = process.communicate()
    return RunResult(
        command=list(command),
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        runtime=round(time.monotonic() - start, 3),
        stdout=_clipped(out.decode("utf-8", "replace")),
        stderr=_clipped(err.decode("utf-8", "replace")),
    )


def prepare(source: Path, workdir: Path, inputs: Iterable[Path] = ()) -> None:
    """A working directory for a tutorial: its neighbours, then its dependencies' outputs.

    Files beside the tutorial are copied, so it may change or overwrite them;
    directories beside it are linked, since tutorials read those and copying
    ``machine_learning/data`` for every run would dwarf the runs.
    """
    for entry in sorted(Path(source).iterdir()):
        target = workdir / entry.name
        if entry.name.startswith(".") or target.exists():
            continue
        if entry.is_dir():
            target.symlink_to(entry.resolve(), target_is_directory=True)
        elif entry.is_file():
            shutil.copy2(entry, target)
    for directory in inputs:
        _copy_tree(Path(directory), workdir)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    for path in sorted(source.rglob("*")):
        if path.is_file():
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _walk(workdir: Path) -> Iterable[Path]:
    """The regular files under a directory, not following links into the source tree."""
    for root, dirs, names in os.walk(workdir):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not os.path.islink(Path(root, d))]
        for name in names:
            path = Path(root, name)
            if path.is_file() and not path.is_symlink():
                yield path


def snapshot(workdir: Path) -> dict[str, tuple[int, int]]:
    """Each file's size and modification time, to tell afterwards what changed."""
    found = {}
    for path in _walk(workdir):
        stat = path.stat()
        found[path.relative_to(workdir).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return found


def _ignored(name: str) -> bool:
    return name.endswith(IGNORED_OUTPUTS) or "AutoDict_" in name


def produced(workdir: Path, before: Mapping[str, tuple[int, int]]) -> list[str]:
    """The files a run made or changed, interpreter droppings left out."""
    after = snapshot(workdir)
    return sorted(
        name for name, stamp in after.items() if before.get(name) != stamp and not _ignored(name)
    )


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(workdir: Path, names: Iterable[str], keep: Path) -> dict[str, str]:
    """Copy each output to ``keep`` and return its checksum; too-large ones only summed."""
    sums = {}
    for name in names:
        path = workdir / name
        sums[name] = checksum(path)
        if path.stat().st_size <= FILE_LIMIT:
            destination = keep / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    return sums


@dataclass(frozen=True)
class Sandbox:
    """Where a run's inputs come from and where its outputs go."""

    source: Path
    inputs: tuple[Path, ...]
    keep: Path
    scratch: Path | None = None


def run_in_sandbox(
    sandbox: Sandbox, command: Sequence[str], env: Mapping[str, str], timeout: float
) -> RunResult:
    """Prepare a fresh directory, run the command in it, keep its outputs, remove it.

    The command may name ``{workdir}``, which becomes the directory's path.
    """
    if sandbox.scratch is not None:
        sandbox.scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tutorial-", dir=sandbox.scratch) as scratch:
        workdir = Path(scratch).resolve()
        prepare(sandbox.source, workdir, sandbox.inputs)
        before = snapshot(workdir)
        filled = [part.replace("{workdir}", str(workdir)) for part in command]
        filled_env = {key: value.replace("{workdir}", str(workdir)) for key, value in env.items()}
        result = execute(filled, workdir, filled_env, timeout)
        if sandbox.keep.exists():
            shutil.rmtree(sandbox.keep)
        sandbox.keep.mkdir(parents=True)
        result.files = collect(workdir, produced(workdir, before), sandbox.keep)
        result.workdir = str(workdir)
    return result
