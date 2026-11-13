"""Running one command: to its end or its timeout, with its output caught as the driver catches it.

``RootTestDriver.cmake`` gives a test that checks both streams one variable
for both - CMake's ``execute_process`` then reads them through one pipe, so
what the reference holds is the two interleaved as they were written. A
command run with ``merge`` does the same: standard error goes into standard
output's pipe. The timeout kills the whole process group, so a test's own
children do not outlive it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = ["Result", "execute", "STREAM_LIMIT"]

#: At most this much of a stream is kept: its head and its tail, the middle elided.
STREAM_LIMIT = 400_000


@dataclass
class Result:
    """What one command did: its exit (``None`` when it was killed), its streams, its time."""

    command: list[str]
    exit_code: int | None
    timed_out: bool
    runtime: float
    stdout: str
    stderr: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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


def execute(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
    merge: bool = False,
    stdin: Path | None = None,
) -> Result:
    """Run ``command`` in ``cwd``; with ``merge``, its standard error joins its standard output."""
    start = time.monotonic()
    try:
        with open(stdin, "rb") if stdin is not None else open(os.devnull, "rb") as source:
            process = subprocess.Popen(
                list(command), cwd=str(cwd), env=dict(env), stdin=source,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if merge else subprocess.PIPE,
                start_new_session=True,
            )
    except OSError as why:
        return Result(list(command), None, False, 0.0, "", f"cannot start: {why}")
    timed_out = False
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill(process)
        out, err = process.communicate()
    return Result(
        command=list(command),
        exit_code=None if timed_out else process.returncode,
        timed_out=timed_out,
        runtime=round(time.monotonic() - start, 3),
        stdout=_clipped(out.decode("utf-8", "replace")),
        stderr=_clipped((err or b"").decode("utf-8", "replace")),
    )
