"""The C++ translator's own tests: each leaves the runtime's shared state as it found it.

A program's declarations are kept for later lines, a host may adopt its
exceptions into GoogleTest's, and output streams keep their settings; a test
here that changes any of them must not change what the next one sees. These
tests run with nothing above :mod:`xrdroot.pycxx` imported, so none of them
may need a library past the standard one.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from xrdroot.pycxx import cache
from xrdroot.pycxx.runtime import streams
from xrdroot.pycxx.runtime.gtest import exceptions, internal, runner
from xrdroot.pycxx.runtime.namespace import DECLARED


@pytest.fixture(autouse=True)
def _runtime_restored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    declared = dict(DECLARED)
    kinds = (exceptions.INTERNAL, exceptions.LIBRARY, dict(exceptions.CAUGHT),
             list(exceptions.REFUSAL), list(runner.AFTER_TEST), dict(vars(internal)))  # fmt: skip
    outputs = [(stream, dict(vars(stream))) for stream in (streams.cout, streams.cerr)]
    try:
        yield
    finally:
        DECLARED.clear()
        DECLARED.update(declared)
        exceptions.INTERNAL, exceptions.LIBRARY = kinds[0], kinds[1]
        exceptions.CAUGHT.clear()
        exceptions.CAUGHT.update(kinds[2])
        exceptions.REFUSAL[:] = kinds[3]
        runner.AFTER_TEST[:] = kinds[4]
        vars(internal).clear()
        vars(internal).update(kinds[5])
        for stream, saved in outputs:
            vars(stream).clear()
            vars(stream).update(saved)
