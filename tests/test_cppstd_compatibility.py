"""Generic facilities retain their contracts through the interpreter's old imports."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from xrdroot.cppstd import streams, strings


@pytest.mark.parametrize("name", ["streams", "strings"])
def test_compatibility_imports_share_every_public_object(name: str) -> None:
    import importlib

    generic = importlib.import_module(f"xrdroot.cppstd.{name}")
    legacy = importlib.import_module(f"xrdroot.cint.runtime.{name}")
    assert legacy.__all__ is generic.__all__
    for exported in generic.__all__:
        assert getattr(legacy, exported) is getattr(generic, exported), exported


def test_both_stream_paths_share_registered_formatters(monkeypatch: pytest.MonkeyPatch) -> None:
    from xrdroot.cint.runtime import streams as legacy

    class Token:
        pass

    assert legacy.FORMATTERS is streams.FORMATTERS
    monkeypatch.setitem(streams.FORMATTERS, Token, lambda output, value: output << "registered")
    output = legacy.ostringstream()
    output << Token() << streams.endl
    assert output.str() == "registered\n"
    assert legacy._formatter(Token()) is streams.FORMATTERS[Token]


def test_string_conversion_and_stream_formatting_work_without_the_interpreter() -> None:
    source = """
import importlib.abc
import sys

class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("xrdclient"):
            raise AssertionError(fullname)
        if fullname.startswith("xrdroot.") and not fullname.startswith("xrdroot.cppstd"):
            raise AssertionError(fullname)

sys.meta_path.insert(0, Boundary())
from xrdroot.cppstd import streams, strings
out = streams.ostringstream()
out << streams.boolalpha << True << " " << streams.fixed << streams.setprecision(2) << 1.25
assert out.str() == "true 1.25"
assert strings.to_string(1.25) == "1.250000"
assert strings.substr("abcd", 1, 2) == "bc"
assert not any(name.startswith("xrdclient") for name in sys.modules)
assert not any(name.startswith("xrdroot.") and not name.startswith("xrdroot.cppstd")
               for name in sys.modules)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", source], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_legacy_private_string_helpers_keep_their_original_identity() -> None:
    from xrdroot.cint.runtime import strings as legacy

    assert legacy._first is strings._first
    assert legacy._last is strings._last
    assert legacy._leading is strings._leading
    assert legacy._same is strings._same
