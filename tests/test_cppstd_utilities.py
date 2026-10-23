"""Standard utilities keep the same factories through ROOT and interpreter adapters."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(("new", "old"), [
    ("chrono", "cint.runtime.chrono"), ("threads", "cint.runtime.threads"),
    ("random", "pyroot.stlrandom"), ("bitset", "pyroot.bitset"),
])
def test_standard_utility_compatibility_paths_share_public_objects(new: str, old: str) -> None:
    generic = importlib.import_module(f"xrdroot.cppstd.{new}")
    legacy = importlib.import_module(f"xrdroot.{old}")
    assert generic.__all__ is legacy.__all__
    for name in generic.__all__:
        assert getattr(generic, name) is getattr(legacy, name), name


def test_standard_random_engines_continue_the_same_stream_across_import_paths() -> None:
    from xrdroot.cppstd.random import mt19937
    from xrdroot.pyroot.stlrandom import canonical

    engine = mt19937(5489)
    reference = mt19937(5489)
    assert canonical(engine) == canonical(reference)
    assert engine() == reference()


def test_standard_durations_are_shared_with_interpreter_thread_calls(monkeypatch) -> None:
    from xrdroot.cint.runtime.threads import this_thread
    from xrdroot.cppstd import threads
    from xrdroot.cppstd.chrono import chrono

    observed = []
    monkeypatch.setattr(threads.time, "sleep", observed.append)
    this_thread.sleep_for(chrono.milliseconds(250))
    assert observed == [0.25]
