"""Standard utilities keep the same factories through ROOT and interpreter adapters."""

from __future__ import annotations

import importlib

import pytest

from xrdroot.cppstd import threads


@pytest.mark.parametrize(("new", "old"), [
    ("chrono", "pycxx.runtime.chrono"), ("threads", "pycxx.runtime.threads"),
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


def test_a_thread_moved_into_another_is_the_one_that_runs():
    """C++ has only std::thread's move, and roottest's ttree_write moves one into make_shared.

    Treating the argument as a callable would run nothing and say 'thread object is not callable'.
    """
    ran = []
    first = threads.thread(ran.append, 1)
    second = threads.thread(first)
    second.join()
    assert ran == [1]
    assert first.joinable() is False


def test_standard_durations_are_shared_with_interpreter_thread_calls(monkeypatch) -> None:
    from xrdroot.cppstd import threads
    from xrdroot.cppstd.chrono import chrono
    from xrdroot.pycxx.runtime.threads import this_thread

    observed = []
    monkeypatch.setattr(threads.time, "sleep", observed.append)
    this_thread.sleep_for(chrono.milliseconds(250))
    assert observed == [0.25]
