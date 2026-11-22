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


def test_calendar_time_is_brokendown_as_c_struct_tm_holds_it() -> None:
    """``localtime``/``gmtime`` give a ``struct tm``: ``tm_year`` counts from 1900 and
    ``tm_mon`` from January being 0, which is why a program adds 1900 to the year it prints,
    and ``strftime`` spells it. A ``time_t`` here may be a cell, as ``gmtime(&t)`` hands it."""
    from xrdroot.cppstd.cells import Cell
    from xrdroot.cppstd.chrono import formatted_time, gmtime, localtime, strftime

    utc = gmtime(Cell(0, "long"))
    assert (utc.tm_year, utc.tm_mon, utc.tm_mday) == (70, 0, 1)
    assert (utc.tm_hour, utc.tm_min, utc.tm_sec) == (0, 0, 0)
    assert (utc.tm_isdst, utc.tm_wday, utc.tm_yday) == (0, 3, 0)
    assert formatted_time("%Y-%m-%d %H:%M:%S", utc) == "1970-01-01 00:00:00"
    assert repr(utc) == "<tm 1970-01-01 00:00:00>"
    always = localtime(0)
    assert isinstance(always.tm_year, int)
    # A char[] of text is given the formatted time back; a buffer is written in place.
    assert strftime("", 6, "%Y", utc) == "1970"
    buffer = bytearray(8)
    assert strftime(buffer, 8, "%H:%M", utc) == 5
    assert bytes(buffer).split(b"\0")[0] == b"00:00"


def test_a_time_that_points_nowhere_is_the_clock_s_own_now(monkeypatch) -> None:
    """``localtime(NULL)`` and ``gmtime(NULL)`` are the clock's now, as C's are: a macro
    writes ``strftime(buf, 64, "%H:%M:%S", localtime(0))`` for 1970 and nothing at all for
    the minute it is running in."""
    from xrdroot.cppstd import chrono

    monkeypatch.setattr(chrono.time, "time", lambda: 86400.0)
    assert chrono.formatted_time("%Y-%m-%d %H:%M", chrono.gmtime()) == "1970-01-02 00:00"
    assert chrono.formatted_time("%Y", chrono.localtime(None))[:2] == "19"
    assert chrono._when(None) == 86400.0
