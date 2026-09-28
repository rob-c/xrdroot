"""``<thread>``, ``<mutex>``, ``<atomic>`` and ``<chrono>``: a macro's threads, run as Python's."""

from __future__ import annotations

import datetime
import threading

import pytest

from cintfake import fake
from xrdroot.cint import translate
from xrdroot.cint.execute import run_source
from xrdroot.cint.runtime import (
    Duration,
    atomic,
    chrono,
    condition_variable,
    mutex,
    recursive_mutex,
    this_thread,
    thread,
    unique_lock,
)

SOURCE = r"""
std::mutex m; std::atomic<int> n{0};
void work(int k) { std::lock_guard<std::mutex> g(m); n += k; }
void t() {
  auto s = std::chrono::high_resolution_clock::now();
  std::thread a(work, 1), b(work, std::ref(n) == 0 ? 2 : 2);
  a.join(); b.join();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto e = std::chrono::high_resolution_clock::now();
  double d = std::chrono::duration_cast<std::chrono::duration<double>>(e - s).count();
  printf("%d %d\n", (int)n.load(), d > 0);
}
"""


def test_threads_run_under_a_lock_guard_released_as_its_scope_ends(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = translate(SOURCE, "t.C")
    assert "g = lock_guard(m)" in text and "g._destruct()" in text
    assert "chrono.duration_cast['std::chrono::duration<double>'](e - s).count()" in text
    run_source(SOURCE, "t.C", root=fake())
    assert capsys.readouterr().out == "3 1\n"


def test_durations_count_in_their_ticks_and_casts_recount_them() -> None:
    assert chrono.milliseconds(1500).count() == 1500
    assert chrono.duration_cast["std::chrono::seconds"](chrono.milliseconds(1500)).count() == 1
    assert chrono.duration["double", "std::nano"](500).total_seconds() == pytest.approx(5e-7)
    assert chrono.duration(2).count() == 2.0
    later = chrono.seconds(1) + chrono.milliseconds(500) - datetime.timedelta(seconds=0.5)
    assert later == chrono.seconds(1) and chrono.seconds(1) < chrono.minutes(1) and later == 1.0
    assert "Duration(" in repr(later) and chrono.hours(1).count() == 1
    start = chrono.system_clock.now()
    assert start.time_since_epoch().count() > 0
    with pytest.raises(AttributeError, match="std::chrono has no weeks"):
        chrono.weeks  # noqa: B018


def test_an_atomic_changes_whole_and_reads_as_its_value() -> None:
    value = atomic(3)
    assert value.fetch_add(2) == 3 and value.fetch_sub(1) == 5
    value -= 1
    value.store(7)
    assert int(value) == 7 and float(value) == 7.0 and value + 1 == 8
    assert value == 7 and value < 8 and repr(value) == "atomic(7)"
    assert [0, 1, 2, 3, 4, 5, 6, 7][value] == 7


def test_mutexes_and_locks_are_taken_and_given_back() -> None:
    plain, again = mutex(), recursive_mutex()
    assert plain.try_lock() and not plain.try_lock()
    plain.unlock()
    held = unique_lock(plain, again)
    assert held.owns_lock()
    held.unlock()
    held._destruct()
    assert plain.try_lock()


def test_a_condition_variable_wakes_a_waiter_once_its_predicate_holds() -> None:
    lock, ready, seen = mutex(), condition_variable(), []
    state = {"go": False}

    def waiter() -> None:
        held = unique_lock(lock)
        ready.wait(held, lambda: state["go"])
        seen.append("woke")
        held._destruct()

    other = thread(waiter)
    this_thread.sleep_for(0.01)
    with threading.Lock():
        state["go"] = True
    ready.notify_one()
    other.join()
    assert seen == ["woke"] and not other.joinable()


def test_a_notify_between_the_predicate_and_the_wait_is_not_missed() -> None:
    lock, ready, calls = mutex(), condition_variable(), []

    def predicate() -> bool:
        calls.append(True)
        if len(calls) == 1:
            ready.notify_all()
            return False
        return True

    held = unique_lock(lock)
    ready.wait(held, predicate)
    assert len(calls) == 2 and held.owns_lock()


def test_a_wait_with_no_predicate_returns_at_the_first_notify() -> None:
    lock, ready = mutex(), condition_variable()
    woke = []

    def waiter() -> None:
        held = unique_lock(lock)
        ready.wait(held)
        woke.append(True)
        held.unlock()

    other = thread(waiter)
    while not woke:
        ready.notify_all()
        this_thread.sleep_for(Duration(1, 1e-3))
    other.join()


def test_a_thread_has_an_id_and_a_detached_or_empty_one_has_none() -> None:
    worker = thread(lambda: None)
    assert worker.get_id() != 0
    worker.detach()
    assert worker.get_id() == 0 and thread().get_id() == 0
    thread().join()
    assert this_thread.get_id() == threading.get_ident()
    getattr(this_thread, "yield")()
