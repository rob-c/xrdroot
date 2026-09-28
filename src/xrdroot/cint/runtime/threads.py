"""``<thread>``, ``<mutex>``, ``<atomic>`` and ``<condition_variable>`` over Python's threading.

``std::thread t(f, a)`` starts ``f(a)`` at once, as C++'s does, and
``t.join()`` waits for it. A ``std::mutex`` is a lock; ``std::lock_guard``
and ``std::unique_lock`` take it as they are made and give it back in
``_destruct``, which the translation calls as the guard's scope ends - the
same as the destructor of one of the macro's own classes. ``std::atomic``
holds a value that ``++``, ``+=`` and ``fetch_add`` change under a lock.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any

__all__ = [
    "thread",
    "mutex",
    "recursive_mutex",
    "lock_guard",
    "unique_lock",
    "scoped_lock",
    "atomic",
    "condition_variable",
    "this_thread",
    "ref",
    "cref",
]


def ref(value: Any) -> Any:
    """``std::ref(x)``: ``x`` itself, which a Python call shares anyway."""
    return value


cref = ref


class thread:
    """``std::thread(f, args...)``: ``f(args...)`` running, from the moment it is made."""

    def __init__(self, function: Any = None, *args: Any) -> None:
        self._thread: threading.Thread | None = None
        if function is not None:
            self._thread = threading.Thread(target=function, args=args, daemon=True)
            self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def joinable(self) -> bool:
        return self._thread is not None

    def detach(self) -> None:
        self._thread = None

    def get_id(self) -> int:
        return (self._thread.ident or 0) if self._thread is not None else 0


class mutex:
    """``std::mutex``: ``lock()``, ``unlock()``, ``try_lock()``."""

    def __init__(self) -> None:
        self._lock: Any = threading.Lock()

    def lock(self) -> None:
        self._lock.acquire()

    def unlock(self) -> None:
        self._lock.release()

    def try_lock(self) -> bool:
        return bool(self._lock.acquire(blocking=False))


class recursive_mutex(mutex):
    """``std::recursive_mutex``: a lock the thread holding it may take again."""

    def __init__(self) -> None:
        self._lock = threading.RLock()


class unique_lock:
    """``std::unique_lock(m)`` and ``std::lock_guard(m)``: ``m`` held until ``_destruct``."""

    def __init__(self, *mutexes: Any) -> None:
        self.mutexes = mutexes
        self.owns = False
        self.lock()

    def lock(self) -> None:
        for held in self.mutexes:
            held.lock()
        self.owns = True

    def unlock(self) -> None:
        for held in reversed(self.mutexes):
            held.unlock()
        self.owns = False

    def owns_lock(self) -> bool:
        return self.owns

    def _destruct(self) -> None:
        if self.owns:
            self.unlock()


lock_guard = unique_lock
scoped_lock = unique_lock


class atomic:
    """``std::atomic<T>``: a value read, written and changed each time whole."""

    def __init__(self, value: Any = 0) -> None:
        self._value = value
        self._lock = threading.Lock()

    def load(self) -> Any:
        return self._value

    def store(self, value: Any) -> None:
        with self._lock:
            self._value = value

    def fetch_add(self, delta: Any) -> Any:
        with self._lock:
            old = self._value
            self._value = old + delta
        return old

    def fetch_sub(self, delta: Any) -> Any:
        return self.fetch_add(-delta)

    def __iadd__(self, delta: Any) -> atomic:
        self.fetch_add(delta)
        return self

    def __isub__(self, delta: Any) -> atomic:
        self.fetch_add(-delta)
        return self

    def __int__(self) -> int:
        return int(self._value)

    __index__ = __int__

    def __float__(self) -> float:
        return float(self._value)

    def __add__(self, other: Any) -> Any:
        return self._value + other

    def __eq__(self, other: object) -> bool:
        return bool(self._value == other)

    def __lt__(self, other: Any) -> bool:
        return bool(self._value < other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"atomic({self._value!r})"


class condition_variable:
    """``std::condition_variable``: ``wait(lock[, predicate])`` until notified."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._generation = 0

    def wait(self, lock: unique_lock, predicate: Any = None) -> None:
        while True:
            # The generation is read before the predicate, so a notify between them counts.
            with self._condition:
                seen = self._generation
            if predicate is not None and predicate():
                return
            lock.unlock()
            with self._condition:
                self._condition.wait_for(functools.partial(self._moved_on, seen))
            lock.lock()
            if predicate is None:
                return

    def _moved_on(self, seen: int) -> bool:
        return self._generation != seen

    def notify_all(self) -> None:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    notify_one = notify_all


class _ThisThread:
    """``std::this_thread``: ``sleep_for(duration)``, ``get_id()``, ``yield``."""

    @staticmethod
    def sleep_for(duration: Any) -> None:
        seconds = duration.total_seconds() if hasattr(duration, "total_seconds") else duration
        time.sleep(max(float(seconds), 0.0))

    @staticmethod
    def get_id() -> int:
        return threading.get_ident()

    @staticmethod
    def _yield() -> None:
        time.sleep(0)


# ``std::this_thread::yield()`` is written ``getattr(this_thread, 'yield')()``: a keyword.
setattr(_ThisThread, "yield", staticmethod(_ThisThread._yield))
this_thread = _ThisThread()
