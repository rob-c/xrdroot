"""``std::atomic`` of a translation: reading one hands the interpreter to another thread.

C++'s ``while (counter < kNThreads) {}`` is a busy wait, and on a machine with a
core to spare that is all it costs. Here every C++ thread is a Python thread of
one interpreter: the spinning one holds the GIL, and the thread whose work it is
waiting for needs that GIL to do the work - CPython hands it over only at its
switch interval. A test whose workers finish in a millisecond then takes seconds
per barrier, and ``roottest/root/multicore/testSetAddressLoop`` - nine workers
between two such spins, thirty-three times - outlasts the harness's 300 s.

So the reads a loop condition makes, truthiness and the comparisons, give the
interpreter up with ``time.sleep(0)``, as a C++ spin gives up its core. This is
:class:`xrdroot.cppstd.threads.atomic` and nothing else; the translation's
``from xrdroot.pycint.runtime import *`` binds it over the plain one.
"""

from __future__ import annotations

import time
from typing import Any

from ...cppstd.threads import atomic as _atomic

__all__ = ["atomic"]


def _shared() -> None:
    """Let whoever else wants the interpreter have it, before looking at the value again."""
    time.sleep(0)


class atomic(_atomic):
    """``std::atomic<T>`` of a translation: a read of one shares the interpreter."""

    def __bool__(self) -> bool:
        """``operator bool()``: whether the value is true - what ``while (flag) {}`` asks."""
        _shared()
        return bool(self._value)

    def load(self) -> Any:
        _shared()
        return super().load()

    def __eq__(self, other: object) -> bool:
        _shared()
        return super().__eq__(other)

    def __lt__(self, other: Any) -> bool:
        _shared()
        return super().__lt__(other)

    def __gt__(self, other: Any) -> bool:
        _shared()
        return bool(self._value > other)

    def __le__(self, other: Any) -> bool:
        _shared()
        return bool(self._value <= other)

    def __ge__(self, other: Any) -> bool:
        _shared()
        return bool(self._value >= other)
