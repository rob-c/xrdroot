"""``std::chrono``: durations that count in their units, and the clocks that time things.

A :class:`Duration` is a length of time and the tick it counts in -
``std::chrono::milliseconds(20).count()`` is ``20`` - and a clock's
``now()`` a :class:`TimePoint`, whose difference is a duration counted in
nanoseconds, as ``high_resolution_clock``'s is. ``duration_cast<T>(d)``
recounts ``d`` in ``T``'s ticks, truncating where ``T`` counts in integers.
Durations sleep: ``std::this_thread::sleep_for`` takes their seconds.
"""

from __future__ import annotations

import datetime
import re
import time
from typing import Any

__all__ = ["chrono", "Duration", "TimePoint"]

#: ``std::ratio``'s names, and the seconds one tick of each is.
RATIOS = {"nano": 1e-9, "micro": 1e-6, "milli": 1e-3, "centi": 1e-2, "deci": 1e-1}

#: The named durations, the seconds a tick of each is, and that they count in integers.
NAMED = {
    "nanoseconds": 1e-9,
    "microseconds": 1e-6,
    "milliseconds": 1e-3,
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
}


class Duration:
    """A length of time, counted in ticks of ``period`` seconds - integers unless ``real``."""

    def __init__(self, count: Any = 0, period: float = 1.0, real: bool = True) -> None:
        if isinstance(count, (Duration, datetime.timedelta)):
            count = _seconds(count) / period
        self.seconds = float(count) * period
        self.period = period
        self.real = real

    def count(self) -> Any:
        ticks = self.seconds / self.period
        return ticks if self.real else int(ticks)

    def total_seconds(self) -> float:
        return self.seconds

    def __add__(self, other: Any) -> Duration:
        return Duration((self.seconds + _seconds(other)) / self.period, self.period, self.real)

    def __sub__(self, other: Any) -> Duration:
        return Duration((self.seconds - _seconds(other)) / self.period, self.period, self.real)

    def __eq__(self, other: object) -> bool:
        return self.seconds == _seconds(other)

    def __lt__(self, other: Any) -> bool:
        return self.seconds < _seconds(other)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"Duration({self.count()!r} x {self.period!r}s)"


def _seconds(value: Any) -> float:
    if isinstance(value, (Duration, datetime.timedelta)):
        return float(value.total_seconds())
    return float(value)


class TimePoint:
    """A clock's ``now()``: subtracting another gives the nanoseconds between them."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __sub__(self, other: TimePoint) -> Duration:
        return Duration((self.seconds - other.seconds) / 1e-9, 1e-9, real=False)

    def time_since_epoch(self) -> Duration:
        return Duration(self.seconds / 1e-9, 1e-9, real=False)


class _Clock:
    def __init__(self, source: Any) -> None:
        self._source = source

    def now(self) -> TimePoint:
        return TimePoint(self._source())


class _Kind:
    """A duration type, ``milliseconds`` or ``duration<double, std::milli>``: it makes one."""

    def __init__(self, period: float, real: bool) -> None:
        self.period = period
        self.real = real

    def __call__(self, count: Any = 0) -> Duration:
        return Duration(count, self.period, self.real)


def _kind(spelled: Any) -> _Kind:
    """The duration type a template argument spells: its tick, and whether it counts in reals."""
    text = str(spelled).replace(" ", "")
    for name, period in NAMED.items():
        if text.endswith(name):
            return _Kind(period, False)
    ratio = re.search(r",(?:std::)?(\w+)>?$", text)
    period = RATIOS.get(ratio.group(1), 1.0) if ratio else 1.0
    return _Kind(period, "double" in text or "float" in text)


class _Durations:
    """``std::chrono::duration<Rep, Period>``: subscripted with its arguments, it is a type."""

    def __getitem__(self, args: Any) -> _Kind:
        items = args if isinstance(args, tuple) else (args,)
        return _kind("duration<" + ",".join(str(item) for item in items) + ">")

    def __call__(self, count: Any = 0) -> Duration:
        return Duration(count)


class _Casts:
    """``std::chrono::duration_cast<T>``: subscripted with ``T``, it recounts a duration."""

    def __getitem__(self, target: Any) -> _Kind:
        return _kind(target)


class _Chrono:
    """``std::chrono``: its durations, casts and clocks."""

    duration = _Durations()
    duration_cast = _Casts()
    high_resolution_clock = _Clock(time.perf_counter)
    steady_clock = _Clock(time.perf_counter)
    system_clock = _Clock(time.time)

    def __getattr__(self, name: str) -> _Kind:
        if name in NAMED:
            return _Kind(NAMED[name], False)
        raise AttributeError(f"std::chrono has no {name} this runtime knows")


chrono = _Chrono()
