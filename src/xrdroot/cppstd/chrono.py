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

__all__ = ["chrono", "c_time", "formatted_time", "gmtime", "localtime", "strftime", "tm",
           "Duration", "TimePoint"]  # fmt: skip

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


def c_time(where: Any = None) -> int:
    """C's ``time(&t)``: the seconds since 1970 began, stored where ``where`` points as well,
    when it points anywhere."""
    now = int(time.time())
    if where is not None and hasattr(where, "value"):
        where.value = now
    return now


class tm:
    """C's ``struct tm``: a broken-down calendar time, with ``strftime``'s fields.

    ``tm_year`` counts from 1900 and ``tm_mon`` from January being 0, as C's do - which is why
    a program writes ``tt->tm_year + 1900`` for the year it prints.
    """

    __slots__ = ("tm_sec", "tm_min", "tm_hour", "tm_mday", "tm_mon", "tm_year", "tm_wday",
                 "tm_yday", "tm_isdst")

    def __init__(self, fields: time.struct_time) -> None:
        self.tm_sec, self.tm_min, self.tm_hour = fields.tm_sec, fields.tm_min, fields.tm_hour
        self.tm_mday, self.tm_mon, self.tm_year = fields.tm_mday, fields.tm_mon - 1, \
            fields.tm_year - 1900
        self.tm_wday, self.tm_yday, self.tm_isdst = fields.tm_wday, fields.tm_yday - 1, \
            fields.tm_isdst

    def __repr__(self) -> str:
        return (f"<tm {self.tm_year + 1900}-{self.tm_mon + 1:02d}-{self.tm_mday:02d} "
                f"{self.tm_hour:02d}:{self.tm_min:02d}:{self.tm_sec:02d}>")


def _when(t: Any) -> float:
    """The seconds a ``time_t *`` points at, or the clock's own now when it points nowhere."""
    if t is None:
        return time.time()
    return float(t.value if hasattr(t, "value") else t)


def localtime(t: Any = None) -> tm:
    """``localtime``: the calendar time in the machine's own zone."""
    return tm(time.localtime(_when(t)))


def gmtime(t: Any = None) -> tm:
    """``gmtime``: the calendar time in UTC."""
    return tm(time.gmtime(_when(t)))


def formatted_time(fmt: Any, when: tm) -> str:
    """The time ``fmt`` spells, which ``strftime`` gives the character array it was handed."""
    return time.strftime(str(fmt), _struct_of(when))


def strftime(buffer: Any, size: Any, fmt: Any, when: tm) -> int:
    """``strftime(buf, size, fmt, tt)``: the time formatted into the character array ``size``
    allows, and how many characters were written.

    A ``char buf[N]`` is a Python ``str`` here, which no write reaches in place, so the text is
    what this returns and the translation gives it to the variable - the caller hands the
    buffer over and keeps what comes back.
    """
    text = formatted_time(fmt, when)[: max(int(size) - 1, 0)]
    where = buffer.value if hasattr(buffer, "value") else buffer
    if isinstance(where, str) or where is None:
        # A char[] the translation holds as text: the caller gives the variable what this
        # returns, as it does for ``sprintf`` and ``fgets``.
        return text  # type: ignore[return-value]
    for at, char in enumerate(text):
        where[at] = ord(char)
    return len(text)


def _struct_of(when: tm) -> time.struct_time:
    """The ``time.struct_time`` a ``tm`` holds, for Python's own ``strftime``."""
    return time.struct_time((
        when.tm_year + 1900, when.tm_mon + 1, when.tm_mday, when.tm_hour, when.tm_min,
        when.tm_sec, when.tm_wday, when.tm_yday + 1, when.tm_isdst,
    ))  # fmt: skip
