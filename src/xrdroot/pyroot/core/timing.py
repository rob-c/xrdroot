"""``TStopwatch``, ``TBenchmark`` and ``TDatime``: clocks and dates, printed as ROOT prints them."""

from __future__ import annotations

import time
from typing import Any

from .cformat import c_format
from .objects import TNamed, TObject

__all__ = ["TStopwatch", "TBenchmark", "gBenchmark", "TDatime", "TTimeStamp"]


class TStopwatch(TObject):
    """``TStopwatch``: real and CPU time, started when made, as ROOT's is."""

    kUndefined, kStopped, kRunning = 0, 1, 2

    def __init__(self) -> None:
        super().__init__()
        self._real = self._cpu = 0.0
        self._counter = 0
        self._state = self.kUndefined
        self.Start()

    def Start(self, reset: bool = True) -> None:
        """``Start``: from zero, or with ``reset`` false carrying on from what was kept."""
        if reset:
            self._real = self._cpu = 0.0
            self._counter = 0
        if self._state != self.kRunning:
            self._real_start, self._cpu_start = time.perf_counter(), time.process_time()
        self._state = self.kRunning
        self._counter += 1

    def Stop(self) -> None:
        if self._state == self.kRunning:
            self._real += time.perf_counter() - self._real_start
            self._cpu += time.process_time() - self._cpu_start
        self._state = self.kStopped

    def Continue(self) -> None:
        """``Continue``: start again without losing the time kept."""
        if self._state == self.kUndefined:
            return
        if self._state == self.kStopped:
            self._real_start, self._cpu_start = time.perf_counter(), time.process_time()
        self._state = self.kRunning

    def Reset(self) -> None:
        self.ResetCpuTime()
        self.ResetRealTime()

    def ResetCpuTime(self, time: float = 0.0) -> None:
        self.Stop()
        self._cpu = float(time)

    def ResetRealTime(self, time: float = 0.0) -> None:
        self.Stop()
        self._real = float(time)

    def RealTime(self) -> float:
        """``RealTime``: the seconds on the clock, stopping it first if it is running."""
        self.Stop()
        return self._real

    def CpuTime(self) -> float:
        """``CpuTime``: the processor seconds, stopping the clock first if it is running."""
        self.Stop()
        return self._cpu

    def Counter(self) -> int:
        return self._counter

    def Print(self, option: str = "") -> None:
        """``Print``: ``Real time 0:00:01, CP time 0.990``, with ``m`` or ``u`` finer."""
        real, cpu = max(self.RealTime(), 0.0), max(self.CpuTime(), 0.0)
        hours = int(real / 3600)
        real -= hours * 3600
        minutes = int(real / 60)
        real -= minutes * 60
        seconds = {"m": "%06.3f", "u": "%09.6f"}.get(str(option)[:1], "%02d")
        text = c_format(f"Real time %d:%02d:{seconds}, CP time %.3f", hours, minutes, real, cpu)
        if self._counter > 1:
            text += f", {self._counter} slices"
        print(text)


class TBenchmark(TNamed):
    """``TBenchmark``: stopwatches by name, and their times printed as ROOT prints them."""

    LINE = "%-10s: Real Time = %6.2f seconds Cpu Time = %6.2f seconds"

    def __init__(self) -> None:
        super().__init__("Benchmark", "ROOT benchmark")
        self._watches: dict[str, TStopwatch] = {}

    def Start(self, name: Any) -> None:
        self._watches[str(name)] = TStopwatch()

    def Stop(self, name: Any) -> None:
        watch = self._watches.get(str(name))
        if watch is not None:
            watch.Stop()

    def Reset(self) -> None:
        self._watches.clear()

    def GetRealTime(self, name: Any) -> float:
        watch = self._watches.get(str(name))
        return 0.0 if watch is None else watch.RealTime()

    def GetCpuTime(self, name: Any) -> float:
        watch = self._watches.get(str(name))
        return 0.0 if watch is None else watch.CpuTime()

    def Show(self, name: Any) -> None:
        """``Show``: one stopwatch's times, stopping it."""
        called = str(name)
        self.Stop(called)
        print(c_format(self.LINE, called, self.GetRealTime(called), self.GetCpuTime(called)))

    def Print(self, name: Any = "") -> None:
        print(c_format(self.LINE, str(name), self.GetRealTime(name), self.GetCpuTime(name)))

    def Summary(self, *totals: Any) -> tuple[float, float]:
        """``Summary``: every stopwatch's times, then their total."""
        real = cpu = 0.0
        for called, watch in self._watches.items():
            real, cpu = real + watch.RealTime(), cpu + watch.CpuTime()
            print(c_format(self.LINE, called, watch.RealTime(), watch.CpuTime()))
        print(c_format(self.LINE, "TOTAL", real, cpu))
        return real, cpu


#: ``gBenchmark``.
gBenchmark = TBenchmark()

#: ``EDayOfWeek``'s names as ``ctime`` spells them.
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


class TDatime(TObject):
    """``TDatime``: a date and time to the second, local, as ROOT keeps it."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        self.Set(*args)

    def Set(self, *args: Any) -> None:
        """``Set()``: now; ``Set(date, time)`` as ``YYYYMMDD`` and ``HHMMSS``; or six numbers."""
        if not args:
            self._t = time.localtime()
        elif len(args) == 2:
            date, clock = int(args[0]), int(args[1])
            self._set_parts(date // 10000, date // 100 % 100, date % 100, clock // 10000,
                            clock // 100 % 100, clock % 100)  # fmt: skip
        elif len(args) == 1:
            self._t = time.localtime(int(args[0]))
        else:
            self._set_parts(*(int(value) for value in args))

    def _set_parts(self, *parts: int) -> None:
        full = (*parts, 0, 0, 0, 0, 0, 0)[:6]
        self._t = time.localtime(time.mktime((*full, 0, 0, -1)))

    def Convert(self, toGMT: bool = False) -> int:
        return int(time.mktime(self._t))

    def GetDate(self) -> int:
        return self._t.tm_year * 10000 + self._t.tm_mon * 100 + self._t.tm_mday

    def GetTime(self) -> int:
        return self._t.tm_hour * 10000 + self._t.tm_min * 100 + self._t.tm_sec

    def GetYear(self) -> int:
        return self._t.tm_year

    def GetMonth(self) -> int:
        return self._t.tm_mon

    def GetDay(self) -> int:
        return self._t.tm_mday

    def GetHour(self) -> int:
        return self._t.tm_hour

    def GetMinute(self) -> int:
        return self._t.tm_min

    def GetSecond(self) -> int:
        return self._t.tm_sec

    def GetDayOfWeek(self) -> int:
        """``GetDayOfWeek``: 1 for Monday to 7 for Sunday."""
        return self._t.tm_wday + 1

    def AsString(self) -> str:
        """``AsString``: as ``ctime`` writes it, ``Thu Sep 25 12:00:00 2026``."""
        return time.asctime(self._t)

    def AsSQLString(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", self._t)

    def Print(self, option: str = "") -> None:
        print(f"Date/Time = {self.AsString()}")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TDatime) and self.Convert() == other.Convert()

    def __hash__(self) -> int:
        return self.Convert()


class TTimeStamp(TDatime):
    """``TTimeStamp``: a moment, which is a date and time here, to the second."""

    def AsDouble(self) -> float:
        return float(self.Convert())

    def GetSec(self) -> int:
        return self.Convert()
