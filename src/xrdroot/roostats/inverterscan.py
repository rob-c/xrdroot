"""``HypoTestInverter::RunLimit``: the automatic search for the upper limit.

The top of the range is doubled until ``CLs`` there is surely below the
target; the bottom is the parameter's minimum - zero, for ``CLs``. Then the
bracket is closed by the log-secant rule - the point where ``log CLs``
would cross the target on the line between the ends - each point run with
toys until it is precise near the target, until the estimate is as accurate
as asked; and if it never is, an exponential is fitted to the points near
the crossing.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.messages import ERROR, INFO, WARNING, log
from ..roofit.printing import g
from . import inverter as inv

__all__ = ["run_limit"]


def _last(results: Any) -> tuple[float, float]:
    return results.GetLastYValue(), results.GetLastYError()


def _upper_end(it: Any, r_max: float, target: float) -> Any:
    """``rMax``, doubled until ``CLs`` there is three errors below the target - or ``None``."""
    for tries in range(6):
        if not it.RunOnePoint(r_max):
            log(None, ERROR, "Eval", "HypoTestInverter::RunLimit - Hypotest failed at upper limit "
                f"of scan range: {g(r_max)}")  # fmt: skip
            r_max *= 0.95
            continue
        cls_max = _last(it._results)
        if cls_max[0] == 0 or cls_max[0] + 3 * abs(cls_max[1]) < target:
            return r_max, cls_max
        r_max += r_max
        if tries == 5:
            log(None, ERROR, "Eval", "HypoTestInverter::RunLimit - Cannot determine upper limit of "
                f"scan range. At {it._var.GetName()} = {g(r_max)}  still getting "
                f"{'CLs' if it._use_cls else 'CLsplusb'} = {g(cls_max[0])}")  # fmt: skip
            return None
    return r_max, (0.0, 0.0)


def _lower_end(it: Any, r_min: float, r_max: float, target: float) -> Any:
    """``rMin`` and ``CLs`` there - at one for ``CLs`` from zero - moved down until surely above
    the target, or ``None``."""
    if it._use_cls and r_min == 0:
        cls_min = (1.0, 0.0)
    else:
        if not it.RunOnePoint(r_min):
            log(None, ERROR, "Eval", "HypoTestInverter::RunLimit - Hypotest failed at lower limit "
                f"of scan range: {g(r_min)}")  # fmt: skip
            return None
        cls_min = _last(it._results)
    if cls_min[0] == 1 or cls_min[0] - 3 * abs(cls_min[1]) >= target:
        return r_min, cls_min
    if it._use_cls:
        return 0.0, (1.0, 0.0)
    return _lower_search(it, -r_max / 4, target)


def _lower_search(it: Any, r_min: float, target: float) -> Any:
    cls_min = (1.0, 0.0)
    for tries in range(6):
        if not it.RunOnePoint(r_min):
            log(None, ERROR, "Eval", "HypoTestInverter::RunLimit - Hypotest failed at lower limit "
                f"of scan range: {g(r_min)}")  # fmt: skip
            r_min = 0.1 if r_min == 0.0 else r_min * 1.1
            continue
        cls_min = _last(it._results)
        if cls_min[0] == 1 or cls_min[0] - 3 * abs(cls_min[1]) > target:
            return r_min, cls_min
        r_min += r_min
        if tries == 5:
            log(None, ERROR, "Eval", "HypoTestInverter::RunLimit - Cannot determine lower limit of "
                f"scan range. At {it._var.GetName()} = {g(r_min)} still get "
                f"{'CLs' if it._use_cls else 'CLsplusb'} = {g(cls_min[0])}")  # fmt: skip
            return None
    return r_min, cls_min


class _Search:
    """The bracket being closed: its ends, ``CLs`` at each, and the limit's estimate."""

    def __init__(self, it: Any, target: float, accuracy: tuple[float, float]) -> None:
        self.it, self.target, self.accuracy = it, target, accuracy
        self.r_min = self.r_max = 0.0
        self.cls_min, self.cls_max = (1.0, 0.0), (0.0, 0.0)
        self.limit = self.error = 0.0
        self.fit_range = (self.r_min, self.r_max)

    def tolerance(self) -> float:
        return max(self.accuracy[0], self.accuracy[1] * self.limit)

    def estimate(self) -> None:
        """The middle - or, by default, the log-secant point - and its error."""
        low, high = self.cls_min, self.cls_max
        self.limit, self.error = 0.5 * (self.r_min + self.r_max), 0.5 * (self.r_max - self.r_min)
        if inv.ALGORITHM[0] != "logSecant" or high[0] == 0:
            return
        log_min, log_max, log_target = math.log(low[0]), math.log(high[0]), math.log(self.target)
        span = self.r_max - self.r_min
        self.limit = self.r_min + span * (log_target - log_min) / (log_max - log_min)
        if high[1] != 0 and low[1] != 0:
            self.error = math.hypot((log_target - log_max) * (low[1] / low[0]),
                                    (log_target - log_min) * (high[1] / high[0]))  # fmt: skip
            self.error *= span / ((log_max - log_min) * (log_max - log_min))

    def run(self, x: float, where: str, said: Any = None) -> Any:
        """The point at ``x``, adaptively; failed, said - at ``said``, where RooStats says the
        other end."""
        if not self.it.RunOnePoint(x, True, self.target):
            shown = x if said is None else said
            log(None, ERROR, "Eval", f"HypoTestInverter::RunLimit - Hypo test failed at "
                f"x={g(shown)} when trying to find limit{where}.")  # fmt: skip
            return None
        return _last(self.it._results)

    def close(self) -> Any:
        """Bracketing and bisection: ``True`` at the accuracy asked, ``False`` to fit, ``None``
        if a test failed."""
        it = self.it
        while True:
            if 0 < it._max_toys < it._toys_run:
                log(None, WARNING, "Eval", "HypoTestInverter::RunLimit - maximum number of toys "
                    "reached  ")  # fmt: skip
                return False
            self.estimate()
            it._var.setError(self.error)
            if self.error < self.tolerance():
                if it._verbose > 1:
                    log(None, INFO, "Eval", f"HypoTestInverter::RunLimit - reached accuracy "
                        f"{g(self.error)} below {g(self.tolerance())}")  # fmt: skip
                return True
            mid = self.run(self.limit, "")
            if mid is None:
                return None
            if mid[1] == -1:
                log(None, ERROR, "Eval", "Hypotest failed")
                return None
            if abs(mid[0] - self.target) < 2 * mid[1]:
                return self._edges()
            if (mid[0] > self.target) == (self.cls_max[0] > self.target):
                self.r_max, self.cls_max = self.limit, mid
            else:
                self.r_min, self.cls_min = self.limit, mid

    def _edges(self) -> Any:
        """The ends moved in towards a point within two errors of the target: the range to fit."""
        if self.it._verbose > 0:
            log(None, INFO, "Eval", "Trying to move the interval edges closer")
        low_bound, high_bound = self.r_min, self.r_max
        while self.cls_min[1] == 0 or abs(self.r_min - self.limit) > self.tolerance():
            self.r_min = 0.5 * (self.r_min + self.limit)
            found = self.run(self.r_min, " from below")
            if found is None:
                return None
            self.cls_min = found
            if abs(found[0] - self.target) <= 2 * found[1]:
                break
            low_bound = self.r_min
        while self.cls_max[1] == 0 or abs(self.r_max - self.limit) > self.tolerance():
            self.r_max = 0.5 * (self.r_max + self.limit)
            found = self.run(self.r_max, " from above", self.r_min)
            if found is None:
                return None
            self.cls_max = found
            if abs(found[0] - self.target) <= 2 * found[1]:
                break
            high_bound = self.r_max
        self.fit_range = (low_bound, high_bound)
        return False

    def fit(self) -> Any:
        """No accuracy from the scan: ``target e^(b (x - limit))`` fitted to the points in the
        fit range - a point more at random in it each time - until the fitted limit is accurate
        enough; ``None`` if a test failed. The fitted function, for the picture."""
        from ..pyroot.core import TF1
        from ..pyroot.roostats.inverterplot import make_plot
        from ..roofit.rng import generator

        it = self.it
        low, high = self.fit_range
        if it._verbose:
            log(None, INFO, "Eval", "HypoTestInverter::RunLimit - Before fit   --- \nLimit: "
                f"{it._var.GetName()} < {g(self.limit)} +/- {g(self.error)} [{g(self.r_min)}, "
                f"{g(self.r_max)}]")  # fmt: skip
        expo = TF1("expoFit", "[0]*exp([1]*(x-[2]))", low, high)
        expo.FixParameter(0, self.target)
        slope = math.log(self.cls_max[0] / self.cls_min[0]) / (self.r_max - self.r_min)
        expo.SetParameter(1, slope)
        expo.SetParameter(2, self.limit)
        self.error = max(abs(low - self.limit), abs(high - self.limit))
        it._limit_plot = make_plot(it._results)
        npoints = sum(1 for x in it._limit_plot.GetX() if low <= x <= high)
        for i in range(9):
            it._limit_plot.Sort()
            it._limit_plot.Fit(expo, "QNR EX0" if it._verbose <= 1 else "NR EXO")
            value, error = expo.GetParameter(2), expo.GetParError(2)
            if it._verbose:
                log(None, INFO, "Eval", f"Fit to {npoints} points: {g(value)} +/- {g(error)}")
            if self.r_min < value < self.r_max and error < 0.5 * (high - low):
                self.limit, self.error = value, error
                if self.error < self.tolerance():
                    break
            trial = generator().Rndm() * (high - low) + low
            if i != 8 and not it.RunOnePoint(trial, True, self.target):
                return None
            npoints += 1
        return expo


def run_limit(it: Any, abs_tol: float, rel_tol: float, hint: Any) -> tuple[bool, float, float]:
    """The search, and the limit it found kept on the results as fitted."""
    var = it._var
    if hint is not None and hint > var.getMin():
        var.setMax(min(3.0 * hint, var.getMax()))
        var.setMin(max(0.3 * hint, var.getMin()))
        log(None, INFO, "InputArguments", f"HypoTestInverter::RunLimit - Use hint value {g(hint)} "
            f"search in interval {g(var.getMin())} , {g(var.getMax())}")  # fmt: skip
    accuracy = (abs_tol if abs_tol > 0 else inv.ABS_ACCURACY[0],
                rel_tol if rel_tol > 0 else inv.REL_ACCURACY[0])  # fmt: skip
    search = _Search(it, it._size, accuracy)
    search.r_min, search.r_max = var.getMin(), var.getMax()
    search.fit_range = (search.r_min, search.r_max)
    search.limit = 0.5 * (search.r_max + search.r_min)
    search.error = 0.5 * (search.r_max - search.r_min)
    it._create_results()
    it._limit_plot = None
    if it._verbose > 0:
        log(None, INFO, "Eval", "Search for upper limit to the limit")
    upper = _upper_end(it, search.r_max, search.target)
    if upper is None:
        return False, search.limit, search.error
    search.r_max, search.cls_max = upper
    if it._verbose > 0:
        log(None, INFO, "Eval", "HypoTestInverter::RunLimit - Search for lower limit to the limit")
    lower = _lower_end(it, search.r_min, search.r_max, search.target)
    if lower is None:
        return False, search.limit, search.error
    search.r_min, search.cls_min = lower
    if it._verbose > 0:
        log(None, INFO, "Eval", "HypoTestInverter::RunLimit - Now doing proper bracketing & "
            "bisection")  # fmt: skip
    done = search.close()
    expo = None
    if done is False:
        expo = search.fit()
        if expo is None:
            return False, search.limit, search.error
    elif done is None:
        return False, search.limit, search.error
    return _finish(it, search, expo)


def _finish(it: Any, search: Any, expo: Any) -> tuple[bool, float, float]:
    from ..pyroot.roostats.inverterplot import draw_limit_plot

    if it._limit_plot is not None and it._limit_plot.GetN() > 0:
        draw_limit_plot(it, search.target, search.limit, search.error, expo)
    var = it._var
    log(None, INFO, "Eval", f"HypoTestInverter::RunLimit - Result:    \n\tLimit: {var.GetName()} < "
        f"{g(search.limit)} +/- {g(search.error)} @ {g((1 - it._size) * 100)}% CL")  # fmt: skip
    if it._verbose > 1:
        log(None, INFO, "Eval", f"Total toys: {it._toys_run}")
    results = it._results
    results._upper, results._errors[1], results._fitted[1] = search.limit, search.error, True
    results._lower, results._errors[0], results._fitted[0] = var.getMin(), 0.0, True
    return True, search.limit, search.error
