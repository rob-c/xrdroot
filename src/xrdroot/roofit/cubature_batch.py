"""Independent adaptive integrals can share function calls while keeping their own heaps.

Each run requests one box at a time and receives its values in ROOT's point
order. Combining those requests changes neither its subdivisions nor its
arithmetic. The integrand must be free of side effects between groups.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

import numpy as np

from . import cubature as c

#: One integrator pauses until its requested points have been evaluated.
_Steps = Generator[Any, list[float], c.CubatureResult]


def _measure(
    run: c._Run, centre: list[float], width: list[float]
) -> Generator[Any, list[float], tuple[float, float]]:
    values = yield c.rule_points(centre, width)
    return run.accumulate(values, width)


def _push(
    run: c._Run, centre: list[float], width: list[float]
) -> Generator[Any, list[float], None]:
    value, error = yield from _measure(run, centre, width)
    run.heap.sift_up(error)
    run.heap.store(value, error, run.axis, (centre, width))


def _divide(run: c._Run, n: int) -> Generator[Any, list[float], None]:
    heap = run.heap
    heap.isbrgn = heap.irgnst
    value, error, axis, centre, width = heap.top(n)
    run.abserr -= error
    run.result -= value
    cut = max(axis, 1) - 1
    width[cut] = 0.5 * width[cut]
    centre[cut] -= width[cut]
    value, error = yield from _measure(run, centre, width)
    heap.sift_down(error)
    heap.store(value, error, run.axis, (centre, width))
    centre[cut] += 2 * width[cut]
    heap.isbrgs += heap.irgnst
    heap.isbrgn = heap.isbrgs
    yield from _push(run, centre, width)


def _steps(lows: Any, highs: Any, options: tuple[float, float, int | None, int, int]) -> _Steps:
    centre, width = c._box(lows, highs)
    limits, eps_abs, eps_rel = c._start(len(centre), options)
    heap = c._Heap(limits.irgnst, [0.0] * (limits.iwk + 10), limits.irgnst, limits.irgnst)
    run = c._Run(limits, eps_abs, eps_rel, lambda points: [], heap)
    yield from _push(run, centre, width)
    while True:
        relerr = run.relerr()
        status = run.status(relerr)
        if status != c._DIVIDING:
            return c.CubatureResult(run.result, relerr, status, run.ifncls)
        yield from _divide(run, len(centre))


def adaptive_integrals(
    func: Callable[[Any, Any], Any],
    lows: Any,
    highs: Any,
    count: int,
    eps_abs: float = 0.0,
    eps_rel: float = c.DEFAULT_EPS_REL,
    max_pts: int | None = None,
    min_pts: int = 0,
    size: int = 0,
) -> list[c.CubatureResult]:
    """Integrate independent groups with one function call for their pending boxes.

    ``func(groups, points)`` receives a group index per point and a matrix
    of coordinates. It must return one value per point. Each group's
    convergence tests, evaluation count and floating-point sums match a
    separate vectorized ``adaptive_integral`` with the same options.
    """
    if count < 0:
        raise ValueError("The number of independent integrals must be nonnegative.")
    options = eps_abs, eps_rel, max_pts, min_pts, size
    runs = [_steps(lows, highs, options) for _ in range(count)]
    pending = [(i, next(run)) for i, run in enumerate(runs)]
    results: dict[int, c.CubatureResult] = {}
    while pending:
        values = _values(func, pending)
        pending = _advance(runs, pending, values, results)
    return [results[i] for i in range(count)]


def _values(func: Callable[[Any, Any], Any], pending: list[tuple[int, Any]]) -> Any:
    points = np.concatenate([points for _, points in pending])
    groups = np.concatenate([np.full(len(points), i, dtype=np.intp) for i, points in pending])
    values = np.asarray(func(groups, points), dtype=np.float64).ravel()
    if len(values) != len(points):
        raise ValueError("The batched integrand must return one value per requested point.")
    return values


def _advance(
    runs: list[_Steps],
    pending: list[tuple[int, Any]],
    values: Any,
    results: dict[int, c.CubatureResult],
) -> list[tuple[int, Any]]:
    following = []
    cursor = 0
    for i, points in pending:
        given = values[cursor : cursor + len(points)].tolist()
        cursor += len(points)
        try:
            following.append((i, runs[i].send(given)))
        except StopIteration as done:
            results[i] = done.value
    return following
