"""The steps of HypoTestInverter's automatic search, each driven by scripted points.

A stand-in inverter answers each point with the next of a list - a ``CLs``
and its error, or a failed test - so the bracketing of the lower end, the
moving in of the edges and the final fit can each be walked through, as
``RunLimit`` walks them.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.roofit.messages import service
from xrdroot.roostats import inverterscan as scan


@pytest.fixture(autouse=True)
def _quiet() -> Any:
    service().reset()
    yield
    service().reset()


class _Results:
    def __init__(self) -> None:
        self.last = (0.0, 0.0)
        self._x: list[float] = []

    def GetLastYValue(self) -> float:
        return self.last[0]

    def GetLastYError(self) -> float:
        return self.last[1]


class _Inverter:
    """Each point the next answer: ``None`` for a failed test, else ``(CLs, error)``."""

    def __init__(self, answers: list[Any], cls: bool = False) -> None:
        self.answers = list(answers)
        self._results = _Results()
        self._use_cls, self._verbose, self._max_toys, self._toys_run = cls, 1, 0, 0
        self._var = ROOT.RooRealVar("mu", "mu", 1, 0, 10)
        self.tried: list[float] = []

    def RunOnePoint(self, x: float, adaptive: bool = False, target: float = -1) -> bool:
        self.tried.append(x)
        answer = self.answers.pop(0) if self.answers else (0.05, 0.01)
        if answer is None:
            return False
        self._results.last = answer
        return True


def test_the_lower_end_is_moved_down_until_it_is_surely_above_the_target(capsys: Any) -> None:
    """CLs+b: failures moved on from, the end doubled until three errors above the target."""
    it = _Inverter([(0.01, 0.0), None, (0.02, 0.0), (0.9, 0.01)])
    assert scan._lower_end(it, 0.0, 4.0, 0.05) == (-2.2, (0.9, 0.01))
    assert it.tried == [0.0, -1.0, -1.1, -2.2]
    assert "Hypotest failed at lower limit of scan range: -1" in capsys.readouterr().out
    assert scan._lower_end(_Inverter([(0.01, 0.0)], True), 1.0, 4.0, 0.05) == (0.0, (1.0, 0.0))
    it = _Inverter([None] * 6)
    assert scan._lower_search(it, 0.0, 0.05) == (pytest.approx(0.161051), (1.0, 0.0))
    assert it.tried[:2] == [0.0, 0.1]


def _search(answers: list[Any], cls_min: Any, cls_max: Any) -> Any:
    it = _Inverter(answers)
    found = scan._Search(it, 0.05, (1e-6, 1e-6))
    found.r_min, found.r_max, found.cls_min, found.cls_max = 1.0, 3.0, cls_min, cls_max
    found.limit = 2.0
    return found


def test_the_edges_move_in_until_a_point_is_within_two_errors(capsys: Any) -> None:
    edges = _search([(0.2, 0.01), (0.06, 0.02), (0.01, 0.001), (0.04, 0.01)], (0.5, 0.0),
                    (0.0, 0.0))  # fmt: skip
    assert edges._edges() is False
    assert edges.fit_range == (1.5, 2.5)
    assert "Trying to move the interval edges closer" in capsys.readouterr().out
    assert _search([None], (0.5, 0.0), (0.0, 0.0))._edges() is None
    assert _search([(0.05, 0.01), None], (0.5, 0.0), (0.0, 0.0))._edges() is None
    assert "when trying to find limit from above." in capsys.readouterr().out


def test_the_fit_stops_when_a_trial_point_fails(monkeypatch: Any) -> None:
    from xrdroot.roostats import inverterresult

    fit = _search([None], (0.5, 0.01), (0.01, 0.01))
    fit.fit_range, fit.r_min = (1.0, 3.0), 2.5  # the fit's start, 2, outside: a trial point
    results = inverterresult.HypoTestInverterResult("r", ROOT.RooRealVar("mu", "mu", 1, 0, 10))
    fit.it._results = results
    assert fit.fit() is None


def test_edges_already_close_are_the_fit_range_as_they_are() -> None:
    edges = _search([], (0.06, 0.01), (0.04, 0.01))
    edges.it._verbose = 0
    edges.r_min = edges.r_max = edges.limit
    assert edges._edges() is False and edges.fit_range == (2.0, 2.0)


def test_a_fit_accurate_enough_is_kept_at_once() -> None:
    from xrdroot.roostats import inverterresult

    fit = _search([], (0.5, 0.01), (0.01, 0.01))
    fit.accuracy, fit.fit_range = (10.0, 10.0), (1.0, 3.0)
    fit.it._results = inverterresult.HypoTestInverterResult("r", fit.it._var)
    assert fit.fit() is not None and fit.limit == 2.0 and fit.it.tried == []


def test_a_search_whose_fit_fails_has_no_limit(monkeypatch: Any) -> None:
    it = _Inverter([(0.0, 0.0)], cls=True)
    it._size = 0.05
    it._create_results = lambda: None
    monkeypatch.setattr(scan._Search, "close", lambda self: False)
    monkeypatch.setattr(scan._Search, "fit", lambda self: None)
    assert scan.run_limit(it, 0.0, 0.0, None)[0] is False
