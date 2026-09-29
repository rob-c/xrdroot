"""The corners of GSL's QAGS a smooth integrand never reaches: its roundoff counters, its
bisection of a vanishing interval, its extrapolation's give-ups and its final choice."""

from __future__ import annotations

import math

from xrdroot.numerics.kronrod import DBL_EPSILON
from xrdroot.numerics.qags import _extrapolated, _finish, _State, _status, _step
from xrdroot.numerics.qelg import DBL_MAX, Table
from xrdroot.numerics.workspace import Workspace


def _square(x: float) -> float:
    return x * x


def _workspace(a: float = 0.0, b: float = 1.0, result: float = 1 / 3, error: float = 1e-17,
               limit: int = 10) -> Workspace:  # fmt: skip
    ws = Workspace(limit, a, b)
    ws.set_initial(result, error)
    return ws


def test_a_bisection_that_changes_nothing_counts_as_roundoff() -> None:
    """Halves that add up to the whole, with no less error: round1, or round2 extrapolating."""
    state = _State(1 / 3, 1e-17)
    assert _step(_square, _workspace(), state, Table(), 2, 0.0, 1e-300, 10) == ""
    assert state.round1 == 1
    state = _State(1 / 3, 1e-17)
    state.extrapolate, state.round2 = True, 4
    _step(_square, _workspace(), state, Table(), 2, 0.0, 1e-300, 10)
    assert state.round2 == 5 and state.error_type2 == 1


def test_enough_roundoff_is_an_error_of_type_two() -> None:
    state = _State(1 / 3, 1e-17)
    state.round1 = 9
    state.errsum = 1.0  # not converged, so the error stops it
    assert _step(_square, _workspace(), state, Table(), 12, 0.0, 1e-300, 100) == "stop"
    assert state.error_type == 2 and state.round3 == 1


def test_an_interval_too_small_to_halve_is_an_error_of_type_four() -> None:
    ws = _workspace(1.0, 1.0 + 4 * DBL_EPSILON, 4 * DBL_EPSILON, 1.0)
    state = _State(4 * DBL_EPSILON, 1.0)
    state.errsum = 10.0
    assert _step(_square, ws, state, Table(), 2, 0.0, 1e-300, 100) == "stop"
    assert state.error_type == 4


def test_the_last_subdivision_allowed_is_an_error_of_type_one() -> None:
    state = _State(1 / 3, 1.0)
    state.errsum = 10.0
    assert _step(_square, _workspace(error=1.0), state, Table(), 2, 0.0, 1e-300, 3) == "stop"
    assert state.error_type == 1


def test_bisection_after_extrapolation_was_disallowed_just_goes_on() -> None:
    state = _State(1 / 3, 1.0)
    state.errsum, state.disallow = 10.0, True
    assert _step(_square, _workspace(error=1.0), state, Table(), 3, 0.0, 1e-300, 100) == ""


def _extrapolating(table_values: list[float]) -> tuple[Workspace, _State, Table]:
    ws = _workspace()
    ws.update(0.0, 0.5, 0.1, 0.5, 0.5, 1.0, 0.2, 0.4)
    state = _State(0.3, 1.0)
    state.extrapolate, state.large_errors, state.ertest = True, 0.0, 1.0
    table = Table()
    for value in table_values:
        table.append(value)
    return ws, state, table


def test_extrapolation_that_stops_improving_is_an_error_of_type_five() -> None:
    ws, state, table = _extrapolating([1.0, 2.0, 4.0, 8.0])
    state.ktmin, state.err_ext, state.errsum = 5, 1e-9, 1.0
    assert _extrapolated(ws, state, table, 0.0, 1e-300) == "stop"
    assert state.error_type == 5


def test_extrapolation_of_one_estimate_is_disallowed_from_then_on() -> None:
    ws, state, table = _extrapolating([])
    assert _extrapolated(ws, state, table, 0.0, 1e-300) == ""
    assert state.disallow


def _finished(**given: float) -> _State:
    state = _State(1.0, 1.0)
    for key, value in given.items():
        setattr(state, key, value)
    return state


def test_after_the_loop_the_better_of_the_two_answers_is_kept() -> None:
    ws = _workspace(result=2.0, error=0.1)
    worse = _finished(res_ext=1.0, err_ext=0.5, area=2.0, errsum=0.1, error_type=2)
    assert _finish(ws, worse, True, 1.0) == (2.0, 0.1, 18)
    zero = _finished(res_ext=0.0, err_ext=0.5, area=2.0, errsum=0.1, error_type2=1, correc=0.1)
    assert _finish(ws, zero, True, 1.0) == (2.0, 0.1, 18)  # type 3, one down: roundoff
    kept = _finished(res_ext=0.0, err_ext=0.01, area=0.0, errsum=0.1, error_type=1)
    assert _finish(ws, kept, True, 1.0) == (0.0, 0.01, 11)
    neither = _finished(res_ext=0.0, err_ext=0.01, area=1.0, errsum=0.1, error_type=1)
    assert _finish(ws, neither, True, 1.0)[2] == 22


def test_after_the_loop_a_tiny_result_of_a_changing_sign_is_taken() -> None:
    ws = _workspace()
    small = _finished(res_ext=1e-5, err_ext=1e-9, area=1e-5, errsum=1e-9)
    assert _finish(ws, small, False, 1.0) == (1e-5, 1e-9, 0)
    wild = _finished(res_ext=1.0, err_ext=1e-9, area=1e-3, errsum=1e-9)
    assert _finish(ws, wild, True, 1.0)[2] == 22


def test_the_statuses_are_gsls() -> None:
    assert [_status(t) for t in range(8)] == [0, 11, 18, 18, 21, 18, 22, 5]
    assert DBL_MAX > 1e308 and math.isfinite(DBL_MAX)
