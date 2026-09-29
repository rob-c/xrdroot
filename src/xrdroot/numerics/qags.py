"""``gsl_integration_qags``: adaptive integration with extrapolation - ROOT's default 1D integral.

QUADPACK's ``dqagse`` as GSL has it: the 21-point rule over the whole range,
then the interval with the largest error bisected again and again, the
sequence of totals extrapolated by Wynn's epsilon algorithm when the small
intervals are what is left, until the error is within ``max(epsabs, epsrel
|I|)`` - every test and every sum in GSL's order, so the result is GSL's.
"""

from __future__ import annotations

from collections.abc import Callable

from .kronrod import DBL_EPSILON, DBL_MIN, qk21
from .qelg import DBL_MAX, Table
from .workspace import Workspace

__all__ = ["qags"]

#: The status GSL returns: success, and the failures by their ``GSL_E...`` names.
SUCCESS, EMAXITER, EROUND, ESING, EDIVERGE, EFAILED = 0, 11, 18, 21, 22, 5


def _too_small(a1: float, a2: float, b2: float) -> bool:
    tmp = (1 + 100 * DBL_EPSILON) * (abs(a2) + 1000 * DBL_MIN)
    return abs(a1) <= tmp and abs(b2) <= tmp


class _State:
    """The loop's running figures, as ``qags`` keeps them in locals."""

    def __init__(self, result0: float, abserr0: float) -> None:
        self.area, self.errsum = result0, abserr0
        self.res_ext, self.err_ext = result0, DBL_MAX
        self.ertest = self.large_errors = self.correc = 0.0
        self.ktmin = self.round1 = self.round2 = self.round3 = 0
        self.error_type = self.error_type2 = 0
        self.extrapolate = self.disallow = False


def qags(f: Callable[[float], float], a: float, b: float, epsabs: float, epsrel: float,
         limit: int = 1000) -> tuple[float, float, int]:  # fmt: skip
    """The integral of ``f`` over ``[a, b]``, its error, and GSL's status."""
    if epsabs <= 0 and (epsrel < 50 * DBL_EPSILON or epsrel < 0.5e-28):
        raise ValueError("GSL's QAGS cannot achieve that tolerance: with no absolute tolerance, "
                         "the relative one must be at least 50 times the double's epsilon")
    ws = Workspace(limit, a, b)
    result0, abserr0, resabs0, resasc0 = qk21(f, a, b)
    ws.set_initial(result0, abserr0)
    tolerance = max(epsabs, epsrel * abs(result0))
    if abserr0 <= 100 * DBL_EPSILON * resabs0 and abserr0 > tolerance:
        return result0, abserr0, EROUND
    if (abserr0 <= tolerance and abserr0 != resasc0) or abserr0 == 0.0:
        return result0, abserr0, SUCCESS
    if limit == 1:
        return result0, abserr0, EMAXITER
    table = Table()
    table.append(result0)
    s = _State(result0, abserr0)
    positive = abs(result0) >= (1 - 50 * DBL_EPSILON) * resabs0
    iteration, finished = 1, False
    while True:
        iteration += 1
        done = _step(f, ws, s, table, iteration, epsabs, epsrel, limit)
        if done == "result":
            finished = True
            break
        if done == "stop" or iteration >= limit:
            break
    if finished or s.err_ext == DBL_MAX:
        return ws.sum_results(), s.errsum, _status(s.error_type)
    return _finish(ws, s, positive, resabs0)


def _step(f: Callable[[float], float], ws: Workspace, s: _State, table: Table, iteration: int,
          epsabs: float, epsrel: float, limit: int) -> str:  # fmt: skip
    """One bisection: ``"result"`` when the total is good enough, ``"stop"`` to give up, else
    ``""`` to go on."""
    a_i, b_i, r_i, e_i = ws.retrieve()
    level = ws.level[ws.i] + 1
    a1, b1 = a_i, 0.5 * (a_i + b_i)
    a2, b2 = b1, b_i
    area1, error1, _abs1, resasc1 = qk21(f, a1, b1)
    area2, error2, _abs2, resasc2 = qk21(f, a2, b2)
    area12, error12 = area1 + area2, error1 + error2
    s.errsum = s.errsum + error12 - e_i
    s.area = s.area + area12 - r_i
    tolerance = max(epsabs, epsrel * abs(s.area))
    if resasc1 != error1 and resasc2 != error2:
        if abs(r_i - area12) <= 1.0e-5 * abs(area12) and error12 >= 0.99 * e_i:
            if not s.extrapolate:
                s.round1 += 1
            else:
                s.round2 += 1
        if iteration > 10 and error12 > e_i:
            s.round3 += 1
    if s.round1 + s.round2 >= 10 or s.round3 >= 20:
        s.error_type = 2
    if s.round2 >= 5:
        s.error_type2 = 1
    if _too_small(a1, a2, b2):
        s.error_type = 4
    ws.update(a1, b1, area1, error1, a2, b2, area2, error2)
    if s.errsum <= tolerance:
        return "result"
    if s.error_type:
        return "stop"
    if iteration >= limit - 1:
        s.error_type = 1
        return "stop"
    if iteration == 2:
        s.large_errors, s.ertest = s.errsum, tolerance
        table.append(s.area)
        return ""
    if s.disallow:
        return ""
    s.large_errors += -e_i
    if level < ws.maximum_level:
        s.large_errors += error12
    return _extrapolated(ws, s, table, epsabs, epsrel)


def _extrapolated(ws: Workspace, s: _State, table: Table, epsabs: float, epsrel: float) -> str:
    if not s.extrapolate:
        if ws.large_interval():
            return ""
        s.extrapolate = True
        ws.nrmax = 1
    if not s.error_type2 and s.large_errors > s.ertest and ws.increase_nrmax():
        return ""
    table.append(s.area)
    reseps, abseps = table.qelg()
    s.ktmin += 1
    if s.ktmin > 5 and s.err_ext < 0.001 * s.errsum:
        s.error_type = 5
    if abseps < s.err_ext:
        s.ktmin = 0
        s.err_ext, s.res_ext, s.correc = abseps, reseps, s.large_errors
        s.ertest = max(epsabs, epsrel * abs(reseps))
        if s.err_ext <= s.ertest:
            return "stop"
    if table.n == 1:
        s.disallow = True
    if s.error_type == 5:
        return "stop"
    ws.reset_nrmax()
    s.extrapolate = False
    s.large_errors = s.errsum
    return ""


def _finish(ws: Workspace, s: _State, positive: bool, resabs0: float) -> tuple[float, float, int]:
    """After the loop: the extrapolated result, unless the plain sum is the better one."""
    result, abserr = s.res_ext, s.err_ext  # what is returned, even once the test adds correc
    summed = (ws.sum_results(), s.errsum)
    tested = s.err_ext
    if s.error_type or s.error_type2:
        if s.error_type2:
            tested += s.correc
        if s.error_type == 0:
            s.error_type = 3
        if result != 0.0 and s.area != 0.0:
            if tested / abs(result) > s.errsum / abs(s.area):
                return (*summed, _status(s.error_type))
        elif tested > s.errsum:
            return (*summed, _status(s.error_type))
        elif s.area == 0.0:
            return result, abserr, _status(s.error_type)
    if not positive and max(abs(result), abs(s.area)) < 0.01 * resabs0:
        return result, abserr, _status(s.error_type)
    ratio = result / s.area
    if ratio < 0.01 or ratio > 100.0 or s.errsum > abs(s.area):
        s.error_type = 6
    return result, abserr, _status(s.error_type)


def _status(error_type: int) -> int:
    """``return_error``: the type, one down past 2, as GSL's status."""
    if error_type > 2:
        error_type -= 1
    return (SUCCESS, EMAXITER, EROUND, ESING, EROUND, EDIVERGE)[error_type] if (
        error_type <= 5) else EFAILED  # fmt: skip
