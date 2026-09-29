"""Wynn's epsilon algorithm, as QUADPACK's ``qelg`` extrapolates a sequence of integral estimates.

GSL's table, in its order of operations: the new estimate appended, the
table rebuilt a diagonal at a time, the best element and an error kept, and
the error the spread of the last three results.
"""

from __future__ import annotations

from typing import Any

from .kronrod import DBL_EPSILON

__all__ = ["DBL_MAX", "Table"]

DBL_MAX = 1.7976931348623157e308


def _near(u: float, v: float) -> bool:
    """Whether two entries differ by no more than rounding: ``|u - v| <= max(|u|, |v|) eps``."""
    return abs(u - v) <= max(abs(u), abs(v)) * DBL_EPSILON


class Table:
    """``extrapolation_table``: up to 52 estimates, and the last three results."""

    def __init__(self) -> None:
        self.n = 0
        self.rlist2 = [0.0] * 52
        self.nres = 0
        self.res3la = [0.0] * 3

    def append(self, y: float) -> None:
        self.rlist2[self.n] = y
        self.n += 1

    def qelg(self) -> tuple[float, float]:
        """The extrapolated result and its error."""
        epstab = self.rlist2
        n = self.n - 1
        current = epstab[n]
        newelm, n_final, nres_orig = n // 2, n, self.nres
        result, abserr = current, DBL_MAX
        if n < 2:
            return current, max(DBL_MAX, 5 * DBL_EPSILON * abs(current))
        epstab[n + 2] = epstab[n]
        epstab[n] = DBL_MAX
        for i in range(newelm):
            found = self._element(n, i)
            if isinstance(found, tuple):
                return found
            if found is None:
                n_final = 2 * i
                break
            error, res = found, epstab[n - 2 * i]
            if error <= abserr:
                abserr, result = error, res
        return self._shift(result, abserr, n, n_final, newelm, nres_orig)

    def _element(self, n: int, i: int) -> Any:
        """One new element of the table: its error - ``None`` if the table is to be cut here,
        or the answer itself if it has converged."""
        epstab = self.rlist2
        e0, e1, e2 = epstab[n - 2 * i - 2], epstab[n - 2 * i - 1], epstab[n - 2 * i + 2]
        near2, near3 = _near(e2, e1), _near(e1, e0)
        if near2 and near3:
            return e2, max(abs(e2 - e1) + abs(e1 - e0), 5 * DBL_EPSILON * abs(e2))
        e3 = epstab[n - 2 * i]
        epstab[n - 2 * i] = e1
        if _near(e1, e3) or near2 or near3:
            return None
        return self._extrapolated(n - 2 * i, e0, e1, e2, e3)

    def _extrapolated(self, at: int, e0: float, e1: float, e2: float, e3: float) -> Any:
        """Wynn's step: the new entry at ``at`` and its error - ``None`` if it is irregular."""
        ss = (1 / (e1 - e3) + 1 / (e2 - e1)) - 1 / (e1 - e0)
        if abs(ss * e1) <= 0.0001:
            return None
        res = e1 + 1 / ss
        self.rlist2[at] = res
        return abs(e2 - e1) + abs(res - e2) + abs(e1 - e0)

    def _shift(self, result: float, abserr: float, n_orig: int, n_final: int, newelm: int,
               nres_orig: int) -> tuple[float, float]:  # fmt: skip
        epstab, res3la = self.rlist2, self.res3la
        if n_final == 49:
            n_final = 2 * (49 // 2)
        if n_orig % 2 == 1:
            for i in range(newelm + 1):
                epstab[1 + i * 2] = epstab[i * 2 + 3]
        else:
            for i in range(newelm + 1):
                epstab[i * 2] = epstab[i * 2 + 2]
        if n_orig != n_final:
            for i in range(n_final + 1):
                epstab[i] = epstab[n_orig - n_final + i]
        self.n = n_final + 1
        if nres_orig < 3:
            res3la[nres_orig] = result
            abserr = DBL_MAX
        else:
            abserr = abs(result - res3la[2]) + abs(result - res3la[1]) + abs(result - res3la[0])
            res3la[0], res3la[1], res3la[2] = res3la[1], res3la[2], result
        self.nres = nres_orig + 1
        return result, max(abserr, 5 * DBL_EPSILON * abs(result))
