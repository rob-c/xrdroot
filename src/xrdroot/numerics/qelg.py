"""Wynn's epsilon algorithm, as QUADPACK's ``qelg`` extrapolates a sequence of integral estimates.

GSL's table, in its order of operations: the new estimate appended, the
table rebuilt a diagonal at a time, the best element and an error kept, and
the error the spread of the last three results.
"""

from __future__ import annotations

from .kronrod import DBL_EPSILON

__all__ = ["DBL_MAX", "Table"]

DBL_MAX = 1.7976931348623157e308


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
        absolute, relative = DBL_MAX, 5 * DBL_EPSILON * abs(current)
        newelm, n_orig, n_final = n // 2, n, n
        nres_orig = self.nres
        result, abserr = current, DBL_MAX
        if n < 2:
            return current, max(absolute, relative)
        epstab[n + 2] = epstab[n]
        epstab[n] = DBL_MAX
        for i in range(newelm):
            res = epstab[n - 2 * i + 2]
            e0, e1, e2 = epstab[n - 2 * i - 2], epstab[n - 2 * i - 1], res
            e1abs = abs(e1)
            delta2 = e2 - e1
            err2 = abs(delta2)
            tol2 = max(abs(e2), e1abs) * DBL_EPSILON
            delta3 = e1 - e0
            err3 = abs(delta3)
            tol3 = max(e1abs, abs(e0)) * DBL_EPSILON
            if err2 <= tol2 and err3 <= tol3:
                absolute = err2 + err3
                relative = 5 * DBL_EPSILON * abs(res)
                return res, max(absolute, relative)
            e3 = epstab[n - 2 * i]
            epstab[n - 2 * i] = e1
            delta1 = e1 - e3
            err1 = abs(delta1)
            tol1 = max(e1abs, abs(e3)) * DBL_EPSILON
            if err1 <= tol1 or err2 <= tol2 or err3 <= tol3:
                n_final = 2 * i
                break
            ss = (1 / delta1 + 1 / delta2) - 1 / delta3
            if abs(ss * e1) <= 0.0001:
                n_final = 2 * i
                break
            res = e1 + 1 / ss
            epstab[n - 2 * i] = res
            error = err2 + abs(res - e2) + err3
            if error <= abserr:
                abserr, result = error, res
        return self._shift(result, abserr, n_orig, n_final, newelm, nres_orig)

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
