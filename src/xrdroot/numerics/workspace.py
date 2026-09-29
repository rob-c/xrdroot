"""GSL's integration workspace: the subintervals of an adaptive integral, kept ordered by error.

``gsl_integration_workspace`` with its helpers - ``update``, ``qpsrt``,
``increase_nrmax``, ``large_interval`` - as QUADPACK keeps them: the list
of intervals, each's estimate, error and bisection level, and ``order``,
the indices by decreasing error that the next bisection is picked from.
"""

from __future__ import annotations

__all__ = ["Workspace"]


class Workspace:
    """Up to ``limit`` subintervals of ``[a, b]``."""

    def __init__(self, limit: int, a: float, b: float) -> None:
        self.limit = int(limit)
        self.alist, self.blist = [0.0] * limit, [0.0] * limit
        self.rlist, self.elist = [0.0] * limit, [0.0] * limit
        self.order, self.level = [0] * limit, [0] * limit
        self.alist[0], self.blist[0] = a, b
        self.size = self.nrmax = self.i = self.maximum_level = 0

    def set_initial(self, result: float, error: float) -> None:
        self.size = 1
        self.rlist[0], self.elist[0] = result, error

    def retrieve(self) -> tuple[float, float, float, float]:
        i = self.i
        return self.alist[i], self.blist[i], self.rlist[i], self.elist[i]

    def update(self, a1: float, b1: float, area1: float, error1: float, a2: float, b2: float,
               area2: float, error2: float) -> None:  # fmt: skip
        """The bisected interval's two halves: the worse where it was, the other appended."""
        i_max, i_new = self.i, self.size
        new_level = self.level[i_max] + 1
        if error2 > error1:
            self.alist[i_max], self.rlist[i_max], self.elist[i_max] = a2, area2, error2
            self.alist[i_new], self.blist[i_new] = a1, b1
            self.rlist[i_new], self.elist[i_new] = area1, error1
        else:
            self.blist[i_max], self.rlist[i_max], self.elist[i_max] = b1, area1, error1
            self.alist[i_new], self.blist[i_new] = a2, b2
            self.rlist[i_new], self.elist[i_new] = area2, error2
        self.level[i_max] = self.level[i_new] = new_level
        self.size += 1
        self.maximum_level = max(self.maximum_level, new_level)
        self.qpsrt()

    def qpsrt(self) -> None:
        """Keep ``order`` descending by error, and point at the worst interval."""
        last, limit, elist, order = self.size - 1, self.limit, self.elist, self.order
        i_nrmax = self.nrmax
        i_maxerr = order[i_nrmax]
        if last < 2:
            order[0], order[1] = 0, 1
            self.i = i_maxerr
            return
        errmax = elist[i_maxerr]
        while i_nrmax > 0 and errmax > elist[order[i_nrmax - 1]]:
            order[i_nrmax] = order[i_nrmax - 1]
            i_nrmax -= 1
        top = last if last < (limit // 2 + 2) else limit - last + 1
        i = i_nrmax + 1
        while i < top and errmax < elist[order[i]]:
            order[i - 1] = order[i]
            i += 1
        order[i - 1] = i_maxerr
        errmin = elist[last]
        k = top - 1
        while k > i - 2 and errmin >= elist[order[k]]:
            order[k + 1] = order[k]
            k -= 1
        order[k + 1] = last
        self.i = order[i_nrmax]
        self.nrmax = i_nrmax

    def reset_nrmax(self) -> None:
        self.nrmax = 0
        self.i = self.order[0]

    def increase_nrmax(self) -> bool:
        last = self.size - 1
        jupbnd = self.limit + 1 - last if last > (1 + self.limit // 2) else last
        for _ in range(self.nrmax, jupbnd + 1):
            i_max = self.order[self.nrmax]
            self.i = i_max
            if self.level[i_max] < self.maximum_level:
                return True
            self.nrmax += 1
        return False

    def large_interval(self) -> bool:
        return self.level[self.i] < self.maximum_level

    def sum_results(self) -> float:
        total = 0.0
        for k in range(self.size):
            total += self.rlist[k]
        return total
