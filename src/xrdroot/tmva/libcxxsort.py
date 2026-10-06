"""libc++'s ``std::sort``, step for step: the order TMVA's ROC curve puts equal outputs in.

``ROCCurve`` sorts its ``(value, weight, isSignal)`` tuples by value with
``std::sort``, which is not stable: among events of one output value - a
PDE-Foam cell's, a k-NN fraction's - signal and background come out in the
order libc++'s introsort leaves them, and that order moves the curve's
trapezoids. This is libc++ 20's introsort for a type that is neither
arithmetic nor compared by ``std::less`` - Tukey's ninther pivot above 128
elements, the partitions that put equal elements on the right or the left,
insertion sort below 24, the incomplete insertion sort after a partition that
moved nothing - ported from its ``__algorithm/sort.h``. Only the order of
equal keys depends on it, so :func:`order` uses it only when keys repeat.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["order"]

#: Below this many elements, insertion sort.
LIMIT = 24
#: Above this many, the pivot is Tukey's ninther.
NINTHER = 128
#: How many moves the incomplete insertion sort makes before it gives up.
INCOMPLETE = 8


class _Sorter:
    """The array being sorted - keys and the positions they came from - and the steps on it."""

    def __init__(self, keys: Any) -> None:
        self.k = [float(v) for v in keys]
        self.i = list(range(len(self.k)))

    def swap(self, a: int, b: int) -> None:
        k, i = self.k, self.i
        k[a], k[b] = k[b], k[a]
        i[a], i[b] = i[b], i[a]

    def sort3(self, x: int, y: int, z: int) -> None:
        k = self.k
        if not k[y] < k[x]:
            if not k[z] < k[y]:
                return
            self.swap(y, z)
            if k[y] < k[x]:
                self.swap(x, y)
            return
        if k[z] < k[y]:
            self.swap(x, z)
            return
        self.swap(x, y)
        if k[z] < k[y]:
            self.swap(y, z)

    def sort4(self, a: int, b: int, c: int, d: int) -> None:
        self.sort3(a, b, c)
        self._bubble((a, b, c, d))

    def sort5(self, a: int, b: int, c: int, d: int, e: int) -> None:
        self.sort4(a, b, c, d)
        self._bubble((a, b, c, d, e))

    def _bubble(self, at: tuple[int, ...]) -> None:
        """The last of ``at`` swapped down while it is less than the one before it."""
        k = self.k
        for n in range(len(at) - 1, 0, -1):
            if not k[at[n]] < k[at[n - 1]]:
                return
            self.swap(at[n - 1], at[n])

    def small(self, first: int, last: int) -> bool:
        """The cases of five or fewer elements; ``False`` for a longer range."""
        size = last - first
        if size < 2:
            return True
        if size == 2:
            if self.k[last - 1] < self.k[first]:
                self.swap(first, last - 1)
            return True
        if size > 5:
            return False
        at = tuple(range(first, last))
        (self.sort3, self.sort4, self.sort5)[size - 3](*at)
        return True

    def _shift(self, i: int, j: int, lower: int) -> None:
        """Element ``i`` moved down past every greater one, no lower than ``lower``."""
        k, idx = self.k, self.i
        key, pos = k[i], idx[i]
        hole = i
        while True:
            k[hole], idx[hole] = k[j], idx[j]
            hole = j
            if hole == lower or not key < k[j - 1]:
                break
            j -= 1
        k[hole], idx[hole] = key, pos

    def insertion(self, first: int, last: int, guarded: bool) -> None:
        """``__insertion_sort``, or its unguarded form, which trusts a smaller element before."""
        k = self.k
        for i in range(first + 1, last):
            if k[i] < k[i - 1]:
                self._shift(i, i - 1, first if guarded else -1)

    def incomplete(self, first: int, last: int) -> bool:
        """``__insertion_sort_incomplete``: sorted, unless it took more than eight moves."""
        if self.small(first, last):
            return True
        k = self.k
        self.sort3(first, first + 1, first + 2)
        moves, j = 0, first + 2
        for i in range(first + 3, last):
            if k[i] < k[j]:
                self._shift(i, j, first)
                moves += 1
                if moves == INCOMPLETE:
                    return i + 1 == last
            j = i
        return True

    def _place(self, begin: int, first: int, pivot: tuple[float, int]) -> int:
        """The pivot put where the partition ended, the element there moved to the front."""
        k, idx = self.k, self.i
        at = first - 1
        if begin != at:
            k[begin], idx[begin] = k[at], idx[at]
        k[at], idx[at] = pivot
        return at

    def _past_less(self, at: int, pivot: float) -> int:
        """``do ++at; while (*at < pivot)``."""
        at += 1
        while self.k[at] < pivot:
            at += 1
        return at

    def _before_less(self, at: int, pivot: float) -> int:
        """``do --at; while (!(*at < pivot))``."""
        at -= 1
        while not self.k[at] < pivot:
            at -= 1
        return at

    def _past_not_greater(self, at: int, pivot: float) -> int:
        """``do ++at; while (!(pivot < *at))``."""
        at += 1
        while not pivot < self.k[at]:
            at += 1
        return at

    def _before_greater(self, at: int, pivot: float) -> int:
        """``do --at; while (pivot < *at)``."""
        at -= 1
        while pivot < self.k[at]:
            at -= 1
        return at

    def equals_right(self, first: int, last: int) -> tuple[int, bool]:
        """``__partition_with_equals_on_right``: the pivot's place, and if nothing moved."""
        k = self.k
        begin, pivot = first, (k[first], self.i[first])
        first = self._past_less(first, pivot[0])
        if begin == first - 1:
            while first < last:
                last -= 1
                if k[last] < pivot[0]:
                    break
        else:
            last = self._before_less(last, pivot[0])
        already = first >= last
        while first < last:
            self.swap(first, last)
            first = self._past_less(first, pivot[0])
            last = self._before_less(last, pivot[0])
        return self._place(begin, first, pivot), already

    def equals_left(self, first: int, last: int) -> int:
        """``__partition_with_equals_on_left``: after it, the first element above the pivot."""
        k = self.k
        begin, pivot = first, (k[first], self.i[first])
        if pivot[0] < k[last - 1]:
            first = self._past_not_greater(first, pivot[0])
        else:
            first += 1
            while first < last and not pivot[0] < k[first]:
                first += 1
        if first < last:
            last = self._before_greater(last, pivot[0])
        while first < last:
            self.swap(first, last)
            first = self._past_not_greater(first, pivot[0])
            last = self._before_greater(last, pivot[0])
        self._place(begin, first, pivot)
        return first

    def _pivot(self, first: int, last: int) -> None:
        """The median put first: of three, or Tukey's ninther for a long range."""
        half = (last - first) // 2
        if last - first > NINTHER:
            self.sort3(first, first + half, last - 1)
            self.sort3(first + 1, first + half - 1, last - 2)
            self.sort3(first + 2, first + half + 1, last - 3)
            self.sort3(first + half - 1, first + half, first + half + 1)
            self.swap(first, first + half)
        else:
            self.sort3(first + half, first, last - 1)

    def introsort(self, first: int, last: int, depth: int, leftmost: bool = True) -> None:
        """``__introsort``: partition about the pivot, sort the left, loop on the right."""
        while True:
            if self._finished(first, last, depth, leftmost):
                return
            depth -= 1
            self._pivot(first, last)
            if not leftmost and not self.k[first - 1] < self.k[first]:
                first = self.equals_left(first, last)
                continue
            i, already = self.equals_right(first, last)
            if already:
                done = self._settled(first, i, last)
                if done is not None:
                    if done == (first, last):
                        return
                    first, last = done
                    continue
            self.introsort(first, i, depth, leftmost)
            leftmost = False
            first = i + 1

    def _finished(self, first: int, last: int, depth: int, leftmost: bool) -> bool:
        """The ranges introsort sorts without partitioning: short ones, and any past its depth."""
        if self.small(first, last):
            return True
        if last - first < LIMIT:
            self.insertion(first, last, leftmost)
            return True
        if depth == 0:
            self._heap(first, last)
            return True
        return False

    def _settled(self, first: int, i: int, last: int) -> tuple[int, int] | None:
        """After a partition that moved nothing, each side's incomplete insertion sort.

        Both finished: the range as it was, meaning done. One finished: the other side,
        to go on with. Neither: ``None``, to partition as usual.
        """
        left = self.incomplete(first, i)
        if self.incomplete(i + 1, last):
            return (first, last) if left else (first, i)
        return (i + 1, last) if left else None

    def _heap(self, first: int, last: int) -> None:
        """``__partial_sort`` of the whole range, which a sort this deep falls back to."""
        pairs = sorted(
            zip(self.k[first:last], self.i[first:last], strict=False), key=lambda p: p[0]
        )
        self.k[first:last] = [p[0] for p in pairs]
        self.i[first:last] = [p[1] for p in pairs]


def order(keys: Any) -> Any:
    """The permutation ``std::sort`` puts ``keys`` in: NumPy's stable sort where none repeat."""
    keys = np.asarray(keys)
    stable = np.argsort(keys, kind="stable")
    if len(keys) < 2 or np.all(np.diff(keys[stable]) != 0):
        return stable
    made = _Sorter(keys)
    made.introsort(0, len(keys), 2 * (len(keys).bit_length() - 1))
    return np.asarray(made.i, dtype=np.int64)
