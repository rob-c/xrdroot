"""Following a rejection algorithm through a run of draws, without a loop per draw.

ROOT's ``Gaus`` and ``Poisson`` take a varying number of ``Rndm()`` values
each: one most of the time, two, sometimes a dozen. Which draw starts the
tenth number depends on how many the first nine took, so the numbers cannot
simply be cut out of the stream at fixed places. What can be done a whole
array at a time is to ask, of every draw, "if a number started here, where
would the next one start?" - that is arithmetic on the draws around it - and
then to follow those links from the first draw. Following them is a pointer
chase, and :func:`follow` does it by doubling: a table of where each draw
leads in one step is squared into two steps, four, eight, so that a million
numbers are found in twenty passes over the array rather than a million
steps of Python. The chosen starts are exactly the ones ROOT's loop would
have started at, because they are that loop's own links followed in order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["first_from", "follow"]


def follow(links: Any, limit: int) -> Any:
    """The draws a run of numbers starts at, from the first draw, at most ``limit`` of them.

    ``links[i]`` is where the number after one starting at ``i`` starts, and
    ``len(links) + 1`` where a number starting at ``i`` would need draws past
    the end of the run: that number, and every one after it, is left for the
    next run. The starts come back in order, every one of them a number that
    finished inside the run.
    """
    size = len(links)
    stop = size + 1
    table = np.concatenate([np.asarray(links, dtype=np.int64), [stop, stop]])
    starts = np.zeros(1, dtype=np.int64)
    jump = table
    while len(starts) < limit and table[starts[-1]] != stop:
        starts = np.concatenate([starts, jump[starts]])
        jump = jump[jump]
    return starts[table[starts] != stop][:limit]


def first_from(flags: Any, stride: int = 1) -> Any:
    """For every index, the first at or after it, in steps of ``stride``, whose flag is set.

    Where none is set the answer is ``len(flags)``. A loop that draws until
    something is accepted ends at exactly this index, whichever draw it began
    at, which is what lets every possible beginning be answered at once.
    """
    size = len(flags)
    marks = np.where(flags, np.arange(size), size)
    found = np.empty(size, dtype=np.int64)
    for phase in range(stride):
        found[phase::stride] = np.minimum.accumulate(marks[phase::stride][::-1])[::-1]
    return found
