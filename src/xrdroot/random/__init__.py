"""ROOT's random numbers, to the bit: ``TRandom``, its three generators, and ``gRandom``.

A ROOT macro that generates anything - a toy study, a smearing, a histogram
filled from ``gRandom`` - can only be checked against its output with the
generator it was written for. These are those generators: the same seed
gives the same ``Rndm()`` values in the same order as ROOT's, and every
distribution ``TRandom`` offers follows ROOT's own algorithm with ROOT's
constants, so it takes the same draws and returns the same numbers.

    >>> from xrdroot.random import TRandom3, gRandom
    >>> r = TRandom3(4357)
    >>> r.rndm(5)                   # five gRandom->Rndm()        # doctest: +SKIP
    >>> r.gaus(10, 2, n=1_000_000)  # a million gRandom->Gaus(10, 2)  # doctest: +SKIP

Everything is an array at a time: ``n`` numbers are ``n`` ROOT calls, made
with NumPy, including the rejection algorithms, whose draws are followed
through exactly as ROOT's loops take them (see :mod:`.chain`).
"""

from __future__ import annotations

from .mersenne import TRandom3
from .ranlux import TRandom1
from .tausworthe import TRandom2
from .trandom import TRandom

__all__ = ["TRandom", "TRandom1", "TRandom2", "TRandom3", "gRandom"]

#: ROOT's global generator: a ``TRandom3`` at its default seed of 4357, as
#: ``TRandom3.cxx`` makes it when ROOT starts.
gRandom = TRandom3()
