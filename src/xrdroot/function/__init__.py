"""ROOT's ``TF1``, ``TF2``, ``TF3`` and ``TFormula``: functions to evaluate, fit and write.

    >>> from xrdroot import Function
    >>> f = Function("g", "gaus", range=(-5, 5), parameters=[1.0, 0.0, 1.0])
    >>> float(f(0.0))
    1.0

A :class:`Function` is built from ROOT's formula language, from a Python
function, or read from a file - on its own, or among the fits a histogram
or graph carries - and it evaluates over whole arrays, differentiates in
its parameters, integrates, and writes back as the ``TF1`` ROOT reads.
"""

from __future__ import annotations

from .function import FUNCTIONS, Function

__all__ = ["Function", "FUNCTIONS"]
