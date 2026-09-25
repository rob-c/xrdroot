"""ROOT's fitting: ``TH1::Fit``, ``TGraph::Fit`` and Minuit, the way ROOT does them.

    >>> r = h.fit("gaus")                 # h->Fit("gaus")                  # doctest: +SKIP
    >>> r = h.fit("gaus", "L R")          # likelihood, the function's range
    ... # doctest: +SKIP
    >>> r = graph.fit("pol1")             # gr->Fit("pol1"): linear least squares
    ... # doctest: +SKIP
    >>> r.parameters, r.errors, r.chi2, r.ndf, r.prob                   # doctest: +SKIP

A fit here is ROOT's, step for step: the same points taken from the
histogram or the graph (:mod:`.data`), the same starting values for a
built-in shape (:mod:`.guesses`), the same chi-square or likelihood
(:mod:`.cost`), MIGRAD set up as ROOT sets it up (:mod:`.minuit`), a
polynomial - or anything linear in its parameters - solved exactly rather
than iterated (:mod:`.linear`), and the fitted function left on what was
fitted with the members ROOT's ``TF1`` records (:mod:`.hfit`).
:func:`minimize` is Minuit itself, for any function of the parameters, and
:func:`unbinned` an unbinned likelihood fit to arrays.

Minuit comes from iminuit, which is Minuit2's C++: ``pip install
xrdroot[fit]``. Without it, fits linear in their parameters still work.
"""

from __future__ import annotations

from .hfit import fit_object
from .minuit import minimize
from .options import FitOptions, parse
from .result import FitResult
from .unbinned import unbinned

__all__ = ["FitResult", "FitOptions", "fit_object", "minimize", "parse", "unbinned"]
