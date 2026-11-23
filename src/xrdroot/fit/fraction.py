"""Fitting the fractions of Monte Carlo templates to a data histogram - ROOT's ``TFractionFitter``.

The data's bins are Poisson counts, and so is every template's, so the fit varies
both within their statistics: the likelihood is Barlow and Beeston's (Comp. Phys.
Comm. 77 (1993) 219), and its per-bin template factors are solved for analytically
rather than handed to Minuit, leaving one free parameter per template - the
fraction of the data it accounts for. Minuit then minimises that profile
likelihood, which is a negative log-likelihood and so has an error definition of
0.5 rather than 1: one standard deviation is an increase of one half.

What the fit answers with is not the weighted sum of the templates - the factors
of the per-bin solve are what the data prefers, and they are the "smeared"
templates :meth:`FractionFit.predictions` hands back.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .minuit import minimize, minuit_messages
from .result import FitResult

__all__ = ["FractionFit"]

#: The error definition of a negative log-likelihood: ``TFractionFitter``'s ``SetErrorDef(0.5)``.
ERRORDEF = 0.5
#: ROOT's first step for a fraction parameter (``TFractionFitter``'s ``defaultStep``).
STEP = 0.01
#: ``FindPrediction``'s largest Newton step, and how many turns the solve may take.
NEWTON_STEP = 0.2
NEWTON_TURNS = 100000
#: What counts as a solved bin for ``FindPrediction``: the function's value, and its step.
SOLVED = 1e-12
STOPPED = 1e-13


class FractionFit:
    """The profile likelihood of some data's bin contents as fractions of some templates.

    ``data`` holds the contents of the bins being fitted, ``templates`` one row per
    template over the same bins, and ``weights`` - optional - one row per template
    of per-bin factors, as ``TFractionFitter::SetWeight`` sets them. Every array is
    plain numbers: which bins those are, and what a histogram is, belongs to the
    caller (see :class:`xrdroot.pyroot.core.fractionfit.TFractionFitter`).
    """

    __slots__ = ("data", "templates", "weights", "integral", "integrals", "measured")

    def __init__(self, data: Any, templates: Any, weights: Any = None) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.templates = np.asarray(templates, dtype=np.float64).reshape(-1, self.data.size)
        if weights is None:
            self.weights = np.ones_like(self.templates)
        else:
            self.weights = np.asarray(weights, dtype=np.float64).reshape(self.templates.shape)
        self.integral = float(self.data.sum())
        #: Each template's integral over the fitted bins, weights left out, as ROOT has it.
        self.integrals = self.templates.sum(axis=1)
        #: Each template's integral over the fitted bins as weighted: ``ComputeFCN``'s ``tot``.
        self.measured = (self.weights * self.templates).sum(axis=1)

    @property
    def npar(self) -> int:
        """The number of fractions, one per template."""
        return int(self.templates.shape[0])

    def fractions(self, params: Any) -> np.ndarray[Any, Any]:
        """What each fit parameter is a fraction of: the data's integral over the template's own."""
        return np.asarray(params, dtype=np.float64) * self.integral / self.measured

    def objective(self, params: Any) -> float:
        """The negative log-likelihood of the data and the templates at their fitted bins.

        Both terms ROOT's ``ComputeFCN`` sums: each template bin's Poisson likelihood
        at the factor the per-bin solve found for it, and the data's at the total
        those factors predict.
        """
        total, fitted = self.predictions(params)
        spread = fitted > 0
        counted = (self.templates[spread] * np.log(fitted[spread])).sum() - fitted.sum()
        seen = (self.data > 0) & (total > 0)
        counted += (self.data[seen] * np.log(total[seen])).sum() - total.sum()
        return -float(counted)

    def predictions(self, params: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """The data each bin is predicted to hold, and each template as the fit smears it.

        ``GetPlot`` draws the first and ``GetMCPrediction`` the second: a template's
        fitted bin is its content over one plus its strength times the bin's
        ``t_i``, or - where the bin's leading source holds no events - the share
        ``A_ki`` the per-bin solve worked out instead.
        """
        fractions = self.fractions(params)
        strengths = self.weights * fractions[:, None]
        t, leading, shared = self.solutions(strengths)
        same = (leading >= 0)[None, :] & (
            fractions[:, None] == fractions[np.maximum(leading, 0)][None, :]
        )
        fitted = np.zeros_like(self.templates)
        np.divide(
            self.templates,
            1.0 + self.weights * fractions[:, None] * t[None, :],
            out=fitted,
            where=self.templates > 0,
        )
        fitted = np.where(same, shared[None, :], fitted)
        total = (fractions[:, None] * self.weights * fitted).sum(axis=0)
        return total, fitted

    def solutions(self, strengths: Any) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Any]:
        """``FindPrediction`` for every fitted bin: ``t_i``, the leading source and ``A_ki``.

        The leading source is the one with the largest strength, which is ``-1`` where
        none of them is - a bin the data leaves empty, or one solved by Newton's method.

        A bin the data leaves empty needs no solve - its prediction is the data's
        own, zero - and a bin whose leading sources hold no events at all is
        solved in closed form, as the paper's paragraph 5 has it; every other bin
        has its ``t_i`` found by Newton's method.
        """
        empty = np.floor(self.data + 0.5) == 0  # TMath::Nint(d_i) == 0, as ROOT tests it
        top = strengths.max(axis=0)
        closed, leading, shared = self._closed(strengths, top, strengths == top)
        t = np.where(closed, -1.0 / top, 0.0)
        newton = ~empty & ~closed
        t[newton] = self._newton(
            self.data[newton], self.templates[:, newton], strengths[:, newton], top[newton]
        )
        t[empty] = 1.0
        return t, np.where(closed, leading, -1), shared

    def _closed(self, strengths: Any, top: Any, leading: Any) -> tuple[Any, Any, Any]:
        """The bins whose leading sources hold no events: whether it works, and its ``A_ki``.

        ROOT's ``FindPrediction`` asks for the largest strength ``p_j`` and the
        sources sharing it; when none of those sources has an event in the bin, the
        bin is predicted from the ones that do. A negative ``A_ki`` means the data
        cannot be described that way, and the bin goes to Newton's method instead.
        """
        gap = top - strengths
        with np.errstate(divide="ignore", invalid="ignore"):
            others = np.where(leading, 0.0, self.templates * strengths / gap).sum(axis=0)
            shared = (self.data / (1.0 + top) - others) / leading.sum(axis=0)
        found = (self.templates * leading).sum(axis=0) == 0  # contentsMax == 0
        found &= (shared > 0) & (top > 0)
        return found, np.argmax(strengths, axis=0), np.where(found, shared, 0.0)

    def _newton(
        self, data: Any, counts: Any, strengths: Any, top: Any, turns: int = NEWTON_TURNS
    ) -> np.ndarray[Any, Any]:
        """Solve each bin's ``t_i`` by Newton's method, as ``FindPrediction`` does.

        From zero with a step of at most 0.2, restarting from zero with a tenth of
        the step whenever ``t_i`` leaves the range the paper allows it: below
        ``-1/p_max`` the equation has no solution, and at 1 its right-hand side
        diverges. ``t_i`` cannot be smaller than that bound, so this is what keeps
        the iteration inside the interval where the likelihood is defined. ``turns`` is
        how many of them it may take before giving up, as ROOT warns that it has.
        """
        t = np.zeros(strengths.shape[1])
        step = np.full(strengths.shape[1], NEWTON_STEP)
        floor = -1.0 / top
        todo = np.ones(strengths.shape[1], dtype=bool)
        for _ in range(turns):
            if not todo.any():
                break
            out = todo & ((t >= 1.0) | (t < floor))
            t[out] = 0.0
            step[out] /= 10.0
            t, todo = self._turn(data, counts, strengths, t, step, todo)
        return t

    def _turn(
        self, data: Any, counts: Any, strengths: Any, t: Any, step: Any, todo: Any
    ) -> tuple[np.ndarray[Any, Any], Any]:
        """One Newton step for the bins still being solved, and the bins still to solve after it."""
        value = -data / (1.0 - t)
        slope = value / (1.0 - t)
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # a strength of zero leaves its template out
            rate = 1.0 / (t + 1.0 / strengths)
            value += (counts * rate).sum(axis=0)
            slope -= (counts * rate * rate).sum(axis=0)
            settled = todo & (np.abs(value) < SOLVED)
            delta = np.clip(np.where(settled, 0.0, -value / slope), -step, step)
        moved = todo & ~settled
        t = np.where(moved, t + delta, t)
        done = settled | (moved & (np.abs(delta) < STOPPED))
        return t, todo & ~done

    def chisquare(self, total: Any, fitted: Any) -> float:
        """``ComputeChisquareLambda``: -2 ln(L of the fit / L of the data as its own model).

        The ratio is 1 for a perfect fit, so this is a goodness of fit that follows
        a chi-square distribution with the fit's degrees of freedom.
        """
        total = np.asarray(total, dtype=np.float64)
        fitted = np.asarray(fitted, dtype=np.float64)
        predicted = float(_ln(total, self.data).sum() + _ln(fitted, self.templates).sum())
        measured = float(
            _ln(self.data, self.data).sum() + _ln(self.templates, self.templates).sum()
        )
        return -2.0 * predicted + 2.0 * measured

    def solve(self, *, errors: Any = None, limits: Sequence[Any] | None = None) -> FitResult:
        """Minuit on the profile likelihood from equal fractions, as ROOT's ``Fit`` does."""
        start = np.full(self.npar, 1.0 / self.npar)
        with minuit_messages():  # ROOT's own tests read Minuit2's warnings off this fit
            return minimize(
                self.objective,
                start,
                names=tuple(f"frac{par}" for par in range(self.npar)),
                errors=np.full(self.npar, STEP) if errors is None else errors,
                limits=limits,
                errordef=ERRORDEF,
            )


def _ln(values: Any, counts: Any) -> np.ndarray[Any, Any]:
    """``counts * ln(values) - values`` per bin, as ROOT sums it: an empty value adds nothing."""
    values = np.asarray(values, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    nonzero = values != 0
    with np.errstate(divide="ignore"):
        logged = np.where(nonzero, np.log(np.where(nonzero, values, 1.0)), 0.0)
    return np.where(nonzero, counts * logged - values, 0.0)
