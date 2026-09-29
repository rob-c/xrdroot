"""``MCMCCalculator`` and ``MCMCInterval``: a Bayesian interval from a Markov chain.

The calculator runs Metropolis-Hastings over the negative log-likelihood -
times the prior, if the model has one - in every free parameter; the
interval is the chain's, its burn-in dropped: the shortest - the highest
bins of the chain's histogram holding the confidence level - or a tail
fraction's, walked in from each end of the chain's points sorted by the
parameter, as ``MCMCInterval`` walks them.
"""

from __future__ import annotations

import math
from typing import Any

from ..roofit.cmdargs import RooCmdArg
from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, log
from .intervals import ConfInterval, Named, same_parameters
from .markov import MetropolisHastings, UniformProposal, kLog, kNegative
from .utils import RemoveConstantParameters

__all__ = ["MCMCCalculator", "MCMCInterval"]

#: ``MCMCInterval::IntervalType``.
kShortest, kTailFraction = 0, 1


class MCMCInterval(ConfInterval):
    """The credible interval of a Markov chain's points, its burn-in dropped."""

    kShortest, kTailFraction = kShortest, kTailFraction

    def __init__(self, name: Any = "", parameters: Any = None, chain: Any = None) -> None:
        super().__init__(name)
        self._chain = chain
        self._params = RooArgSet(as_list(parameters)) if parameters is not None else RooArgSet()
        self._axes = list(self._params)
        self._burn_in = 0
        self._type = kShortest
        self._left = -1.0
        self._cl = 0.0
        self._use_keys = self._use_sparse = False
        self._hist_strict = True
        self._cutoff, self._hist_cl = -1.0, 0.0
        self._tf = (-math.inf, math.inf)
        self._tf_cl = 0.0
        self._vector: list[int] = []
        self._vec_weight = 0.0
        self._bins: Any = None

    # -- settings -----------------------------------------------------------------

    def SetNumBurnInSteps(self, steps: int) -> None:
        self._burn_in = int(steps)

    def GetNumBurnInSteps(self) -> int:
        return self._burn_in

    def SetIntervalType(self, kind: int) -> None:
        self._type = int(kind)

    def GetIntervalType(self) -> int:
        return self._type

    def SetLeftSideTailFraction(self, fraction: float) -> None:
        self._left = float(fraction)

    def SetUseKeys(self, flag: bool) -> None:
        self._use_keys = bool(flag)

    def SetUseSparseHist(self, flag: bool) -> None:
        self._use_sparse = bool(flag)

    def SetHistStrict(self, flag: bool) -> None:
        self._hist_strict = bool(flag)

    def SetEpsilon(self, epsilon: float) -> None:
        self._epsilon = float(epsilon)

    def SetDelta(self, delta: float) -> None:
        self._delta = float(delta)

    def SetAxes(self, axes: Any) -> None:
        given = as_list(axes)
        if len(given) != len(self._axes):
            log(None, ERROR, "InputArguments", "* Error in MCMCInterval::SetAxes: number of "
                f"variables in axes ({len(given)}) doesn't match number of parameters "
                f"({len(self._axes)})")  # fmt: skip
            return
        self._axes = list(given)

    def GetAxes(self) -> Any:
        from ..roofit.collections import RooArgList

        return RooArgList(self._axes)

    def GetChain(self) -> Any:
        return self._chain

    def GetParameters(self) -> Any:
        return RooArgSet(list(self._params))

    def SetConfidenceLevel(self, cl: float) -> None:
        self._cl = float(cl)
        self._determine()

    def ConfidenceLevel(self) -> float:
        return self._cl

    # -- the shortest interval: the chain's histogram -----------------------------

    def _refuse_unported(self) -> None:
        from ..errors import UnsupportedFeatureError

        if self._use_keys or self._use_sparse:
            what = "a keys density (RooNDKeysPdf)" if self._use_keys else "a sparse histogram"
            raise UnsupportedFeatureError(
                f"MCMCInterval finds its shortest interval from {what} in ROOT; xrdroot finds it "
                "from the chain's binned histogram only - leave SetUseKeys and SetUseSparseHist off"
            )

    def _histogram(self) -> Any:
        """``CreateDataHist``: the chain's weights binned in the parameters, burn-in dropped -
        each bin's centre and weight, the first parameter slowest, as RooDataHist orders them."""
        import itertools

        import numpy as np

        if self._burn_in >= self._chain.Size():
            log(None, ERROR, "InputArguments", "MCMCInterval::CreateDataHist: creation of "
                "histogram failed: Number of burn-in steps (num steps to ignore) >= number of "
                "steps in Markov chain.")  # fmt: skip
            return None
        edges = [np.asarray(p.getBinning().array(), dtype=np.float64) for p in self._params]
        count = int(np.prod([len(e) - 1 for e in edges]))
        weights = np.bincount(self._bin_index(edges), np.asarray(self._chain.weights(
            self._burn_in)), count)  # fmt: skip
        binnings = [p.getBinning() for p in self._params]  # RooDataHist's centres, as it has them
        centres = list(itertools.product(*[[b.binCenter(i) for i in range(b.numBins())]
                                           for b in binnings]))  # fmt: skip
        return centres, [float(w) for w in weights]

    def _bin_index(self, edges: list[Any]) -> Any:
        """Each step's bin, the first parameter's slowest."""
        import numpy as np

        index = np.zeros(self._chain.Size() - self._burn_in, dtype=np.int64)
        for par, edge in zip(self._params, edges):
            values = np.asarray(self._chain.values(par.GetName(), self._burn_in))
            found = np.clip(np.searchsorted(edge, values, side="right") - 1, 0, len(edge) - 2)
            index = index * (len(edge) - 1) + found
        return index

    def _by_hist(self) -> None:
        """``DetermineByDataHist``: the weight the highest bins must reach to hold the level."""
        self._refuse_unported()
        if self._bins is None:
            self._bins = self._histogram()
        if self._bins is None:
            self._cutoff, self._hist_cl = -1.0, 0.0
            return
        weights = self._bins[1]
        order = sorted(range(len(weights)), key=lambda i: weights[i])  # stable, as C++'s
        total = sum(weights)
        running, i = 0.0, len(order) - 1
        while i >= 0:
            content = weights[order[i]]
            if (running + content) / total >= self._cl:
                self._cutoff = content
                if self._hist_strict:
                    running += content
                    i -= 1
                else:
                    i += 1
                break
            running += content
            i -= 1
        running = self._settle(weights, order, i, running)
        self._hist_cl = running / total

    def _settle(self, weights: list[float], order: list[int], i: int, running: float) -> float:
        """The bins tied with the cutoff: counted in, when strict - else the cutoff raised."""
        if self._hist_strict:
            while i >= 0 and weights[order[i]] == self._cutoff:
                running += weights[order[i]]
                i -= 1
            return running
        while i < len(order):
            content = weights[order[i]]
            if content > self._cutoff:
                self._cutoff = content
                break
            running -= content
            if i == len(order) - 1:
                self._cutoff = content + 1.0
            i += 1
        return running

    def _hist_limit(self, param: Any, upper: bool) -> float:
        if self._cutoff < 0:
            self._by_hist()
        names = [p.GetName() for p in self._params]
        if self._cutoff < 0:
            side, end = ("Upper", "getMax") if upper else ("Lower", "getMin")
            log(None, ERROR, "Eval", f"In MCMCInterval::{side}LimitByDataHist: couldn't determine "
                "cutoff.  Check that num burn in steps < num steps in the Markov chain.  "
                f"Returning param.{end}().")  # fmt: skip
        if self._cutoff < 0 or param.GetName() not in names:
            return float(param.getMax() if upper else param.getMin())
        return self._extreme_bin(names.index(param.GetName()), param, upper)

    def _extreme_bin(self, at: int, param: Any, upper: bool) -> float:
        """The farthest centre, in the parameter, of the bins above the cutoff."""
        chosen = [c[at] for c, w in zip(*self._bins) if w >= self._cutoff]
        if upper:
            return float(max([param.getMin(), *chosen]))
        return float(min([param.getMax(), *chosen]))

    # -- the tail-fraction interval -----------------------------------------------

    def _tail_fraction(self) -> None:
        """``DetermineTailFractionInterval``: each tail walked in, point by point, while its
        weight gets closer to its share of what the level leaves out."""
        if not 0 <= self._left <= 1:
            log(None, ERROR, "InputArguments", "MCMCInterval::DetermineTailFractionInterval: "
                f"Fraction must be in the range [0, 1].  {self._left:g}is not "
                "allowed.")  # fmt: skip
            return
        if len(self._axes) != 1:
            log(None, ERROR, "InputArguments", "MCMCInterval::DetermineTailFractionInterval(): "
                "Error: Can only find a tail-fraction interval for 1-D intervals")  # fmt: skip
            return
        param = self._axes[0]
        values = self._chain.values(param.GetName())
        weights = self._chain.weights()
        if not self._vector:
            self._make_vector(values, weights)
        if not self._vector or self._vec_weight == 0:
            self._vector, self._tf = [], (-math.inf, math.inf)
            self._tf_cl, self._vec_weight = 0.0, 0.0
            return
        missing = self._vec_weight * (1 - self._cl)
        low, left = _walk([(values[i], weights[i]) for i in self._vector], missing * self._left,
                          param.getMin())  # fmt: skip
        high, right = _walk([(values[i], weights[i]) for i in reversed(self._vector)],
                            missing * (1 - self._left), param.getMax())  # fmt: skip
        self._tf = (low, high)
        self._tf_cl = 1 - (left + right) / self._vec_weight

    def _make_vector(self, values: Any, weights: Any) -> None:
        """``CreateVector``: the steps after burn-in in order of the parameter, and their weight."""
        if self._burn_in >= self._chain.Size():
            log(None, ERROR, "InputArguments", "MCMCInterval::CreateVector: creation of vector "
                "failed: Number of burn-in steps (num steps to ignore) >= number of steps in "
                "Markov chain.")  # fmt: skip
        chosen = range(self._burn_in, self._chain.Size())
        self._vector = sorted(chosen, key=lambda i: values[i])
        self._vec_weight = _running_sum(weights[i] for i in chosen)

    def _determine(self) -> None:
        if self._type == kShortest:
            self._by_hist()
        elif self._type == kTailFraction:
            self._tail_fraction()
        else:
            log(None, ERROR, "InputArguments", "MCMCInterval::DetermineInterval(): Error: "
                "Interval type not set")  # fmt: skip

    # -- answers ------------------------------------------------------------------

    def LowerLimit(self, param: Any) -> float:
        if self._type == kShortest:
            return self._hist_limit(param, False)
        if self._tf[0] == -math.inf:
            self._tail_fraction()
        return self._tf[0]

    def UpperLimit(self, param: Any) -> float:
        if self._type == kShortest:
            return self._hist_limit(param, True)
        if self._tf[1] == math.inf:
            self._tail_fraction()
        return self._tf[1]

    def GetActualConfidenceLevel(self) -> float:
        return self._hist_cl if self._type == kShortest else self._tf_cl

    def GetHistCutoff(self) -> float:
        if self._cutoff < 0:
            self._by_hist()
        return self._cutoff

    def IsInInterval(self, point: Any) -> bool:
        given = RooArgSet(as_list(point))
        if self._type == kTailFraction:
            x = given.getRealValue(self._axes[0].GetName())
            return bool(self._vector) and self._tf[0] <= x <= self._tf[1]
        if self._bins is None:
            return False
        # ``RooDataHist::getIndex``: the bin the point is in - one always is, the bins clamped
        weight = next(w for centre, w in zip(*self._bins)
                      if all(_in_bin(p, c, given) for p, c in zip(self._params, centre)))
        return bool(weight >= self._cutoff)

    def CheckParameters(self, point: Any) -> bool:
        return same_parameters(point, self._params)

    def GetPosteriorHist(self) -> Any:
        """The chain's weights histogrammed in the parameters, burn-in dropped: a ``TH1F``."""
        from ..pyroot.roostats.mcmcplot import posterior_hist

        return posterior_hist(self)


def _running_sum(values: Any) -> float:
    total = 0.0
    for one in values:
        total += one
    return total


def _walk(points: list[tuple[float, float]], cutoff: float, start: float) -> tuple[float, float]:
    """One tail: points taken while the tail's weight gets closer to ``cutoff``."""
    limit, total = start, 0.0
    for x, w in points:
        if abs(total + w - cutoff) < abs(total - cutoff):
            limit, total = x, total + w
        else:
            break
    return limit, total


def _in_bin(param: Any, centre: float, point: Any) -> bool:
    binning = param.getBinning()
    x = point.getRealValue(param.GetName())
    index = binning.binNumber(x)
    return bool(abs(binning.binCenter(index) - centre) < 1e-12 * max(1.0, abs(centre)))


class MCMCCalculator(Named):
    """Bayesian intervals from a Markov chain over the likelihood - times the prior, if any."""

    def __init__(self, data: Any = None, model: Any = None) -> None:
        super().__init__("")
        self._data = data
        self._pdf: Any = None
        self._prior: Any = None
        self._sets = {k: RooArgSet() for k in ("poi", "nuis", "cond", "glob", "chain")}
        self._proposal: Any = None
        self._iters, self._burn_in, self._bins = 10000, 40, 50
        self._keys = self._sparse = False
        self._size, self._type, self._left = 0.05, kShortest, -1.0
        self._epsilon = self._delta = -1.0
        self._axes: Any = None
        if model is not None:
            self.SetModel(model)

    def SetModel(self, model: Any) -> None:
        self._pdf, self._prior = model.GetPdf(), model.GetPriorPdf()
        for key, found in (("poi", model.GetParametersOfInterest()),
                           ("nuis", model.GetNuisanceParameters()),
                           ("cond", model.GetConditionalObservables()),
                           ("glob", model.GetGlobalObservables())):  # fmt: skip
            self._sets[key] = RooArgSet(as_list(found)) if found is not None else RooArgSet()

    def SetData(self, data: Any) -> None:
        self._data = data

    def SetPdf(self, pdf: Any) -> None:
        self._pdf = pdf

    def SetPriorPdf(self, pdf: Any) -> None:
        self._prior = pdf

    def SetParameters(self, items: Any) -> None:
        self._sets["poi"] = RooArgSet(as_list(items))

    def SetNuisanceParameters(self, items: Any) -> None:
        self._sets["nuis"] = RooArgSet(as_list(items))

    def SetChainParameters(self, items: Any) -> None:
        self._sets["chain"] = RooArgSet(as_list(items))

    def SetProposalFunction(self, proposal: Any) -> None:
        self._proposal = proposal

    def SetNumIters(self, iters: int) -> None:
        self._iters = int(iters)

    def SetNumBurnInSteps(self, steps: int) -> None:
        self._burn_in = int(steps)

    def SetNumBins(self, bins: int) -> None:
        self._bins = int(bins)

    def SetAxes(self, axes: Any) -> None:
        self._axes = axes

    def SetUseKeys(self, flag: bool) -> None:
        self._keys = bool(flag)

    def SetUseSparseHist(self, flag: bool) -> None:
        self._sparse = bool(flag)

    def SetIntervalType(self, kind: int) -> None:
        self._type = int(kind)

    def SetLeftSideTailFraction(self, fraction: float) -> None:
        if not 0 <= fraction <= 1:
            log(None, ERROR, "InputArguments", "MCMCCalculator::SetLeftSideTailFraction: "
                f"Fraction must be in the range [0, 1].  {fraction:g}is not allowed.")  # fmt: skip
            return
        self._left, self._type = float(fraction), kTailFraction

    def SetEpsilon(self, epsilon: float) -> None:
        self._epsilon = float(epsilon)

    def SetDelta(self, delta: float) -> None:
        self._delta = float(delta)

    def SetTestSize(self, size: float) -> None:
        self._size = float(size)

    def SetConfidenceLevel(self, cl: float) -> None:
        self._size = 1.0 - float(cl)

    def Size(self) -> float:
        return self._size

    def ConfidenceLevel(self) -> float:
        return 1.0 - self._size

    def _likelihood(self) -> Any:
        """The chain's function: the model's - times the prior's - negative log-likelihood."""
        pdf = self._pdf
        if self._prior is not None:
            from ..roofit.pdfs.prodpdf import RooProdPdf

            name = f"product_{self._pdf.GetName()}_{self._prior.GetName()}"
            pdf = RooProdPdf(name, name, [self._pdf, self._prior])
        constrained = RooArgSet(list(pdf.getParameters(self._data)))
        return pdf.createNLL(self._data, RooCmdArg("Constrain", constrained),
                             RooCmdArg("ConditionalObservables", self._sets["cond"]),
                             RooCmdArg("GlobalObservables", self._sets["glob"]))  # fmt: skip

    def GetInterval(self) -> Any:
        """The chain, then its interval."""
        if self._data is None or self._pdf is None or not len(self._sets["poi"]):
            return None
        if self._size < 0:
            log(None, ERROR, "InputArguments", "MCMCCalculator::GetInterval: Test size/"
                "Confidence level not set.  Returning nullptr.")  # fmt: skip
            return None
        proposal = self._proposal if self._proposal is not None else UniformProposal()
        nll = self._likelihood()
        params = RooArgSet(list(nll.getParameters(self._data)))
        RemoveConstantParameters(params)
        if self._bins > 0:
            self._set_bins(params, proposal)
        return self._interval(self._chain(nll, params, proposal))

    def _set_bins(self, params: Any, proposal: Any) -> None:
        """``SetNumBins``: the parameters' bins, and those of a proposal density's."""
        for one in [*params, *self._sets["poi"]]:
            one.setBins(self._bins)
        pdf = getattr(proposal, "GetPdf", None)
        for one in (pdf().getParameters(None) if pdf is not None else ()):
            one.setBins(self._bins)

    def _chain(self, nll: Any, params: Any, proposal: Any) -> Any:
        mh = MetropolisHastings()
        mh.SetFunction(nll)
        mh.SetType(kLog)
        mh.SetSign(kNegative)
        mh.SetParameters(params)
        if len(self._sets["chain"]):
            mh.SetChainParameters(self._sets["chain"])
        mh.SetProposalFunction(proposal)
        mh.SetNumIters(self._iters)
        return mh.ConstructChain()

    def _interval(self, chain: Any) -> Any:
        """The chain's interval, configured as the calculator was."""
        interval = MCMCInterval(f"MCMCInterval_{self._name}", self._sets["poi"], chain)
        if self._axes is not None:
            interval.SetAxes(self._axes)
        if self._burn_in > 0:
            interval.SetNumBurnInSteps(self._burn_in)
        interval.SetUseKeys(self._keys)
        interval.SetUseSparseHist(self._sparse)
        interval.SetIntervalType(self._type)
        if self._type == kTailFraction:
            interval.SetLeftSideTailFraction(self._left)
        for setter, value in (("SetEpsilon", self._epsilon), ("SetDelta", self._delta)):
            if value >= 0:
                getattr(interval, setter)(value)
        interval.SetConfidenceLevel(1.0 - self._size)
        return interval
