"""Markov chains of parameter points: the proposals, ``MarkovChain`` and ``MetropolisHastings``.

A proposal moves the current point to a candidate - ``SequentialProposal``
one parameter at a time by a Gaussian step wrapped round its range,
``UniformProposal`` anywhere in the ranges; Metropolis-Hastings accepts it
with the probability the likelihood ratio says, from RooFit's generator, and
the chain keeps each point with the number of steps it stayed there as its
weight - as RooStats draws and keeps them, number for number.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..roofit.collections import RooArgSet, as_list
from ..roofit.messages import ERROR, INFO, PROGRESS, log, log_plain
from ..roofit.rng import generator
from .intervals import Named
from .modelconfig import quieted

__all__ = ["MarkovChain", "MetropolisHastings", "ProposalFunction", "SequentialProposal",
           "UniformProposal", "randomize"]  # fmt: skip


def randomize(items: Any) -> None:
    """``RandomizeCollection``: every variable drawn uniformly over its range."""
    for var in as_list(items):
        var.randomize()


class ProposalFunction:
    """What proposes the chain's next point from its current one."""

    def Propose(self, xPrime: Any, x: Any) -> None:
        raise NotImplementedError

    def IsSymmetric(self, x1: Any, x2: Any) -> bool:
        return True

    def GetProposalDensity(self, x1: Any, x2: Any) -> float:
        return 1.0


class UniformProposal(ProposalFunction):
    """A new point anywhere in the parameters' ranges."""

    def Propose(self, xPrime: Any, x: Any) -> None:
        randomize(xPrime)

    def GetProposalDensity(self, x1: Any, x2: Any) -> float:
        volume = 1.0
        for var in as_list(x2):
            volume *= var.getMax() - var.getMin()
        return 1.0 / volume


class SequentialProposal(ProposalFunction):
    """One parameter, picked at random, moved by a Gaussian of its range over ``divisor``,
    wrapped round the range."""

    def __init__(self, divisor: float = 0.1) -> None:
        self._fraction = 1.0 / float(divisor)

    def Propose(self, xPrime: Any, x: Any) -> None:
        RooArgSet(as_list(xPrime)).assign(x)
        chosen = as_list(xPrime)
        j = math.floor(generator().Rndm() * len(chosen))
        var = chosen[j]
        low, high = var.getMin(), var.getMax()
        length = high - low
        value = var.getVal() + generator().Gaus() * length * self._fraction
        while value > high:
            value -= length
        while value < low:
            value += length
        var.setVal(value)


#: ``MarkovChain``'s names for its weight, its likelihood and its dataset.
WEIGHT, NLL = "weight_MarkovChain_local_", "nll_MarkovChain_local_"


class MarkovChain(Named):
    """The chain's points, each with its likelihood and its weight - the steps it stayed."""

    def __init__(self, *args: Any) -> None:
        strings = [a for a in args if isinstance(a, str)]
        super().__init__(strings[0] if strings else "_markov_chain",
                         strings[1] if len(strings) > 1 else "Markov Chain")  # fmt: skip
        self._params: Any = None
        self._rows: list[list[float]] = []
        self._weights: list[float] = []
        self._nlls: list[float] = []
        given = [a for a in args if not isinstance(a, str)]
        if given:
            self.SetParameters(given[0])

    def SetParameters(self, parameters: Any) -> None:
        self._params = RooArgSet(as_list(parameters)).snapshot()
        self._rows, self._weights, self._nlls = [], [], []

    def Add(self, entry: Any, nllValue: float, weight: float = 1.0) -> None:
        if self._params is None:
            self.SetParameters(entry)
        mine = RooArgSet(as_list(entry))
        self._rows.append([float(mine.getRealValue(p.GetName())) for p in self._params])
        self._nlls.append(float(nllValue))
        self._weights.append(float(weight))

    def Size(self) -> int:
        return len(self._rows)

    def Get(self, i: Any = None) -> Any:
        """The ``i``-th point, loaded into the chain's own copies of the parameters."""
        index = self._current if i is None else int(i)
        self._current = index
        for par, value in zip(self._params, self._rows[index]):
            par.setVal(value)
        return self._params

    def Weight(self, i: Any = None) -> float:
        return self._weights[self._current if i is None else int(i)]

    def NLL(self, i: Any = None) -> float:
        return self._nlls[self._current if i is None else int(i)]

    def GetParameters(self) -> Any:
        return self._params

    def values(self, name: str, start: int = 0) -> list[float]:
        """Every point's value of the parameter ``name``, from ``start`` on."""
        index = [p.GetName() for p in self._params].index(name)
        return [row[index] for row in self._rows[start:]]

    def weights(self, start: int = 0) -> list[float]:
        return list(self._weights[start:])

    _current = 0


#: ``MetropolisHastings::FunctionType`` and ``FunctionSign``.
kRegular, kLog, kTypeUnset = 0, 1, 2
kNegative, kPositive, kSignUnset = 0, 1, 2


def _evaluated(function: Any) -> tuple[float, bool]:
    """The function's value, and whether it failed - a NaN, where RooFit logs an error."""
    value = float(function.getVal())
    return value, value != value


class MetropolisHastings:
    """The Metropolis-Hastings algorithm over a (log-)likelihood and a proposal function."""

    kRegular, kLog, kNegative, kPositive = kRegular, kLog, kNegative, kPositive

    def __init__(self, function: Any = None, paramsOfInterest: Any = None,
                 proposalFunction: Any = None, numIters: int = 0) -> None:  # fmt: skip
        self._function = function
        self._params = RooArgSet(as_list(paramsOfInterest)) if paramsOfInterest else RooArgSet()
        self._chain_params = RooArgSet()
        self._proposal = proposalFunction
        self._iters = int(numIters)
        self._type, self._sign = kTypeUnset, kSignUnset

    def SetFunction(self, function: Any) -> None:
        self._function = function

    def SetParameters(self, params: Any) -> None:
        self._params = RooArgSet(as_list(params))

    def SetChainParameters(self, params: Any) -> None:
        self._chain_params = RooArgSet(as_list(params))

    def SetProposalFunction(self, proposal: Any) -> None:
        self._proposal = proposal

    def SetNumIters(self, iters: int) -> None:
        self._iters = int(iters)

    def SetType(self, kind: int) -> None:
        self._type = int(kind)

    def SetSign(self, sign: int) -> None:
        self._sign = int(sign)

    def _start(self, x: Any) -> float:
        """A starting point where the function can be evaluated - a thousand tries at most."""
        value, bad = 0.0, True
        for _ in range(1000):
            if not bad:
                break
            randomize(x)
            self._params.assign(x)
            value, failed = _evaluated(self._function)
            bad = failed if self._type == kLog else (value == 0.0 if self._type == kRegular
                                                      else False)  # fmt: skip
        if bad:
            log(None, ERROR, "Eval", "Problem finding a good starting point in "
                "MetropolisHastings::ConstructChain() ")  # fmt: skip
        return value

    def _nll(self, value: float) -> float:
        """``CalcNLL``: the chain's negative log-likelihood of a function value."""
        if self._type == kLog:
            return value if self._sign == kNegative else -value
        return -_c(np.log, value if self._sign == kPositive else -value)

    def _take(self, a: float) -> bool:
        """``ShouldTakeStep``: always uphill, else with the ratio's probability."""
        if (self._type == kLog and a <= 0.0) or (self._type == kRegular and a >= 1.0):
            return True
        rand = generator().Rndm()
        if self._type == kLog:
            return -math.log(rand) >= a
        return rand < a

    def _ratio(self, x_value: float, candidate: float) -> float:
        if self._type == kLog:
            return x_value - candidate if self._sign == kPositive else candidate - x_value
        return _c(np.divide, candidate, x_value)

    def ConstructChain(self) -> Any:
        """The chain: ``numIters`` proposals, each taken or not, the points weighted by stays."""
        if not len(self._params) or self._proposal is None or self._function is None:
            log(None, ERROR, "Eval", "Critical members uninitialized: parameters, proposal  "
                "function, or (log) likelihood function")  # fmt: skip
            return None
        if self._sign == kSignUnset or self._type == kTypeUnset:
            log(None, ERROR, "Eval", "Please set type and sign of your function using "
                "MetropolisHastings::SetType() and MetropolisHastings::SetSign()")  # fmt: skip
            return None
        if not len(self._chain_params):
            self._chain_params.add(list(self._params))
        x, candidate = self._params.snapshot(), None
        randomize(x)
        candidate = self._params.snapshot()
        randomize(candidate)
        chain = MarkovChain()
        chain.SetParameters(self._chain_params)
        with quieted(PROGRESS):
            x_value = self._start(x)
            chain = self._steps(chain, x, candidate, x_value)
        with np.errstate(all="ignore"):  # in single precision, as RooStats works it out
            rate = float(np.float32(chain.Size()) / np.float32(self._iters) * np.float32(100))
        log(None, INFO, "Eval", f"Proposal acceptance rate: {g_(rate)}%")
        log(None, INFO, "Eval", f"Number of steps in chain: {chain.Size()}")
        return chain

    def _proposed(self, candidate: Any, x: Any, x_value: float) -> tuple[float, bool, float]:
        """A proposal from ``x``: the function there, whether it failed, the acceptance ratio -
        corrected by the proposal's densities each way if it is not symmetric."""
        self._proposal.Propose(candidate, x)
        self._params.assign(candidate)
        value, failed = _evaluated(self._function)
        failed = failed and self._type == kLog  # an evaluation error, which only kLog heeds
        if failed:
            return math.inf, True, self._ratio(x_value, math.inf)
        a = self._ratio(x_value, value)
        if self._proposal.IsSymmetric(candidate, x):
            return value, False, a
        there = self._proposal.GetProposalDensity(candidate, x)
        back = self._proposal.GetProposalDensity(x, candidate)
        if self._type == kRegular:
            return value, False, a * _c(np.divide, back, there)
        return value, False, a + _c(np.log, there) - _c(np.log, back)

    def _steps(self, chain: MarkovChain, x: Any, candidate: Any, x_value: float) -> MarkovChain:
        log_plain(None, PROGRESS, "Generation", "Metropolis-Hastings progress: ")
        weight = 0
        tick = max(self._iters // 100, 1)
        for i in range(self._iters):
            if i % tick == 0:
                log_plain(None, PROGRESS, "Generation", ".")
            value, failed, a = self._proposed(candidate, x, x_value)
            if not failed and self._take(a):
                if weight != 0:
                    chain.Add(x, self._nll(x_value), float(weight))
                weight = 1
                RooArgSet(as_list(x)).assign(candidate)
                x_value = value
            else:
                weight += 1
        if weight != 0:
            chain.Add(x, self._nll(x_value), float(weight))
        log_plain(None, PROGRESS, "Generation", "\n")
        return chain


def _c(function: Any, *args: float) -> float:
    """``function`` of ``args`` as C works it out: a zero's logarithm or quotient infinite, or
    not a number, where Python would raise."""
    with np.errstate(all="ignore"):
        return float(function(*(np.float64(a) for a in args)))


def g_(value: float) -> str:
    from ..roofit.printing import g

    return g(value)
