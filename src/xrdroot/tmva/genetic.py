"""TMVA's genetic algorithm: ``Interval``, ``GeneticRange``, the population, and ``GeneticFitter``.

A fit's parameters each have an :class:`Interval` - a range, or a range
of ``nbins`` discrete values - and a population of candidate parameter sets
("genes") evolves: the better half kept, the worse half replaced by
children of two parents (each parameter taken from one or the other), then
mutated - some near where they were, by a Gaussian of a spread the
algorithm steers by how often it improves, some anywhere in the range - and
this until the best fitness has not changed by more than ``ConvCrit`` for
``Steps`` generations. All of it is TMVA's code, draw for draw, with its
``TRandom3`` seeded as TMVA seeds it, so a fit here takes the path TMVA's
takes.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..random.mersenne import TRandom3
from .log import Logger
from .options import Options
from .tools import CxxVector

__all__ = ["FactorVector", "GeneticAlgorithm", "GeneticFitter", "IFitterTarget", "Interval"]

#: ``DBL_MAX``, the fitness nothing has yet.
DBL_MAX = 1.7976931348623157e308


#: ``std::vector<Double_t>`` of a fit's parameters, as a fitness function is handed them.
FactorVector = CxxVector


class IFitterTarget:
    """``TMVA::IFitterTarget``: what is minimised - a class overrides ``EstimatorFunction``."""

    def EstimatorFunction(self, parameters: Any) -> float:
        raise NotImplementedError

    def ProgressNotifier(self, *_: Any) -> None:
        """``ProgressNotifier``: told how the fit is getting on, which most targets ignore."""


class Interval:
    """``TMVA::Interval(min, max, nbins=0)``: a range, or ``nbins`` evenly spaced values in it."""

    def __init__(self, low: float, high: float, nbins: int = 0) -> None:
        self.low, self.high, self.nbins = float(low), float(high), int(nbins)
        if self.high - self.low < 0:
            raise Logger("Interval").fatal("maximum lower than minimum")
        if self.nbins < 0 or self.nbins == 1:
            raise Logger("Interval").fatal("interval has to have at least 2 bins if discrete")

    def GetMin(self) -> float:
        return self.low

    def GetMax(self) -> float:
        return self.high

    def GetNbins(self) -> int:
        return self.nbins

    def GetWidth(self) -> float:
        return self.high - self.low

    def GetMean(self) -> float:
        return (self.high + self.low) / 2

    def SetMin(self, value: float) -> None:
        self.low = float(value)

    def SetMax(self, value: float) -> None:
        self.high = float(value)

    def GetElement(self, index: int) -> float:
        """The ``index``-th of the discrete values."""
        if self.nbins <= 0 or not 0 <= index < self.nbins:
            raise Logger("Interval").fatal(f"bin {index} out of range of a {self.nbins}-bin interval")
        return self.low + (float(index) / (self.nbins - 1)) * (self.high - self.low)

    def GetStepSize(self, _: int = 0) -> float:
        return (self.high - self.low) / (self.nbins - 1)


class GeneticRange:
    """``TMVA::GeneticRange``: draws in one parameter's interval, from the population's generator."""

    def __init__(self, random: TRandom3, interval: Interval) -> None:
        self.random, self.interval = random, interval
        self.low, self.high = interval.low, interval.high
        self.length = self.high - self.low

    def random_value(self, near: bool = False, value: float = 0.0, spread: float = 0.1,
                     mirror: bool = False) -> float:
        """``Random``: a discrete value, the one value of an empty range, a Gaussian step, or anywhere."""
        if self.interval.nbins > 0:
            draw = float(self.random.uniform(0.0, 1.0))
            return self.interval.GetElement(int(draw * self.interval.nbins))
        if self.low == self.high:
            return self.low
        if near:
            moved = float(self.random.gaus(value, self.length * spread))
            return self._mirror(moved) if mirror else self._wrap(moved)
        return float(self.random.uniform(self.low, self.high))

    def _wrap(self, value: float) -> float:
        """``ReMap``: a value past one end brought in from the other."""
        while self.low < self.high and not self.low <= value < self.high:
            value = value - self.low + self.high if value < self.low else value - self.high + self.low
        return value

    def _mirror(self, value: float) -> float:
        """``ReMapMirror``: a value past an end reflected back, then wrapped."""
        if self.low >= self.high:
            return value
        if value < self.low:
            return self._wrap(self.low - (value - self.low))
        if value >= self.high:
            return self._wrap(self.high - (value - self.high))
        return value


class Genes:
    """``TMVA::GeneticGenes``: one candidate's parameters and its fitness."""

    __slots__ = ("factors", "fitness")

    def __init__(self, factors: list[float], fitness: float = 0.0) -> None:
        self.factors, self.fitness = list(factors), fitness

    def GetFactors(self) -> CxxVector:
        return CxxVector(self.factors)

    def GetFitness(self) -> float:
        return self.fitness

    def SetFitness(self, fitness: float) -> None:
        self.fitness = float(fitness)


class GeneticPopulation:
    """``TMVA::GeneticPopulation``: the genes, their ranges, and the one generator they draw from."""

    def __init__(self, ranges: list[Interval], size: int, seed: int = 0) -> None:
        self.random = TRandom3(100)
        self.random.uniform(0.0, 1.0)
        self.random.set_seed(seed)
        self.ranges = [GeneticRange(self.random, interval) for interval in ranges]
        self.genes = [Genes([r.random_value() for r in self.ranges]) for _ in range(size)]
        self.limit = size

    def MakeChildren(self) -> None:
        """``MakeChildren``: the worse half replaced by children of the better half."""
        half = len(self.genes) // 2
        for index in range(half):
            partner = int(self.random.integer(half))
            self.genes[half + index] = self.MakeSex(self.genes[index], self.genes[partner])

    def MakeSex(self, male: Genes, female: Genes) -> Genes:
        factors = [
            male.factors[i] if int(self.random.integer(2)) == 0 else female.factors[i]
            for i in range(len(self.ranges))
        ]
        return Genes(factors)

    def Mutate(self, probability: float, start: int, near: bool = False, spread: float = 0.1,
               mirror: bool = False) -> None:
        """``Mutate``: each parameter of each gene from ``start`` on redrawn with ``probability`` %."""
        for genes in self.genes[start:]:
            for i, current in enumerate(genes.factors):
                if float(self.random.uniform(100.0)) <= probability:
                    genes.factors[i] = self.ranges[i].random_value(near, current, spread, mirror)

    def GiveHint(self, factors: list[float], fitness: float) -> None:
        self.genes.append(Genes(factors, fitness))

    def AddPopulation(self, other: GeneticPopulation) -> None:
        for genes in other.genes:
            self.GiveHint(genes.factors, genes.fitness)

    def Sort(self) -> None:
        self.genes.sort(key=lambda genes: genes.fitness)

    def GetGenes(self, index: int) -> Genes:
        return self.genes[index]

    def GetPopulationSize(self) -> int:
        return len(self.genes)

    def SetRandomSeed(self, seed: int) -> None:
        self.random.set_seed(seed)

    def MakeCopies(self, number: int) -> None:
        for genes in self.genes[:number]:
            self.GiveHint(genes.factors, genes.fitness)

    def Print(self, until: int = -1) -> None:
        """``Print(untilIndex)``: each gene's fitness and factors, as many as asked for."""
        log = Logger("GeneticPopulation")
        for genes in self.genes:
            if until >= -1:
                if until == -1:
                    return
                until -= 1
            factors = "".join(f"f_{i}: {value:g}     " for i, value in enumerate(genes.factors))
            log.info(f"fitness: {genes.fitness:g}    {factors}")

    def TrimPopulation(self) -> None:
        """``TrimPopulation``: sorted, and the worst dropped until there are as many as asked for."""
        self.Sort()
        del self.genes[self.limit :]


class GeneticAlgorithm:
    """``TMVA::GeneticAlgorithm``: one population's evolution towards the least fitness."""

    def __init__(self, target: Any, size: int, ranges: list[Interval], seed: int = 0) -> None:
        self.target = target
        self.population = GeneticPopulation(ranges, size, seed)
        self.population.random.set_seed(seed)
        self.spread, self.mirror, self.first = 0.1, True, True
        self.best, self.last_result = DBL_MAX, DBL_MAX
        self.converge_counter, self.converge_value = -1, 0.0
        self.successes: deque[int] = deque()

    def GetGeneticPopulation(self) -> GeneticPopulation:
        return self.population

    def GetSpread(self) -> float:
        return self.spread

    def SetSpread(self, spread: float) -> None:
        self.spread = float(spread)

    def SetMakeCopies(self, copies: bool) -> None:
        self.copies = bool(copies)

    def Evolution(self) -> None:
        """``Evolution``: children made in the worse half, then two rounds of mutation."""
        if getattr(self, "copies", False):
            self.population.MakeCopies(5)
        self.population.MakeChildren()
        self.population.Mutate(10, 3, True, self.spread, self.mirror)
        self.population.Mutate(40, len(self.population.genes) * 3 // 4)

    def Init(self) -> None:
        """``Init``: nothing the first time, a generation's evolution every time after."""
        if self.first:
            self.first = False
            return
        self.Evolution()

    def CalculateFitness(self) -> float:
        """``CalculateFitness``: every gene's fitness from the target; the population sorted by it."""
        self.best = DBL_MAX
        for genes in self.population.genes:
            genes.fitness = float(self.target.EstimatorFunction(FactorVector(genes.factors)))
            self.best = min(self.best, genes.fitness)
        self.population.Sort()
        return self.best

    def SpreadControl(self, steps: int, successes: int, factor: float) -> float:
        """``SpreadControl``: the mutation spread narrowed after too much success, widened after too little."""
        improved = self.best < self.last_result or not self.successes
        if improved:
            self.last_result = self.best
        self.successes.appendleft(1 if improved else 0)
        if len(self.successes) >= steps:
            total = sum(self.successes)
            self.successes.pop()
            if total > successes:
                self.spread /= factor
            elif total < successes:
                self.spread *= factor
        return self.spread

    def HasConverged(self, steps: int, improvement: float) -> bool:
        """``HasConverged``: has the best fitness stayed within ``improvement`` for ``steps`` generations?"""
        if self.converge_counter < 0:
            self.converge_value = self.best
        if abs(self.best - self.converge_value) <= improvement or steps < 0:
            self.converge_counter += 1
        else:
            self.converge_counter = 0
            self.converge_value = self.best
        return self.converge_counter >= steps


class GeneticFitter:
    """``TMVA::GeneticFitter(target, name, ranges, options)``: ``Run`` gives the best parameters."""

    def __init__(self, target: Any, name: Any, ranges: Any, options: Any = "") -> None:
        self.target, self.name = target, str(name)
        self.ranges = list(ranges)
        parsed = Options(options)
        self.pop_size = parsed.integer("PopSize", 300)
        self.steps = parsed.integer("Steps", 40)
        self.cycles = parsed.integer("Cycles", 3)
        self.sc_steps = parsed.integer("SC_steps", 10)
        self.sc_rate = parsed.integer("SC_rate", 5)
        self.sc_factor = parsed.number("SC_factor", 0.95)
        self.conv_crit = parsed.number("ConvCrit", 0.001)
        self.save_generation = parsed.integer("SaveBestGen", 1)
        self.save_cycle = parsed.integer("SaveBestCycle", 10)
        self.trim = parsed.flag("Trim", False)
        self.seed = parsed.integer("Seed", 100)
        self.log = Logger("FitterBase")

    def SetParameters(self, cycles: int, steps: int, pop_size: int, sc_steps: int, sc_rate: int,
                      sc_factor: float, conv_crit: float) -> None:
        self.cycles, self.steps, self.pop_size = int(cycles), int(steps), int(pop_size)
        self.sc_steps, self.sc_rate = int(sc_steps), int(sc_rate)
        self.sc_factor, self.conv_crit = float(sc_factor), float(conv_crit)

    def _cycle(self, store: GeneticAlgorithm, parameters: Any, last: bool) -> None:
        """One independent cycle of the algorithm, its best genes saved into ``store``."""
        ga = GeneticAlgorithm(self.target, self.pop_size, self.ranges, self.seed)
        if len(parameters) == len(self.ranges):
            ga.population.GiveHint(list(parameters), 0.0)
        if last:
            ga.population.AddPopulation(store.population)
        ga.CalculateFitness()
        ga.population.TrimPopulation()
        while True:
            ga.Init()
            ga.CalculateFitness()
            if self.trim:
                ga.population.TrimPopulation()
            ga.SpreadControl(self.sc_steps, self.sc_rate, self.sc_factor)
            self._save(store, ga)
            if ga.HasConverged(self.steps, self.conv_crit):
                break
        self._save(store, ga)

    def _save(self, store: GeneticAlgorithm, ga: GeneticAlgorithm) -> None:
        ga.population.Sort()
        for genes in ga.population.genes[: min(self.save_generation, self.pop_size)]:
            store.population.GiveHint(genes.factors, genes.fitness)

    def Run(self, parameters: Any = None) -> float:
        """``Run(pars)``: the fit, its best parameters put in ``pars``; the best fitness."""
        self.log.header(
            "<GeneticFitter> Optimisation, please be patient ... (inaccurate progress timing for GA)"
        )
        if parameters is None:
            parameters = FactorVector(interval.GetMean() for interval in self.ranges)
        store = GeneticAlgorithm(self.target, self.pop_size, self.ranges)
        for cycle in range(self.cycles):
            self._cycle(store, list(parameters), cycle == self.cycles - 1)
        self.log.info("Elapsed time: 0 sec                            ")
        fitness = store.CalculateFitness()
        best = store.population.genes[0].factors
        _fill(parameters, best)
        return fitness


def _fill(target: Any, values: list[float]) -> None:
    """``pars.swap(best)``: the best parameters put in the caller's vector, whatever kind it is."""
    if hasattr(target, "clear") and hasattr(target, "push_back"):
        target.clear()
        for value in values:
            target.push_back(value)
    else:
        target[:] = values
