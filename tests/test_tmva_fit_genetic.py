"""TMVA's fitters by themselves: the intervals, the genetic algorithm, and Monte Carlo sampling."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from tmvasupport import session
from xrdroot.random.mersenne import TRandom3
from xrdroot.tmva import TMVAError
from xrdroot.tmva.cutsfit import draw_mc
from xrdroot.tmva.genetic import Genes, GeneticPopulation, GeneticRange
from xrdroot.tmva.mcfitter import MCFitter

__all__ = ["session"]


class Parabola(ROOT.TMVA.IFitterTarget):
    """The squared distance from (1, -2): the least of it is 0, there."""

    def __init__(self) -> None:
        self.calls = 0

    def EstimatorFunction(self, parameters):
        self.calls += 1
        return (parameters[0] - 1.0) ** 2 + (parameters[1] + 2.0) ** 2


class Batched(Parabola):
    """The parabola, able to say its value of many parameter sets at once."""

    def batch_estimator(self, parameters):
        return [(p[0] - 1.0) ** 2 + (p[1] + 2.0) ** 2 for p in parameters]


def _ranges():
    return [ROOT.TMVA.Interval(-5, 5), ROOT.TMVA.Interval(-5, 5)]


def test_an_interval_knows_its_range_its_middle_and_its_discrete_values():
    interval = ROOT.TMVA.Interval(0, 4, 5)
    assert (interval.GetMin(), interval.GetMax(), interval.GetNbins()) == (0.0, 4.0, 5)
    assert interval.GetWidth() == 4.0 and interval.GetMean() == 2.0
    assert interval.GetElement(3) == 3.0 and interval.GetStepSize() == 1.0
    interval.SetMin(1)
    interval.SetMax(9)
    assert (interval.GetMin(), interval.GetMax()) == (1.0, 9.0)


def test_an_interval_upside_down_or_of_one_bin_is_refused():
    with pytest.raises(TMVAError, match="maximum lower than minimum"):
        ROOT.TMVA.Interval(1, 0)
    for nbins in (1, -2):
        with pytest.raises(TMVAError, match="at least 2 bins"):
            ROOT.TMVA.Interval(0, 1, nbins)


def test_an_element_of_a_continuous_interval_or_past_the_last_bin_is_refused():
    with pytest.raises(TMVAError, match="out of range"):
        ROOT.TMVA.Interval(0, 1).GetElement(0)
    with pytest.raises(TMVAError, match="out of range"):
        ROOT.TMVA.Interval(0, 1, 3).GetElement(3)


def test_a_fitter_target_that_does_not_say_its_estimator_is_refused():
    target = ROOT.TMVA.IFitterTarget()
    target.ProgressNotifier("fit", "step")
    with pytest.raises(NotImplementedError):
        target.EstimatorFunction([0.0])


def test_the_genetic_fitter_finds_the_least_of_a_parabola(session):
    target = Parabola()
    fitter = ROOT.TMVA.GeneticFitter(target, "GA", _ranges(), "PopSize=40:Steps=5:Cycles=2")
    best = ROOT.std.vector("double")()
    fitness = fitter.Run(best)
    assert fitness < 0.05 and abs(best[0] - 1.0) < 0.3 and abs(best[1] + 2.0) < 0.3
    assert target.calls > 0


def test_the_genetic_fitter_fills_a_list_and_starts_from_the_middle_when_given_none(session):
    fitter = ROOT.TMVA.GeneticFitter(Parabola(), "GA", _ranges(), "")
    fitter.SetParameters(1, 3, 20, 5, 2, 0.9, 0.01)
    assert fitter.Run() < 5
    found = [0.0, 0.0]
    fitter.Run(found)
    assert len(found) == 2


def test_the_genetic_fitter_trims_its_population_and_copies_its_best_when_asked(session):
    options = "PopSize=20:Steps=3:Cycles=1:Trim=True:SaveBestGen=3"
    assert ROOT.TMVA.GeneticFitter(Parabola(), "GA", _ranges(), options).Run([0.0]) < 5


def test_the_genetic_algorithm_evolves_a_population_step_by_step(session):
    ga = ROOT.TMVA.GeneticAlgorithm(Parabola(), 12, _ranges(), 3)
    ga.SetSpread(0.2)
    ga.SetMakeCopies(True)
    assert ga.GetSpread() == 0.2
    ga.Init()
    first = ga.CalculateFitness()
    for _ in range(4):
        ga.Init()
        ga.CalculateFitness()
    assert ga.CalculateFitness() <= first
    population = ga.GetGeneticPopulation()
    assert population.GetPopulationSize() > 12
    population.SetRandomSeed(5)
    genes = population.GetGenes(0)
    genes.SetFitness(2)
    assert genes.GetFitness() == 2.0 and len(genes.GetFactors()) == 2


def test_the_spread_narrows_after_many_successes_and_widens_after_few():
    ga = ROOT.TMVA.GeneticAlgorithm(Parabola(), 4, _ranges())
    for best in (5.0, 4.0, 3.0):
        ga.best = best
        ga.SpreadControl(3, 1, 0.5)
    assert ga.GetSpread() == pytest.approx(0.2)
    for _ in range(2):
        ga.SpreadControl(3, 2, 0.5)
    assert ga.GetSpread() == pytest.approx(0.1)
    ga.SpreadControl(3, 0, 0.5)
    assert ga.GetSpread() == pytest.approx(0.1)


def test_a_negative_number_of_steps_converges_at_once_and_a_jump_starts_counting_again():
    ga = ROOT.TMVA.GeneticAlgorithm(Parabola(), 4, _ranges())
    ga.best = 1.0
    assert ga.HasConverged(-1, 0.0)
    ga.best = 3.0
    assert not ga.HasConverged(2, 0.1) and ga.converge_counter == 0


def test_a_population_prints_as_many_genes_as_it_is_asked_for(session, capsys):
    population = GeneticPopulation(_ranges(), 3, 1)
    population.Print()
    assert "fitness" not in capsys.readouterr().out
    population.Print(1)
    assert capsys.readouterr().out.count("fitness") == 2
    population.Print(-2)
    assert capsys.readouterr().out.count("fitness") == 3
    assert isinstance(population.MakeSex(population.genes[0], population.genes[1]), Genes)


def test_a_range_draws_discrete_values_its_one_value_and_steps_wrapped_or_mirrored():
    random = TRandom3(1)
    assert GeneticRange(random, ROOT.TMVA.Interval(0, 2, 3)).random_value() in (0.0, 1.0, 2.0)
    assert GeneticRange(random, ROOT.TMVA.Interval(3, 3)).random_value(True, 3.0) == 3.0
    span = GeneticRange(random, ROOT.TMVA.Interval(0, 1))
    for mirror in (False, True):
        for _ in range(20):
            assert 0.0 <= span.random_value(True, 0.5, 2.0, mirror) < 1.0
    assert span._wrap(-0.25) == 0.75 and span._wrap(2.25) == pytest.approx(0.25)
    assert span._mirror(-0.25) == 0.25 and span._mirror(1.25) == 0.75
    assert span._mirror(0.5) == 0.5


def test_a_range_whose_interval_was_turned_upside_down_leaves_a_step_where_it_is():
    interval = ROOT.TMVA.Interval(0, 1)
    interval.SetMin(2)
    upside = GeneticRange(TRandom3(1), interval)
    assert upside._mirror(7.0) == 7.0 and upside._wrap(7.0) == 7.0


def test_the_monte_carlo_fitter_samples_one_by_one_in_batches_and_around_the_best(session):
    for target, options in (
        (Parabola(), "SampleSize=300"),
        (Batched(), "SampleSize=4500"),
        (Parabola(), "SampleSize=300:Sigma=0.1"),
    ):
        fitter = MCFitter(target, "MC", _ranges(), options)
        best = [0.0, 0.0]
        assert fitter.Run(best) < 1.5
        assert abs(best[0] - 1.0) < 1.5
    assert MCFitter(Parabola(), "MC", _ranges(), "SampleSize=10").Run() < 30


def test_the_monte_carlo_draws_discrete_values_and_fixed_ones():
    drawn = draw_mc([ROOT.TMVA.Interval(0, 2, 3), ROOT.TMVA.Interval(4, 4)], 50, 7)
    assert set(drawn[:, 0]) <= {0.0, 1.0, 2.0} and set(drawn[:, 1]) == {4.0}
    assert draw_mc([ROOT.TMVA.Interval(4, 4)], 3, 7).tolist() == [[4.0]] * 3
