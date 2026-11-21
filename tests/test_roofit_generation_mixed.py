"""A density that draws one observable itself, the rest numerically: ``RooGenContext``'s split."""

from __future__ import annotations

from typing import Any

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.functions import RooFormulaVar, RooPolyVar
from xrdroot.roofit.pdfs.basic import RooGaussian
from xrdroot.roofit.pdfs.prodpdf import RooProdPdf
from xrdroot.roofit.rng import RooRandom
from xrdroot.roofit.variables import RooRealVar


def test_a_gaussian_in_x_about_a_function_of_y_draws_x_itself_and_y_by_foam() -> None:
    """rf301's model, seed 1234: ROOT 6.40's five events and its next draw, to the bit."""
    x, y = RooRealVar("x", "x", -5, 5), RooRealVar("y", "y", -5, 5)
    a0, a1 = RooRealVar("a0", "a0", -0.5, -5, 5), RooRealVar("a1", "a1", -0.5, -1, 1)
    fy = RooPolyVar("fy", "fy", y, [a0, a1])
    model = RooGaussian("model", "model", x, fy, RooRealVar("sigma", "sigma", 0.5))
    RooRandom.randomGenerator().SetSeed(1234)
    data = model.generate([x, y], 5)
    assert list(zip(data.column("x").tolist(), data.column("y").tolist(), strict=False)) == [
        (1.4402277289514984, -3.771667250257451),
        (-2.063946812965178, 4.334382233082776),
        (2.2233136274444405, -4.2763367088628),
        (0.3751258397996935, -0.4948165779114788),
        (2.4388942853547633, -4.842276310082525),
    ]
    assert RooRandom.randomGenerator().Rndm() == 0.2977002162951976


def test_observables_the_density_cannot_draw_itself_are_all_sampled_numerically() -> None:
    """A width has no generator, and ``y`` under a mean ``f(y)`` is not safe to draw directly."""
    from xrdroot.roofit.generation.contexts import MixedContext, NumericContext, context_for

    x, y = RooRealVar("x", "x", -5, 5), RooRealVar("y", "y", -5, 5)
    s = RooRealVar("s", "s", 1.0, 0.5, 2.0)
    fy = RooPolyVar("fy", "fy", y, [RooRealVar("a0", "a0", -0.5), RooRealVar("a1", "a1", 0.5)])
    model = RooGaussian("model", "model", x, fy, s)
    chosen = (context_for(model, frozenset(["s", "y"])), context_for(model, frozenset(["x", "y"])))
    assert (type(chosen[0]), type(chosen[1])) == (NumericContext, MixedContext)


def test_a_products_generation_announces_each_factors_integral_and_not_its_own(
    capsys: Any,
) -> None:
    """rf309's model in a product, as ROOT 6.40 prints it: three lines for the factor whose
    normalisation is numeric - one at each of the generator's two copies, and one for its own
    context - and none for the product, whose integral is its factors'."""
    x, y, z = (RooRealVar(n, n, -5, 5) for n in "xyz")
    a0, a1 = RooRealVar("a0", "a0", -3.5, -5, 5), RooRealVar("a1", "a1", -1.5, -1, 1)
    fy = RooFormulaVar("fy", "a0-a1*sqrt(10*abs(y))", RooArgSet(y, a0, a1))
    model = RooGaussian("model", "model", x, fy, RooRealVar("sigma", "sigma", 1.5))
    gz = RooGaussian("gz", "gz", z, RooRealVar("m", "m", 0.0), RooRealVar("w", "w", 2.0))
    together = RooProdPdf("model3", "model3", RooArgSet(model, gz))
    capsys.readouterr()
    RooRandom.randomGenerator().SetSeed(7)
    together.generate([x, y, z], 5)
    said = capsys.readouterr().out.splitlines()
    assert len(said) == 3
    assert {line.split("RooRealIntegral::init(")[1].split(")")[0] for line in said} == {
        "model_Int[x,y]"
    }
