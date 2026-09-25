"""FOAM as RooFit drives it, held to the events ROOT 6.40.04 generated.

Every reference number below was printed by ROOT itself - ``pdf.generate``
through PyROOT for the RooFit cases, a bare ``TFoam`` with a Python
``TFoamIntegrand`` for the settings RooFit does not use - with ``repr`` so
that nothing is lost. The densities are RooFit's normalised pdfs written
out as RooFit computes them, with the normalisation integrals ROOT reported.
The number ``Rndm()`` gave next is checked too: it says the foam took
exactly ROOT's draws, no more and no fewer.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot.random import TRandom3
from xrdroot.roofit.generation.foam import Foam, FoamGenerator

#: ``RooExponential`` with ``c = -0.3`` on [0, 10]: the first 20 of 1000 events.
EXPONENTIAL = [
    3.770775238024271, 5.248203310038662, 3.6526121302074444, 5.328677688376047,
    1.6222330968957976, 2.7414108423698735, 0.10886648218729533, 1.2993865995667875,
    1.0105262185243191, 0.6425811393273761, 1.3289180455103633, 0.1632645685094758,
    1.591365611711808, 0.01065373005985748, 1.1939488467760384, 1.222582135378616,
    7.510397503647255, 0.4135690462135244, 0.6876594693676452, 2.1187819291299093,
]
#: ``RooGenericPdf("exp(-x*y/10)+0.1")`` on [0, 10] squared: the first 12 of 200 events.
FORMULA = [
    (8.07856377712838, 1.5923498395407876), (9.244015112939223, 4.153850963813852),
    (3.122924342578699, 2.038487561629836), (0.14569587177163612, 7.74672273248143),
    (8.112711350823147, 6.383890587610317), (0.6323510611644423, 7.691739499850883),
    (4.641862034416704, 0.8263430691977192), (0.022184310248007932, 9.437725057941861),
    (7.932404183633963, 5.377201835667922), (2.2445288450853695, 1.0348655716734356),
    (8.102348295423099, 9.482005193111078), (2.105521759419844, 1.4845479958734131),
]
#: The formula's normalisation, from ``RooAdaptiveIntegratorND`` as ROOT reported it.
FORMULA_NORM = 38.7980491683548
#: ``RooChebychev`` with coefficients 0.5 and -0.2 on [-1, 3], after the formula's events.
CHEBYCHEV = [
    0.5659389379416098, 1.9433359196118545, 1.7467068417172413, -0.5795744279485007,
    2.0136640649288893, 2.4376259830314666, 1.852776789048221, 2.65706096123904,
    2.323842979269102, 1.061851021473558, 0.031560017760057235, 2.0015481961891055,
    1.597157697447983, -0.2394195434753783, 1.652595848951023,
]
#: ``RooGenericPdf("1+x*y*z")`` on the unit cube: the first 5 of 20 events.
CUBE = [
    (0.2844876430614725, 0.7821626563854807, 0.8604111976419517),
    (0.48366992827223143, 0.2847281590948114, 0.6789708158354415),
    (0.7093903726872668, 0.7245473587991, 0.5743889950190342),
    (0.8565377028646708, 0.683910916937748, 0.10778151526821489),
    (0.07777216925751418, 0.44769098194956314, 0.2800003666533988),
]


def exponential(point: np.ndarray) -> float:
    """``RooExponential``'s value over its analytic integral, as RooFit divides them."""
    c = -0.3
    norm = (math.exp(c * 10.0) - math.exp(c * 0.0)) / c
    return math.exp(c * float(point[0])) / norm


def formula(point: np.ndarray) -> float:
    x, y = float(point[0]), float(point[1])
    return (math.exp(-x * y / 10) + 0.1) / FORMULA_NORM


def chebychev(point: np.ndarray) -> float:
    """``MathFuncs::chebychev`` for two coefficients, over ``chebychevIntegral``'s 4.2666..."""
    x_prime = (float(point[0]) - 0.5 * (3.0 + -1.0)) / (0.5 * (3.0 - -1.0))
    total, last, curr = 1.0, x_prime, 2 * x_prime * x_prime - 1.0
    for coeff in (0.5, -0.2):
        total += last * coeff
        last, curr = curr, 2 * x_prime * curr - last
    return total / 4.266666666666667


def test_an_exponential_is_generated_as_root_generates_it() -> None:
    rng = TRandom3()
    events = FoamGenerator(exponential, [(0.0, 10.0)], rng).generate_many(1000)
    assert events.shape == (1000, 1)
    assert events[:20, 0].tolist() == EXPONENTIAL
    assert events[999, 0] == 0.76627350270428
    assert rng.rndm() == 0.5695793216582388


def test_a_formula_in_two_dimensions_then_a_chebychev_are_generated_as_root_does() -> None:
    rng = TRandom3()
    generator = FoamGenerator(formula, [(0.0, 10.0), (0.0, 10.0)], rng)
    events = generator.generate_many(200)
    assert [tuple(row) for row in events[:12].tolist()] == FORMULA
    assert events[199].tolist() == [3.7192750909889583, 0.06744198234628129]
    # ROOT's check of the next number took it, so the Chebychev starts one later.
    assert rng.rndm() == 0.09599034604616463
    events = FoamGenerator(chebychev, [(-1.0, 3.0)], rng).generate_many(50)
    assert events[:15, 0].tolist() == CHEBYCHEV
    assert rng.rndm() == 0.2638880806043744


def test_a_three_dimensional_formula_is_generated_as_root_generates_it_a_batch_at_a_time() -> None:
    rng = TRandom3()
    cube = FoamGenerator(
        lambda p: (1 + p[:, 0] * p[:, 1] * p[:, 2]) / 1.125, [(0, 1)] * 3, rng, vectorized=True
    )
    assert len(cube.foam.cells) == 4999
    assert [tuple(cube.generate().tolist()) for _ in range(5)] == CUBE
    cube.generate_many(15)
    assert rng.rndm() == 0.9934110806789249


#: A bare ``TFoam`` - 2 dimensions, 60 cells, 300 samples, ``TRandom3(12345)`` -
#: by ``SetOptDrive`` and ``SetOptRej``: five events and weights, the next
#: ``Rndm()``, the primary integral and the number of calls to the density.
BARE = {
    (1, 1): (
        [(0.18304699076043107, 0.5649964305425783, 1.045630962307752),
         (0.32022247103896007, 0.6361230811108811, 1.0),
         (0.20473922152791602, 0.6762232696391948, 1.0465136673866862),
         (0.4617397573415474, 0.6515508576248976, 1.1106824695086053),
         (0.699047805275768, 0.1600421522416582, 1.0)],
        0.5596259464509785, 0.11459820387828243, 13421,
    ),
    (2, 0): (
        [(0.4850780840497464, 0.9145708775904495, 0.5619368533384348),
         (0.3108515210697078, 0.6774961177143268, 0.7489017523498558),
         (0.9849389144219458, 0.3629253446124494, 0.9240698178471196),
         (0.3214119274643963, 0.6191385603160597, 0.9616293163716974),
         (0.1584436382545391, 0.3823588937812019, 0.2332031068021101)],
        0.14975547953508794, 0.16874638668484324, 13610,
    ),
    (1, 0): (
        [(0.18304699076043107, 0.5649964305425783, 1.1501940585385273),
         (0.23971515443588487, 0.6123053212357945, 1.1335902975716095),
         (0.2647007055426229, 0.6120147648581167, 1.1034550359324182),
         (0.8159022432519123, 0.3243503368466918, 0.9832559459175958),
         (0.4617397573415474, 0.6515508576248976, 1.2217507164594659)],
        0.3878474640659988, 0.11459820387828243, 13421,
    ),
}


def peak(points: np.ndarray) -> np.ndarray:
    """The ``TFoamIntegrand`` ROOT was given: a narrow peak on a flat floor."""
    return np.array([
        math.exp(-((float(a) - 0.3) ** 2 + (float(b) - 0.6) ** 2) / 0.02) + 0.05
        for a, b in points
    ])


@pytest.mark.parametrize(("drive", "rej"), sorted(BARE))
def test_a_bare_foam_builds_and_draws_as_root_does_for_each_driver_and_rejection(
    drive: int, rej: int,
) -> None:
    events, after, prime, calls = BARE[(drive, rej)]
    rng = TRandom3(12345)
    foam = Foam(peak, 2, rng, n_cells=60, n_sampl=300, opt_drive=drive, opt_rej=rej)
    foam.initialize()
    made = []
    for _ in range(5):
        point = foam.make_event()
        made.append((float(point[0]), float(point[1]), foam.mc_wt))
    assert made == events
    assert rng.rndm() == after
    assert foam.prime == prime
    assert foam.n_calls == calls
    assert foam.nev_gen >= 5
    assert foam.wt_min <= foam.wt_max


def test_negative_values_count_as_zero_as_the_binding_makes_them() -> None:
    generator = FoamGenerator(lambda p: float(p[0]) - 5.0, [(0.0, 10.0)], TRandom3(7))
    events = generator.generate_many(200)
    assert events.min() > 5.0


def test_without_a_generator_it_starts_a_fresh_trandom3_as_roofits_first_use_does() -> None:
    fresh = FoamGenerator(exponential, [(0.0, 10.0)]).generate_many(3)
    assert fresh[:, 0].tolist() == EXPONENTIAL[:3]


def test_a_foam_of_one_cell_draws_uniformly_under_the_density() -> None:
    generator = FoamGenerator(exponential, [(0.0, 10.0)], TRandom3(3), n_cells=1)
    assert len(generator.foam.cells) == 1
    events = generator.generate_many(2000)
    assert 0.0 < events.min() and events.max() < 10.0
    assert abs(events.mean() - 3.0) < 0.3


def test_the_variance_driver_is_not_misled_by_infinite_weights() -> None:
    foam = Foam(lambda p: np.full(len(p), np.inf), 2, TRandom3(), n_cells=1, opt_drive=1)
    foam.initialize()
    root = foam.cells[0]
    assert (root.best, root.xdiv) == (1, 0.0)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"dim": 0}, "at least one dimension"),
        ({"n_sampl": 0}, "at least one exploration point"),
        ({"n_bin": 0}, "at least one exploration point"),
        ({"opt_drive": 3}, "FOAM's driver is 1"),
    ],
)
def test_settings_foam_cannot_build_with_are_refused(options: dict[str, int], message: str) -> None:
    settings = {"dim": 1, **options}
    dim = settings.pop("dim")
    with pytest.raises(ValueError, match=message):
        Foam(peak, dim, TRandom3(), **settings)


@pytest.mark.parametrize(
    ("value", "cells", "message"),
    [
        (math.nan, 10, "driver integral to choose it by"),
        (-1.0, 10, "no edge to cut along"),
        (0.0, 10, "zero everywhere"),
        (0.0, 1, "zero everywhere"),
    ],
)
def test_densities_foam_cannot_draw_from_are_refused(
    value: float, cells: int, message: str,
) -> None:
    foam = Foam(lambda p: np.full(len(p), value), 2, TRandom3(), n_cells=cells)
    with pytest.raises(ValueError, match=message):
        foam.initialize()
