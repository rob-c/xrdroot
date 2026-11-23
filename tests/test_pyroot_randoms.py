"""``TRandom`` and its generators by ROOT's names: the same streams as ROOT's, draw for draw."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot import random as xrandom


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


@pytest.mark.parametrize("cls", ["TRandom", "TRandom1", "TRandom2", "TRandom3"])
def test_each_generator_is_the_xrdroot_one_of_its_name(cls):
    made = getattr(ROOT, cls)(17)
    beneath = getattr(xrandom, cls)(17)
    expect(
        (made.Rndm(), float(beneath.rndm())),
        (made.Gaus(1, 2), float(beneath.gaus(1, 2))),
        (made.ClassName(), cls),
        (bool(made.GetName().startswith("Random")), True),
    )


def test_a_uniform_drawn_one_at_a_time_is_the_one_the_column_holds():
    """``Uniform`` of one number is the column's arithmetic done on one draw, and the stream
    must not be able to tell the difference: the same seed gives the same numbers either way."""
    for parameters in ((1.0,), (10.0,), (2.5, 7.25), (-3, 4)):
        one_at_a_time, column = ROOT.TRandom3(4242), ROOT.TRandom3(4242)
        drawn = [one_at_a_time.Uniform(*parameters) for _ in range(64)]
        together = column.Uniform(parameters[0], parameters[1] if len(parameters) > 1 else None, 64)
        expect(*zip([float(value) for value in together], drawn, strict=True))


def test_every_distribution_root_offers():
    r = ROOT.TRandom3(4357)
    expect(
        (bool(0 < r.Rndm() <= 1), True),
        (bool(0 < r.Uniform(5) < 5), True),
        (bool(2 < r.Uniform(2, 3) < 3), True),
        (bool(isinstance(r.Gaus(), float)), True),
        (bool(r.Exp(2.0) > 0), True),
        (bool(0 <= r.Integer(10) < 10), True),
        (bool(isinstance(r.BreitWigner(), float)), True),
        (bool(isinstance(r.Landau(), float)), True),
        (bool(0 <= r.Binomial(10, 0.5) <= 10), True),
        (bool(r.Poisson(3.0) >= 0), True),
        (bool(r.PoissonD(3.0) >= 0), True),
        (bool(isinstance(r.Integer(10), int)), True),
        (bool(isinstance(r.Poisson(2.0), int)), True),
        (bool(isinstance(r.Rndm(5), float)), True),
        (len(r.Gaus(0, 1, 4)), 4),
        (len(r.Uniform(0, 1, 3)), 3),
        (len(r.Exp(1.0, 3)), 3),
        (len(r.Integer(5, 3)), 3),
        (len(r.BreitWigner(0, 1, 2)), 2),
        (len(r.Landau(0, 1, 2)), 2),
        (len(r.Binomial(5, 0.5, 2)), 2),
        (len(r.Poisson(2.0, 2)), 2),
        (len(r.PoissonD(2.0, 2)), 2),
    )


def test_draws_by_reference_fill_what_they_are_given():
    r = ROOT.TRandom3(1)
    a, b = ctypes.c_double(0), ctypes.c_double(0)
    x, y = r.Rannor(a, b)
    expect(
        ((a.value, b.value), (x, y)),
        (bool(r.Rannor() != (0, 0)), True),
    )
    px, py = r.Circle(a, b, 2.0)
    expect(
        (np.hypot(px, py), pytest.approx(2.0)),
        (a.value, px),
    )
    z = ctypes.c_double(0)
    sx, sy, sz = r.Sphere(a, b, z, 3.0)
    expect(
        (np.sqrt(sx * sx + sy * sy + sz * sz), pytest.approx(3.0)),
        (z.value, sz),
    )
    held = np.zeros(4)
    r.RndmArray(4, held)
    assert np.all((held > 0) & (held <= 1))


def test_seeds_restart_the_stream(capsys):
    r = ROOT.TRandom3()
    r.SetSeed(99)
    first = r.Rndm()
    r.SetSeed(99)
    expect(
        (r.Rndm(), first),
        (bool(r.GetSeed() > 0), True),
    )
    r.SetSeed()
    r.Print()
    expect(
        (
            bool(
                capsys.readouterr().out.startswith("Random number generator: TRandom3 with seed ")
            ),
            True,
        ),
        (bool(ROOT.TRandom1(1, 4).Rndm() > 0), True),
        (bool(ROOT.TRandom2().Rndm() > 0), True),
        (bool(ROOT.TRandom().Rndm() > 0), True),
    )


def test_grandom_is_xrdroots_and_a_replacement_is_followed():
    from xrdroot.pyroot.core import randoms

    expect(
        (bool(ROOT.gRandom._xrd is xrandom.gRandom), True),
        (bool(randoms.current_generator() is xrandom.gRandom), True),
    )
    replacement = ROOT.TRandom2(5)
    ROOT.gRandom = replacement
    try:
        assert randoms.current_generator() is replacement._xrd
    finally:
        ROOT.gRandom = randoms.gRandom


def test_a_translated_float_filled_by_reference_is_rounded_as_cpp_rounds_it():
    from xrdroot.pycint.runtime import Cell

    r = ROOT.TRandom3(1)
    px, py, wide = Cell(0.0, "float"), Cell(0.0, "float"), Cell(0.0, "double")
    x, y = r.Rannor(px, py)
    r.SetSeed(1)
    r.Rannor(wide, None)
    # Rannor(Float_t&, Float_t&) draws in double and rounds as it stores, as ROOT's does.
    assert (px.value, py.value) == (float(np.float32(x)), float(np.float32(y)))
    assert px.value != x and wide.value == x


def test_mixmax_is_root_interface_over_a_stream_that_is_not_mixmaxs():
    """ROOT's ``TRandomMixMax`` is ``TRandomGen<MixMaxEngine<240,0>>``, and its numbers come
    from seeding that ROOT does with a table of skip coefficients this does not carry, so the
    class says in its docstring that the stream is a Mersenne twister's instead. This pins the
    substitution: the same seed gives ``TRandom3``'s stream, under ROOT's name and title."""
    made, beneath = ROOT.TRandomMixMax(11), ROOT.TRandom3(11)
    expect(
        (made.ClassName(), "TRandomMixMax"),
        (str(made.GetName()), "Random_MixMax240"),
        (str(made.GetTitle()), "Random number generator: MixMax240"),
        (made.Rndm(), beneath.Rndm()),
        (made.Gaus(-3, 1.5), beneath.Gaus(-3, 1.5)),
        ("not MIXMAX's" in type(made).__doc__, True),
    )


def test_mixmax_starts_at_root_seed_one_and_restarts_with_the_same_numbers():
    r = ROOT.TRandomMixMax()
    first = (r.Rndm(), r.Gaus(2, 2.5))
    r.SetSeed(1)
    expect(
        ((r.Rndm(), r.Gaus(2, 2.5)), first),
        (ROOT.TRandomMixMax().Rndm(), ROOT.TRandomMixMax().Rndm()),
        (bool(isinstance(r.Poisson(3.0), int)), True),
        (len(r.Gaus(0, 1, 4)), 4),
        (bool(r.GetSeed() > 0), True),
    )


def test_mixmax_is_a_generator_a_histogram_can_be_filled_from():
    """What it is for: a deterministic stream to fill a histogram with, exactly as ROOT's own
    is used - the same seed fills the same bins the same way twice."""
    def filled():
        r = ROOT.TRandomMixMax()
        h = ROOT.TH1D("h", "h", 20, -5, 5)
        for _ in range(1000):
            h.Fill(r.Gaus(0, 2))
        return [h.GetBinContent(at) for at in range(0, 22)]

    expect((filled(), filled()), (sum(filled()), 1000.0))
def test_sobols_first_coordinate_is_the_van_der_corput_sequence_and_the_second_its_own():
    from xrdroot.pyroot.core.randoms import QuasiRandomSobol

    sequence = QuasiRandomSobol(2)
    assert [sequence.next_point() for _ in range(4)] == [
        [0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.375, 0.375],
    ]  # fmt: skip
    assert (sequence.Size(), sequence.Dimension()) == (2, 2)


def test_the_niederreiter_order_gives_every_coordinate_a_polynomial_of_its_own():
    from xrdroot.pyroot.core.randoms import QuasiRandomNiederreiter

    sequence = QuasiRandomNiederreiter(3)
    assert [sequence.next_point()[0] for _ in range(4)] == [0.5, 0.25, 0.75, 0.375]
    assert all(0.0 < value < 1.0 for value in sequence.next_point())


def test_a_point_and_a_column_of_points_go_into_the_array_the_call_gave():
    from xrdroot.pyroot.core.randoms import QuasiRandomNiederreiter, QuasiRandomSobol

    point = np.zeros(2)
    assert QuasiRandomSobol(2).Next(point) == [0.5, 0.5]
    assert point.tolist() == [0.5, 0.5]
    column = np.zeros(4)
    QuasiRandomNiederreiter(2).RndmArray(2, column)
    assert column.tolist() == [0.5, 0.5, 0.25, 0.25]


def test_discard_steps_over_points_with_and_without_a_stride():
    from xrdroot.pyroot.core.randoms import QuasiRandomSobol

    plain, strided = QuasiRandomSobol(2), QuasiRandomSobol(2)
    plain.Discard(1)
    strided.Discard(1, 2)  # one point, then two skipped
    assert plain.next_point() == [0.75, 0.25]
    assert strided.next_point() == [0.375, 0.375]


def test_the_directions_of_later_coordinates_come_from_the_primitive_polynomials():
    from xrdroot.pyroot.core.randoms import QuasiRandomSobol

    sequence = QuasiRandomSobol(6)
    points = [sequence.next_point() for _ in range(8)]
    assert len({tuple(point) for point in points}) == 8  # eight distinct points of six axes
    assert all(0.0 < value < 1.0 for point in points for value in point)


def test_random_mt_is_the_twister_under_the_random_front():
    assert ROOT.RandomMT(7).Rndm() == ROOT.TRandom3(7).Rndm()
    assert ROOT.RandomMT().Rndm() == ROOT.TRandom3().Rndm()
