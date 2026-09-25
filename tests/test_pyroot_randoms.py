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
        (len(r.Rndm(5)), 5),
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
