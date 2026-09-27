"""RooFit's sums, products and simultaneous densities, held to what ROOT 6.40 printed.

Values and integrals are ROOT's ``%.12g``, events ``repr`` after
``RooRandom::randomGenerator()->SetSeed(4357)``, and printed lines ROOT's
own, for the same models built here with the engine's classes.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.pdf import CAN_NOT_BE_EXTENDED, MUST_BE_EXTENDED
from xrdroot.roofit.pdfs.addpdf import RooAddPdf, RooRecursiveFraction
from xrdroot.roofit.pdfs.basic import RooExponential, RooGaussian
from xrdroot.roofit.pdfs.extend import RooExtendPdf
from xrdroot.roofit.rng import generator
from xrdroot.roofit.variables import RooRealVar

REL = 1e-11


def column(data: Any, name: str) -> list[float]:
    return [data.get(i).getRealValue(name) for i in range(data.numEntries())]


def integral(pdf: Any, over: list[Any], rng: str | None = None) -> float:
    made = pdf.createIntegral(over, Range=rng) if rng else pdf.createIntegral(over)
    return float(made.getVal())


class Sum:
    """Two Gaussians and an exponential in ``x``, and the fractions and yields that add them."""

    def __init__(self, width: float = 2.0) -> None:
        self.x = RooRealVar("x", "x", 0.5, -10, 10)
        self.x.setRange("win", -1.5, 2.5)
        self.m1 = RooRealVar("m1", "m1", 1, -5, 5)
        self.s1 = RooRealVar("s1", "s1", width, 0.1, 10)
        self.g1 = RooGaussian("g1", "g1", self.x, self.m1, self.s1)
        self.m2 = RooRealVar("m2", "m2", -2, -5, 5)
        self.s2 = RooRealVar("s2", "s2", 3, 0.1, 10)
        self.g2 = RooGaussian("g2", "g2", self.x, self.m2, self.s2)
        self.c = RooRealVar("c", "c", -0.2, -2, -0.01)
        self.e = RooExponential("e", "e", self.x, self.c)
        self.f = RooRealVar("f", "f", 0.3, 0, 1)
        self.f2 = RooRealVar("f2", "f2", 0.4, 0, 1)
        self.add = RooAddPdf("add", "add", [self.g1, self.g2], [self.f])

    def fixed(self) -> None:
        for one in (self.m1, self.s1, self.m2, self.s2):
            one.setConstant(True)


def test_a_sum_of_fractions_is_normalised_component_by_component() -> None:
    """``f g1 + (1-f) g2``, each normalised on its own, needs no normalisation of its own."""
    s = Sum()
    assert s.add.getVal() == pytest.approx(0.785423764843, rel=REL)
    assert s.add.getVal([s.x]) == pytest.approx(0.124034881813, rel=REL)
    assert integral(s.add, [s.x], "win") == pytest.approx(2.93614542948, rel=REL)
    three = RooAddPdf("add3", "add3", s.g1, s.g2, s.f)
    assert three.getVal([s.x]) == pytest.approx(0.124034881813, rel=REL)
    assert s.add.extendMode() == CAN_NOT_BE_EXTENDED
    assert list(s.add.pdfList()) == [s.g1, s.g2]
    assert list(s.add.coefList()) == [s.f]
    assert s.add.servers() == [s.g1, s.f, s.g2]
    assert s.add.state_word() == "Clean"
    assert s.add.normalized_name([s.x]) == "add"


def test_a_sum_prints_its_terms_and_the_rest_as_root_does(capsys: Any) -> None:
    """``[%]`` stands for the fraction that is the rest; yields print as their names."""
    s = Sum()
    capsys.readouterr()
    s.add.Print()
    yields = RooAddPdf(
        "ext", "ext", [s.g1, s.g2], [RooRealVar("n1", "n1", 300), RooRealVar("n2", "n2", 700)]
    )
    yields.Print()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("RooAddPdf::add[ f * g1 + [%] * g2 ] = ")
    assert lines[1].startswith("RooAddPdf::ext[ n1 * g1 + n2 * g2 ] = ")
    assert lines[0].endswith("/1")


@pytest.mark.xfail(
    strict=True,
    reason="addpdf.py:251-252: ROOT prints the sum's value in its last normalisation set",
)
def test_a_sum_prints_its_value_normalised_as_last_asked(capsys: Any) -> None:
    """After ``getVal([x])`` ROOT's ``Print`` shows the normalised value, ``0.124035/1``."""
    s = Sum()
    s.add.getVal([s.x])
    capsys.readouterr()
    s.add.Print()
    assert capsys.readouterr().out == "RooAddPdf::add[ f * g1 + [%] * g2 ] = 0.124035/1\n"


def test_a_recursive_sum_is_made_of_recursive_fractions_named_as_root_names_them() -> None:
    """``f1 A + (1-f1)(f2 B + (1-f2) C)``: each coefficient a ``RooRecursiveFraction``."""
    s = Sum()
    rec = RooAddPdf("rec", "rec", [s.g1, s.g2, s.e], [s.f, s.f2], True)
    assert rec.getVal([s.x]) == pytest.approx(0.0948924580633, rel=REL)
    assert [one.GetName() for one in rec.coefList()] == [
        "f",
        "rec_recursive_fraction_g2_2",
        "rec_recursive_fraction_e_3",
    ]
    assert rec.extendMode() == CAN_NOT_BE_EXTENDED
    last = rec.coefList()[2]
    assert isinstance(last, RooRecursiveFraction)
    assert last.getVal() == pytest.approx(0.7 * 0.6, rel=REL)
    assert last.state_word() == "Clean"
    assert rec.printMetaArgs() == (
        "f * g1 + rec_recursive_fraction_g2_2 * g2 + rec_recursive_fraction_e_3 * e "
    )


def test_a_sum_of_yields_is_extended_and_expects_their_total() -> None:
    """As many yields as components: the fractions are the yields over their total."""
    s = Sum()
    n1, n2 = RooRealVar("n1", "n1", 300, 0, 1000), RooRealVar("n2", "n2", 700, 0, 1000)
    ext = RooAddPdf("ext", "ext", [s.g1, s.g2], [n1, n2])
    assert ext.getVal([s.x]) == pytest.approx(0.124034881813, rel=REL)
    assert ext.expectedEvents([s.x]) == 1000.0
    assert ext.extendMode() == MUST_BE_EXTENDED
    assert ext.expected(frozenset(["x"]), "win") == pytest.approx(
        300 * integral(s.g1, [s.x], "win") / integral(s.g1, [s.x])
        + 700 * integral(s.g2, [s.x], "win") / integral(s.g2, [s.x]),
        rel=REL,
    )


def test_a_sum_of_extended_densities_takes_their_yields_for_coefficients() -> None:
    """No coefficients at all: each extended component brings its own yield."""
    s = Sum()
    e1 = RooExtendPdf("e1", "e1", s.g1, RooRealVar("ne1", "ne1", 100))
    e2 = RooExtendPdf("e2", "e2", s.g2, RooRealVar("ne2", "ne2", 50))
    both = RooAddPdf("allext", "allext", [e1, e2])
    assert both.getVal([s.x]) == pytest.approx(0.16033479651, rel=REL)
    assert both.expectedEvents([s.x]) == 150.0
    assert both.extendMode() == MUST_BE_EXTENDED


@pytest.mark.xfail(
    strict=True,
    reason="addpdf.py:210: a RooArgList is always true, so a sum without coefficients prints "
    "'[%] * e1'",
)
def test_a_sum_of_extended_densities_prints_them_plainly(capsys: Any) -> None:
    """ROOT prints ``e1 + e2`` for a sum whose components bring their own yields."""
    s = Sum()
    e1 = RooExtendPdf("e1", "e1", s.g1, RooRealVar("ne1", "ne1", 100))
    e2 = RooExtendPdf("e2", "e2", s.g2, RooRealVar("ne2", "ne2", 50))
    assert RooAddPdf("allext", "allext", [e1, e2]).printMetaArgs() == "e1 + e2 "


def test_a_sum_given_inconsistent_components_is_refused_with_roots_message() -> None:
    """Too many or too few coefficients, or a density with no yield of its own, are refused."""
    s = Sum()
    inconsistent = "number of pdfs and coefficients inconsistent, must have Npdf=Ncoef or Npdf"
    with pytest.raises(ValueError, match=inconsistent):
        RooAddPdf("bad", "bad", [s.g1], [s.f, s.f])
    with pytest.raises(ValueError, match=inconsistent):
        RooAddPdf("bad", "bad", [s.g1, s.g1, s.g1], [s.f])
    with pytest.raises(ValueError, match="Recursive fractions option can only be used if Npdf"):
        RooAddPdf("bad", "bad", [s.g1, s.g1], [s.f, s.f], True)
    with pytest.raises(ValueError, match="pdf g1 is not extendable, RooAddPdf constructor call"):
        RooAddPdf("bad", "bad", [s.g1, s.g1])


def test_a_sum_whose_fractions_exceed_one_warns_as_root_does(capsys: Any) -> None:
    """The rest is negative: RooFit says so, with the sum of the coefficients."""
    s = Sum()
    big = RooAddPdf("big", "big", [s.g1, s.g2], [RooRealVar("fbig", "fbig", 1.2, 0, 2)])
    big.getVal([s.x])
    assert (
        "[#0] WARNING:Eval -- RooAddPdf::updateCoefCache(big) WARNING: sum of PDF coefficients "
        "not in range [0-1], value=1.2" in capsys.readouterr().out
    )


@pytest.mark.xfail(
    strict=True,
    reason="addpdf.py:123-132: ROOT 6.40 evaluates a sum whose coefficients leave [0, 1] as NaN",
)
def test_a_sum_whose_fractions_exceed_one_is_not_a_number() -> None:
    """ROOT's value of ``1.2 g1 - 0.2 g2`` normalised is NaN, so a fit backs out of the region."""
    s = Sum()
    big = RooAddPdf("big", "big", [s.g1, s.g2], [RooRealVar("fbig", "fbig", 1.2, 0, 2)])
    assert big.getVal([s.x]) != big.getVal([s.x])


def test_a_sum_draws_each_event_from_a_component_chosen_by_its_fraction_as_root_does() -> None:
    """The events are ROOT's to the last bit, fractions and yields alike."""
    s = Sum()
    generator().SetSeed(4357)
    assert column(s.add.generate([s.x], 5), "x") == [
        -3.3042931682430208,
        -2.0901583139784634,
        -2.170151976437095,
        -2.2241134009544634,
        -3.2322894997417246,
    ]
    n1, n2 = RooRealVar("n1", "n1", 300, 0, 1000), RooRealVar("n2", "n2", 700, 0, 1000)
    ext = RooAddPdf("ext", "ext", [s.g1, s.g2], [n1, n2])
    generator().SetSeed(4357)
    assert column(ext.generate([s.x], 4), "x") == [
        -3.3042931682430208,
        -2.0901583139784634,
        -2.170151976437095,
        -2.2241134009544634,
    ]


def test_a_sum_fitted_in_a_range_normalises_its_fractions_over_the_full_range() -> None:
    """In a ranged fit the fractions are those of the full range, then normalised in the range."""
    s = Sum(width=1.0)
    generator().SetSeed(4357)
    data = s.add.generate([s.x], 400)
    s.fixed()
    s.f.setVal(0.5)
    result = s.add.fitTo(data, Range="win", PrintLevel=-1, Save=True)
    assert s.f.getVal() == pytest.approx(0.397127709726, rel=1e-7)
    assert result.minNll() == pytest.approx(251.206932919, rel=1e-10)


def test_a_sum_of_yields_fitted_in_a_range_expects_the_events_inside_it() -> None:
    """Extended in a range: each yield counts the events of its component inside."""
    s = Sum(width=1.0)
    generator().SetSeed(4357)
    data = s.add.generate([s.x], 400)
    s.fixed()
    n1, n2 = RooRealVar("n1", "n1", 100, 0, 1000), RooRealVar("n2", "n2", 300, 0, 1000)
    ext = RooAddPdf("ext", "ext", [s.g1, s.g2], [n1, n2])
    result = ext.fitTo(data, Range="win", PrintLevel=-1, Save=True)
    assert (n1.getVal(), n2.getVal()) == pytest.approx((127.039542598, 193.323730106), rel=1e-6)
    assert result.minNll() == pytest.approx(-550.483230436, rel=1e-10)


def test_a_sum_describes_itself_in_a_fit_as_its_normalised_terms_as_root_does() -> None:
    """What a fit's error log prints: the terms as the normalised densities, and their values."""
    s = Sum(width=1.0)
    s.f.setRange(0, 3)
    s.f.setVal(2.5)
    nset = frozenset(["x"])
    assert s.add.compiled_origin(nset) == (
        "RooAddPdf::add[ f * g1_over_g1_Int[x] + [%] * g2_over_g2_Int[x] ]"
    )
    assert s.add.compiled_servers(nset) == (
        "!refCoefNorm=(x = 0.5), !pdfs=(g1_over_g1_Int[x] = 0.352065,"
        "g2_over_g2_Int[x] = 0.094335), !coefficients=(f = 2.5)"
    )
    inner = RooAddPdf("inner", "inner", [s.g1, s.g2], [RooRealVar("fi", "fi", 0.4)])
    outer = RooAddPdf("add", "add", [inner, s.g2], [s.f])
    assert outer.compiled_origin(nset) == "RooAddPdf::add[ f * inner + [%] * g2_over_g2_Int[x] ]"


@pytest.mark.xfail(
    strict=True,
    reason="addpdf.py:239: a self-normalised term prints its value as ROOT's printValue, '/1'",
)
def test_a_sum_of_a_sum_describes_the_inner_sum_by_its_printed_value() -> None:
    """ROOT prints the inner sum as ``inner = 0.197427/1`` in a fit's error log."""
    s = Sum(width=1.0)
    s.f.setRange(0, 3)
    s.f.setVal(2.5)
    inner = RooAddPdf("inner", "inner", [s.g1, s.g2], [RooRealVar("fi", "fi", 0.4)])
    outer = RooAddPdf("add", "add", [inner, s.g2], [s.f])
    assert "!pdfs=(inner = 0.197427/1," in outer.compiled_servers(frozenset(["x"]))


def test_a_sum_of_extended_densities_fitted_in_a_range_expects_their_yields_there() -> None:
    """Without coefficients, what the sum expects in a range is its components' in it."""
    s = Sum()
    e1 = RooExtendPdf("e1", "e1", s.g1, RooRealVar("ne1", "ne1", 100))
    e2 = RooExtendPdf("e2", "e2", s.g2, RooRealVar("ne2", "ne2", 50))
    both = RooAddPdf("allext", "allext", [e1, e2])
    names = frozenset(["x"])
    inside = 100 * integral(s.g1, [s.x], "win") / integral(s.g1, [s.x])
    inside += 50 * integral(s.g2, [s.x], "win") / integral(s.g2, [s.x])
    assert both.expected(names, "win") == pytest.approx(inside, rel=REL)
