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


class Product:
    """``gx(x) gy(y)``: a product of densities of separate observables."""

    def __init__(self) -> None:
        from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

        self.x = RooRealVar("x", "x", 0.5, -10, 10)
        self.x.setRange("win", -1.5, 2.5)
        self.y = RooRealVar("y", "y", 1.5, -5, 5)
        self.y.setRange("win", -1, 3)
        sx, sy = RooRealVar("sx", "sx", 2, 0.1, 10), RooRealVar("sy", "sy", 1.5, 0.1, 10)
        self.gx = RooGaussian("gx", "gx", self.x, RooRealVar("mx", "mx", 1), sx)
        self.gy = RooGaussian("gy", "gy", self.y, RooRealVar("my", "my", 0.5), sy)
        self.prod = RooProdPdf("prod", "prod", [self.gx, self.gy])


def test_a_product_of_separate_observables_is_normalised_factor_by_factor() -> None:
    """Each factor normalised over its own observables: the product is normalised as it is."""
    p = Product()
    assert p.prod.getVal() == pytest.approx(0.776101302995, rel=REL)
    assert p.prod.getVal([p.x, p.y]) == pytest.approx(0.0412343220226, rel=REL)
    assert integral(p.prod, [p.x, p.y], "win") == pytest.approx(9.98789751997, rel=REL)
    assert integral(p.prod, [p.x]) == pytest.approx(4.0142883139, rel=REL)
    assert list(p.prod.pdfList()) == [p.gx, p.gy]
    assert p.prod.state_word() == "Dirty"
    assert p.prod.extendMode() == CAN_NOT_BE_EXTENDED


def test_a_product_takes_a_cutoff_number_and_ignores_commands_it_does_not_know() -> None:
    """``RooProdPdf(name, title, pdf1, pdf2, cutOff)`` is the same product."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    p = Product()
    cut = RooProdPdf("cut", "cut", p.gx, p.gy, 1e-5, RooCmdArg("Unknown", 1))
    assert cut.getVal([p.x, p.y]) == pytest.approx(0.0412343220226, rel=REL)
    assert cut._cutoff == 1e-5


@pytest.mark.xfail(
    strict=True,
    reason="prodpdf.py:89-92: ROOT takes a factor with no observable in the normalisation set as 1",
)
def test_a_product_normalised_over_one_factors_observables_drops_the_other_factor() -> None:
    """ROOT's ``prod.getVal([x])`` is ``gx`` normalised - ``gy``'s value plays no part."""
    p = Product()
    assert p.prod.getVal([p.x]) == pytest.approx(0.193334718961, rel=REL)
    p.y.setVal(-2.5)
    assert p.prod.getVal([p.x]) == pytest.approx(0.193334718961, rel=REL)
    assert p.prod.getVal([p.y]) == pytest.approx(0.0360470665125, rel=REL)


@pytest.mark.xfail(
    strict=True, reason="prodpdf.py: ROOT prints a product's factors, 'gx * gy', and '/1'"
)
def test_a_product_prints_its_factors_as_root_does(capsys: Any) -> None:
    """``RooProdPdf::prod[ gx * gy ] = 0.776101/1``."""
    p = Product()
    capsys.readouterr()
    p.prod.Print()
    assert capsys.readouterr().out == "RooProdPdf::prod[ gx * gy ] = 0.776101/1\n"


def test_a_product_of_factors_sharing_an_observable_is_normalised_numerically() -> None:
    """``gx(x) e(x)`` does not factorise: the product is normalised as a whole."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    p = Product()
    e = RooExponential("ex", "ex", p.x, RooRealVar("c", "c", -0.2, -2, -0.01))
    shared = RooProdPdf("shared", "shared", [p.gx, e])
    assert shared.getVal([p.x]) == pytest.approx(0.197239793463, rel=1e-9)
    assert shared.analytic_names(frozenset(["x"]), None) == frozenset()
    assert shared.getVal() == pytest.approx(p.gx.getVal() * e.getVal(), rel=REL)


@pytest.mark.xfail(
    strict=True,
    reason="prodpdf.py:141-148: ROOT integrates factors sharing an observable one by one",
)
def test_a_product_of_factors_sharing_an_observable_integrates_as_root_does() -> None:
    """ROOT's ``createIntegral`` of ``gx(x) e(x)`` is the product of the factors' integrals."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    p = Product()
    e = RooExponential("ex", "ex", p.x, RooRealVar("c", "c", -0.2, -2, -0.01))
    shared = RooProdPdf("shared", "shared", [p.gx, e])
    assert integral(shared, [p.x], "win") == pytest.approx(12.4413287728, rel=1e-9)
    assert integral(shared, [p.x]) == pytest.approx(181.823195698, rel=1e-9)


def test_a_product_with_an_extended_factor_expects_that_factors_yield() -> None:
    """The first extended factor says how many events the product expects."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    p = Product()
    ext = RooExtendPdf("ext", "ext", p.gx, RooRealVar("n", "n", 250, 0, 1000))
    pe = RooProdPdf("pe", "pe", [ext, p.gy])
    assert pe.expectedEvents([p.x, p.y]) == 250.0
    assert pe.extendMode() == 1
    assert p.prod.expected(frozenset(["x", "y"])) == 0.0


def test_a_product_draws_each_factors_observables_from_that_factor_as_root_does() -> None:
    """The events are ROOT's to the last bit."""
    p = Product()
    generator().SetSeed(4357)
    drawn = p.prod.generate([p.x, p.y], 3)
    assert column(drawn, "x") + column(drawn, "y") == [
        2.997865435218796,
        2.5635925123910157,
        2.648527370025855,
        -0.1521465841215104,
        0.4549208430107683,
        0.41492401178145255,
    ]


class Channels:
    """A Gaussian for state ``phys`` and an exponential for ``ctl``."""

    def __init__(self) -> None:
        from xrdroot.roofit.categories import RooCategory

        self.x = RooRealVar("x", "x", 0.5, -10, 10)
        self.g = RooGaussian("g1", "g1", self.x, RooRealVar("m1", "m1", 1, -5, 5), 2.0)
        self.e = RooExponential("e", "e", self.x, RooRealVar("c", "c", -0.2, -2, -0.01))
        self.cat = RooCategory("cat", "cat", {"phys": 0, "ctl": 1})


def test_a_simultaneous_density_is_its_current_states_density() -> None:
    """The category's state picks the density, each normalised over its own observables."""
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    c = Channels()
    sim = RooSimultaneous("sim", "sim", {"phys": c.g, "ctl": c.e}, c.cat)
    c.cat.setLabel("phys")
    assert sim.getVal([c.x]) == pytest.approx(0.193334718961, rel=REL)
    c.cat.setLabel("ctl")
    assert sim.getVal([c.x]) == pytest.approx(0.0249482283928, rel=REL)
    assert sim.getVal([c.x, c.cat]) == pytest.approx(0.0249482283928, rel=REL)
    assert sim.getVal() == pytest.approx(c.e.getVal(), rel=REL)
    assert sim.extendMode() == CAN_NOT_BE_EXTENDED
    assert sim.getPdf("phys") is c.g
    assert sim.getPdf("none") is None
    assert sim.indexCat() is c.cat
    assert sim.servers() == [c.cat, c.e, c.g]
    assert sim.printMetaArgs() == ""
    assert sim.addPdf(c.g, "phys") is True


def test_a_simultaneous_density_takes_a_list_in_the_order_of_the_states() -> None:
    """``RooSimultaneous(name, title, [pdfs], cat)``: the first density for the first state."""
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    c = Channels()
    sim = RooSimultaneous("sim", "sim", [c.g, c.e], c.cat)
    assert (sim.getPdf("phys"), sim.getPdf("ctl")) == (c.g, c.e)


def extend_modes() -> dict[str, int]:
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    c = Channels()
    g2 = RooGaussian("g2", "g2", c.x, -1.0, 3.0)
    n1, n2 = RooRealVar("n1", "n1", 100, 0, 1000), RooRealVar("n2", "n2", 40, 0, 1000)
    must = RooAddPdf("must", "must", [c.g, g2], [n1, n2])
    can = RooExtendPdf("can", "can", c.g, n1)
    combos = {
        "must_not": (must, g2),
        "must_can": (must, can),
        "must_must": (must, must),
        "can_not": (can, g2),
        "can_can": (can, can),
        "not_not": (c.g, g2),
    }
    return {
        label: RooSimultaneous(f"s{label}", "s", {"A": a, "B": b}, c.cat).extendMode()
        for label, (a, b) in combos.items()
    }


def test_a_simultaneous_density_of_extended_channels_is_extended() -> None:
    """All channels able to be extended, or all bound to be, make the whole so."""
    modes = extend_modes()
    assert (modes["must_must"], modes["can_can"], modes["not_not"]) == (2, 1, 0)


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous.py:96-102: ROOT's mode is Must if any channel must, Can if any can",
)
def test_a_simultaneous_density_is_extended_if_any_channel_is_as_root_says() -> None:
    """ROOT 6.40: one channel that must be extended makes the whole so; one that can, can."""
    modes = extend_modes()
    assert (modes["must_not"], modes["must_can"], modes["can_not"]) == (2, 2, 1)


def test_a_simultaneous_density_expects_the_events_of_all_its_channels_over_the_category() -> None:
    """Normalised over the category too, it expects every channel's events."""
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    c = Channels()
    n1, n2 = RooRealVar("n1", "n1", 100, 0, 1000), RooRealVar("n2", "n2", 40, 0, 1000)
    e1, e2 = RooExtendPdf("e1", "e1", c.g, n1), RooExtendPdf("e2", "e2", c.e, n2)
    sx = RooSimultaneous("sx", "sx", [e1, e2], c.cat)
    assert sx.extendMode() == 1
    assert sx.expectedEvents([c.x, c.cat]) == 140.0


@pytest.mark.xfail(
    strict=True,
    reason="simultaneous.py:104-105: without the category in nset ROOT expects the current "
    "state's events",
)
def test_a_simultaneous_density_expects_its_current_channels_events_as_root_does() -> None:
    """Asked over ``x`` alone, ROOT's simultaneous density expects the current channel's."""
    from xrdroot.roofit.pdfs.simultaneous import RooSimultaneous

    c = Channels()
    n1, n2 = RooRealVar("n1", "n1", 100, 0, 1000), RooRealVar("n2", "n2", 40, 0, 1000)
    e1, e2 = RooExtendPdf("e1", "e1", c.g, n1), RooExtendPdf("e2", "e2", c.e, n2)
    sx = RooSimultaneous("sx", "sx", [e1, e2], c.cat)
    c.cat.setLabel("ctl")
    assert sx.expectedEvents([c.x]) == 40.0


def extended_gaussian() -> tuple[RooRealVar, RooGaussian, RooRealVar, RooExtendPdf]:
    x = RooRealVar("x", "x", 0.5, -10, 10)
    x.setBins(20)
    sx = RooRealVar("sx", "sx", 2, 0.1, 10)
    gx = RooGaussian("gx", "gx", x, RooRealVar("mx", "mx", 1), sx)
    nsig = RooRealVar("nsig", "nsig", 150, 0, 1000)
    return x, gx, nsig, RooExtendPdf("ext", "ext", gx, nsig)


def test_an_extended_density_fitted_finds_the_number_of_events_as_root_does() -> None:
    """An extended fit of 180 events: the yield is ROOT's, and so is the likelihood."""
    x, gx, nsig, ext = extended_gaussian()
    generator().SetSeed(4357)
    data = gx.generate([x], 180)
    result = ext.fitTo(data, PrintLevel=-1, Save=True)
    assert nsig.getVal() == pytest.approx(179.994722932, rel=1e-8)
    assert result.minNll() == pytest.approx(-371.157925298, rel=1e-10)
    frame = x.frame()
    data.plotOn(frame)
    ext.plotOn(frame, Range=(-3.0, 3.0))
    curve = frame.getObject(1)
    assert curve.GetN() == 37
    assert [float(curve.interpolate(v)) for v in (-2.0, 0.4, 2.0)] == pytest.approx(
        [12.00562362, 33.9157324, 31.40462305], rel=1e-8
    )


def test_an_extended_factor_of_a_product_gives_the_product_its_value() -> None:
    """Unnormalised, an extended density is its shape: in a product, it multiplies as that."""
    from xrdroot.roofit.pdfs.prodpdf import RooProdPdf

    _, gx, _, ext = extended_gaussian()
    y = RooRealVar("y", "y", 1.5, -5, 5)
    gy = RooGaussian("gy", "gy", y, 0.5, 1.5)
    both = RooProdPdf("both", "both", [ext, gy])
    assert both.getVal() == pytest.approx(gx.getVal() * gy.getVal(), rel=REL)
    assert ext.selfNormalized() is True


@pytest.mark.xfail(
    strict=True,
    reason="extend.py:63-66: ROOT generates a RooExtendPdf by its own accept-reject, not its "
    "density's direct generator",
)
def test_an_extended_density_draws_its_events_as_root_does() -> None:
    """ROOT's first five events of the extended Gaussian after ``SetSeed(4357)``."""
    x, _, _, ext = extended_gaussian()
    generator().SetSeed(4357)
    assert column(ext.generate([x], 5), "x") == [
        -0.5169668318431775,
        1.330020600798889,
        -1.285572673636679,
        1.149962531744677,
        -1.1411667457119279,
    ]
