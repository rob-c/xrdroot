"""How ``TLatex::Analyse`` puts the pieces of a formula together, drawn on a canvas.

Each formula here is a ``TLatex`` in NDC at (0.1, 0.5) of a 700 by 500
canvas - the pixel (70, 250) - in font 42 at 0.05 of the pad, and is
measured, by ``formula_form``, in the pad of 696 by 472 pixels that ROOT
measured in. What is checked is the layout rather than the face: where a
script, a limit or an isotope's number goes against what it is on, how much
smaller each script of a script is drawn, how a font's precision decides
whether the text is a formula at all, and that a formula ROOT refuses is
not drawn. Where a size is ROOT's own - ``TLatex::GetXsize`` and
``GetYsize`` of ROOT 6.40.04 - it is held to the hundredth of a pixel where
a face of Helvetica's widths is installed, and to a quarter elsewhere, as
:mod:`test_canvas_latex` holds it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from test_canvas_draw import NDC, fills, lines, make, prim, written
from xrdroot.canvas import fonts
from xrdroot.canvas.latex import formula_form

#: Whether ROOT's 4x is a face with Helvetica's widths here, as it was where ROOT measured.
HELVETICA = (
    Path(fonts.face(42))
    .stem.lower()
    .replace(" ", "")
    .startswith(("helvetica", "texgyreheros", "nimbussans", "arial"))
)
#: The pad ROOT measured in: a 700 by 500 canvas's, less its window's edges.
PAD = (696, 472)
#: The Symbol font's alpha, as it is written here.
ALPHA = "\N{GREEK SMALL LETTER ALPHA}"


def drawn(text, **members):
    """The axes a ``TLatex`` of ``text`` is drawn on, at the pixel (70, 250)."""
    latex = prim("TLatex", fTitle=text, fX=0.1, fY=0.5, fBits=0x03000000 | NDC, **members)
    return make([(latex, "")]).plot().axes[0]


def form(text, size=0.05, font=42):
    """``text``'s form in the pad, as ``TLatex::GetBoundingBox`` measures it."""
    return formula_form(text, size, font, PAD, float(PAD[1]))


def roots(value):
    """ROOT's measure, to the hundredth of a pixel with Helvetica and to a quarter without."""
    return pytest.approx(value, abs=0.01) if HELVETICA else pytest.approx(value, rel=0.25)


def test_a_script_of_a_script_shrinks_until_it_is_a_third_and_more_as_small_then_stays():
    ax = drawn("x^{x^{x^{x^{x}}}}")
    sizes = [text.get_fontsize() for text in ax.texts][::-1]  # the base first
    assert sizes[0] > sizes[1] > sizes[2] > sizes[3] == sizes[4]
    shape = form("x^{x^{x^{x^{x}}}}")
    assert (shape.width, shape.height) == (roots(32.0), roots(22.0))


def test_empty_braces_before_a_script_put_it_before_what_follows_as_an_isotope_is_written():
    texts = written(drawn("{}^{14}C"))
    assert sorted(texts) == ["14", "C"]  # the braces drawn as a blank, which draws nothing
    (mx, my), (ex, ey) = texts["14"].get_position(), texts["C"].get_position()
    assert mx < ex and my < ey  # before the element, and raised
    assert form("{}^{14}C") == form("I^{14}C")  # the braces measured as an I


def test_a_subscript_written_first_goes_under_the_superscript_as_it_does_written_second():
    first, second = form("x_{i}^{2}"), form("x^{2}_{i}")
    assert first == second
    assert (first.width, first.height) == (roots(19.0), roots(24.8))
    texts = written(drawn("x_{i}^{2}"))
    (ux, uy), (px, py) = texts["i"].get_position(), texts["2"].get_position()
    assert ux == px and py < texts["x"].get_position()[1] < uy


@pytest.mark.parametrize("text", ["#int_{0}^{1}", "#int^{1}_{0}", "#sum_{0}^{1}", "#sum^{1}_{0}"])
def test_limits_go_over_and_under_an_integral_or_a_sum_rather_than_beside_it(text):
    texts = written(drawn(text))
    sign = texts["∫" if "int" in text else "∑"].get_position()
    over, under = texts["1"].get_position(), texts["0"].get_position()
    assert over[1] < sign[1] < under[1]
    assert sign[0] <= over[0] and sign[0] <= under[0]  # a limit narrower than it is inside it
    assert form(text).width == form(text[:4]).width


@pytest.mark.parametrize("text", ["#int^{abcdef}", "#sum_{abcdef}"])
def test_a_limit_wider_than_its_sign_is_as_wide_as_the_formula_and_the_sign_centred_on_it(text):
    texts = written(drawn(text))
    sign = texts["∫" if "int" in text else "∑"].get_position()
    assert texts["abcdef"].get_position()[0] == 70 < sign[0]
    assert form(text).width == form("abcdef", 0.05 / 1.5).width


def test_a_limit_after_another_command_still_goes_under_its_integral():
    texts = written(drawn("#alpha#int_{0}"))
    alpha, sign, limit = (texts[key].get_position() for key in (ALPHA, "∫", "0"))
    assert alpha[0] < sign[0] <= limit[0] and sign[1] < limit[1]


def test_braces_that_close_and_open_again_in_a_script_are_the_scripts_text():
    """``{#frac{a}}{b}`` is one group then another, not an argument: ROOT draws it as it is."""
    texts = written(drawn("x^{#frac{a}}{b}"))
    assert sorted(texts) == ["x", "{#frac{a}}{b}"]
    assert form("x^{#frac{a}}{b}").height == roots(18.2)


def test_a_font_of_precision_0_or_1_is_no_formula_and_draws_its_text_as_it_is():
    for font in (40, 41):
        assert list(written(drawn("#alpha^{2}", fTextFont=font))) == ["#alpha^{2}"]


def test_a_font_of_precision_3_lays_a_formula_out_at_its_size_in_pixels():
    assert form("x^{2}", 23.6, 43) == form("x^{2}", 0.05, 42)
    by_fraction = written(drawn("x^{2}"))
    in_pixels = written(drawn("x^{2}", fTextFont=43, fTextSize=25.0))  # 0.05 of 500
    assert {key: text.get_fontsize() for key, text in in_pixels.items()} == {
        key: text.get_fontsize() for key, text in by_fraction.items()
    }


@pytest.mark.parametrize("members", [{"fTitle": ""}, {"fTextSize": -0.05}])
def test_a_formula_with_no_text_or_no_size_draws_nothing(members):
    latex = prim("TLatex", **{"fTitle": "x", "fX": 0.1, "fY": 0.5, **members})
    assert list(make([(latex, "")]).plot().axes[0].texts) == []


#: Formulas ROOT refuses - ``GetXsize`` and ``GetYsize`` are both 0 - and draws nothing of.
REFUSED = ["x^{2", "#left(x", "#color{x}", "#color[a]{x}", "#frac{a}}{b}", "x^{#font[}]{a}"]


@pytest.mark.parametrize("text", REFUSED)
def test_a_formula_root_refuses_draws_nothing_of_itself(text):
    ax = drawn(f"y {text}")
    assert (list(ax.texts), lines(ax), fills(ax)) == ([], [], [])
