"""ROOT's ``#`` mathematics, laid out as ``TLatex`` lays it out, against ROOT's own sizes.

ROOT 6.40.04 measured each of these strings - ``TLatex::GetXsize`` and
``GetYsize`` for a ``TLatex`` of font 42 and size 0.05, NDC, in a 700 by 500
canvas (a pad of 696 by 472 pixels) - on macOS, whose TeX Gyre Heros has
Helvetica's widths. Laid out here in the first Helvetica installed, each is
ROOT's to the fraction of a pixel ROOT keeps. Where only a wider face is
installed - matplotlib's DejaVu Sans, which is a tenth or so wider and taller
- each is held to a quarter of ROOT's instead, which is what the layout,
rather than the face, is answerable for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from xrdroot.canvas import fonts
from xrdroot.canvas.latex import formula_form

#: Whether ROOT's 4x is a face with Helvetica's widths here, as it was where ROOT measured.
HELVETICA = Path(fonts.face(42)).stem.lower().replace(" ", "").startswith(
    ("helvetica", "texgyreheros", "nimbussans", "arial")
)
#: The pad ROOT measured in: a 700 by 500 canvas's, less its window's edges.
PAD = (696, 472)

#: Each string, and ROOT's width and height of it in the pad's pixels.
MEASURED = [
    ("x", 11.0, 12.0),
    ("x^{2}", 19.0, 18.2),
    ("x_{i}", 14.0, 18.6),
    ("x^{2}_{i}", 19.0, 24.8),
    ("#sqrt{x}", 22.8, 17.9),
    ("#sqrt[3]{x}", 25.16, 29.9),
    ("#bar{x}", 11.0, 19.87),
    ("#hat{x}", 11.0, 19.87),
    ("#vec{v}", 11.0, 17.9),
    ("#tilde{n}", 11.0, 19.87),
    ("#dot{x}", 11.0, 19.87),
    ("#ddot{x}", 11.0, 19.87),
    ("#left[x#right]", 28.7, 12.0),
    ("#left|x#right|", 28.7, 12.0),
    ("#left{x#right}", 18.5, 12.0),
    ("#font[12]{a}", 11.0, 11.0),
    ("#it{i}", 6.0, 16.0),
    ("#scale[2]{s}", 21.0, 25.0),
    ("#mbox{m}", 17.0, 12.0),
    ("#hbar", 11.8, 11.8),
    ("#odot", 11.8, 11.8),
    ("p_{T} [GeV]", 82.0, 28.6),
    ("Events / ( 0.5 x 0.5 )", 196.0, 22.0),
]


def roots(value: float) -> object:
    """ROOT's measure, to the hundredth of a pixel with Helvetica and to a quarter without."""
    return pytest.approx(value, abs=0.01) if HELVETICA else pytest.approx(value, rel=0.25)


@pytest.mark.parametrize(("text", "width", "height"), MEASURED)
def test_tlatex_is_as_wide_and_as_tall_as_root_measures_it(text, width, height):
    form = formula_form(text, 0.05, 42, PAD, float(PAD[1]))
    assert (form.width, form.over + form.under) == (roots(width), roots(height))


@pytest.mark.parametrize("text", ["x^{2", "#perp"])
def test_what_root_cannot_lay_out_has_no_size(text):
    """A brace left open is refused, as ``CheckLatexSyntax`` refuses it; ``#perp`` is drawn
    as lines of its own, and measures nothing."""
    form = formula_form(text, 0.05, 42, PAD, float(PAD[1]))
    assert (form.width, form.over + form.under) == (0.0, 0.0)


@pytest.mark.parametrize(
    "text", ["#color{x}", "#color[a]{x}", "#frac{a}}{b}", "x^{#font[}]{a}", "#url{x}"]
)
def test_a_setting_or_a_group_root_refuses_measures_nothing_rather_than_failing(text):
    """ROOT's ``GetXsize`` and ``GetYsize`` are 0 for each of these, and a legend or a pave
    sizing its text by one must go on."""
    form = formula_form(text, 0.05, 42, PAD, float(PAD[1]))
    assert (form.width, form.over, form.under) == (0.0, 0.0, 0.0)


def test_a_link_is_no_setting_and_its_text_is_laid_out_alone():
    linked = formula_form("#url[https://root.cern]{x}", 0.05, 42, PAD, float(PAD[1]))
    assert linked == formula_form("x", 0.05, 42, PAD, float(PAD[1]))
