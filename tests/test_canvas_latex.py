"""ROOT's ``#`` mathematics, as the mathtext matplotlib draws.

Every translation is also handed to matplotlib's own mathtext parser, so a
string that translates but will not draw is a failure here rather than in
somebody's saved canvas.
"""

from __future__ import annotations

import pytest
from matplotlib.mathtext import MathTextParser

from xrdroot.canvas.latex import ACCENTS, FACES, SYMBOLS, translate

PARSER = MathTextParser("agg")


def drawable(text: str) -> bool:
    PARSER.parse(text, 72)
    return True


@pytest.mark.parametrize(
    ("root", "mathtext"),
    [
        ("Entries", "Entries"),
        ("cost $5", r"cost \$5"),
        ("#mu^{+}#mu^{-}", r"${\mu}^{\mathrm{+}}{\mu}^{\mathrm{-}}$"),
        ("#sqrt{s} = 13 TeV", r"$\sqrt{\mathrm{s}}\mathrm{\ =\ 13\ TeV}$"),
        ("#sqrt[3]{x}", r"$\sqrt[3]{\mathrm{x}}$"),
        ("p_{T} [GeV]", r"$\mathrm{p}_{\mathrm{T}}\mathrm{\ [GeV]}$"),
        ("x_1", r"$\mathrm{x}_{\mathrm{1}}$"),
        ("#bar{p}p #rightarrow X", r"$\bar{\mathrm{p}}\mathrm{p\ }{\rightarrow}\mathrm{\ X}$"),
        ("x #pm 1%", r"$\mathrm{x\ }{\pm}\mathrm{\ 1\%}$"),
        ("#splitline{a}{b}", r"$\genfrac{}{}{0}{}{\mathrm{a}}{\mathrm{b}}$"),
        ("#frac{1}{2}", r"$\frac{\mathrm{1}}{\mathrm{2}}$"),
        ("#it{p}_{T}", r"${\mathit{p}}_{\mathrm{T}}$"),
        ("#bf{b}", r"${\mathbf{b}}$"),
        ("#font[42]{f} #color[2]{c}", r"${\mathrm{f}}\mathrm{\ }{\mathrm{c}}$"),
        ("#left(x#right)", r"$\mathrm{(}\mathrm{x}\mathrm{)}$"),
        ("#left{x#right}", r"$\mathrm{\{}\mathrm{x}\mathrm{\}}$"),
        ("{a}^{2}", r"${\mathrm{a}}^{\mathrm{2}}$"),
        ("#unknown", r"$\mathrm{unknown}$"),
        ("#{", r"$\mathrm{\{}$"),
        ("#Alpha #Omega", r"${\mathrm{A}}\mathrm{\ }{\Omega}$"),
        ("a}#b", r"$\mathrm{a\}}\mathrm{b}$"),
        ("#&", r"$\mathrm{&}$"),
        ("#sqrt", r"$\sqrt{\ }$"),
        ("x^{}", r"$\mathrm{x}^{\ }$"),
    ],
)
def test_root_latex_translates_to_mathtext_matplotlib_draws(root, mathtext):
    assert translate(root) == mathtext
    assert drawable(mathtext)


@pytest.mark.parametrize("name", sorted({*SYMBOLS, *ACCENTS, *FACES}))
def test_every_command_this_knows_draws(name):
    argument = "{x}" if name in ACCENTS or name in FACES else ""
    assert drawable(translate(f"#{name}{argument}"))


def test_a_delimiter_with_nothing_after_it_is_nothing():
    assert translate("#left") == ""


def test_a_group_left_open_ends_where_the_text_does():
    assert drawable(translate("x^{2"))
    assert drawable(translate("#sqrt[3{x}"))
