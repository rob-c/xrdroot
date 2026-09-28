"""How ``TLatex`` reads a formula before laying it out: ``CheckLatexSyntax`` and ``Analyse``.

``CheckLatexSyntax`` turns ``#left(``...``#right)`` into the operator it
stands for, escapes with an ``@`` every brace and bracket that opens no
operator, and refuses - in ROOT's own words, which ROOT prints on its
standard error - a formula whose operators are left open or do not match.
``Analyse`` then finds the first operator of each piece at its top level:
a script only with its brace, a command only if ROOT knows it and it is
followed as ROOT wants, limits only right after ``#int`` or ``#sum``.
What is not an operator is text, and is drawn as it is written.
"""

from __future__ import annotations

import pytest

from test_canvas_draw import written
from test_canvas_latex_layout import drawn, form, roots
from xrdroot.canvas.latexscan import LatexError, check, scan


def test_a_left_without_its_right_is_refused_in_roots_words():
    with pytest.raises(LatexError, match='Operators "#left" and "#right" don\'t match !'):
        check("#left(x")


def test_left_and_right_become_the_bracket_operator_they_stand_for():
    assert check("#left(x#right)") == "#(){x}"


def test_a_fraction_keeps_its_two_arguments_and_a_loose_brace_is_escaped():
    assert check("#frac{a}{b}") == "#frac{a}{b}"
    assert check("{a}") == "@{a@}"


def test_an_escaped_brace_is_kept_as_it_is_and_drawn_without_its_at():
    assert check("a@{b@}") == "a@{b@}"
    assert list(written(drawn("a@{b@}"))) == ["a{b}"]


def test_a_bracket_closed_in_braces_it_was_not_opened_in_is_refused_when_read():
    """``#font[`` opened inside a script's braces and closed outside them: ROOT refuses it."""
    checked = check("x^{#font[}]{a}")
    with pytest.raises(LatexError, match='Missing "\\["'):
        scan(checked)


def test_a_script_inside_a_script_is_not_the_top_levels_script():
    found = scan("x^{a^{b}}")
    assert found.power == 1
    assert list(written(drawn("x^{#bf{a}}"))) == ["a", "x"]


def test_limits_after_another_command_are_found_as_limits_of_their_integral():
    found = scan("#alpha#int_{0}")
    assert (found.command, found.above_place, found.close_curly) == (("alpha", 0), 1, 5)


def test_text_before_a_command_is_a_piece_of_its_own():
    found = scan("E = #gamma")
    assert (found.command, found.close_curly) == (("gamma", 4), 3)
    assert sorted(written(drawn("E = #gamma"))) == ["E =", "\N{GREEK SMALL LETTER GAMMA}"]


@pytest.mark.parametrize(("text", "width"), [("#foo", 42.0), ("#fracx", 60.0)])
def test_a_command_root_does_not_know_or_not_followed_by_its_brace_is_text(text, width):
    assert list(written(drawn(text))) == [text]
    assert scan(text).command == ("", -1)
    assert form(text).width == roots(width)
