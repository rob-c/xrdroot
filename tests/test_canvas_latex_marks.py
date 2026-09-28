"""The ``#`` commands ``TLatex`` draws: symbols, accents, brackets, fractions and settings.

Each is a ``TLatex`` in NDC at (0.1, 0.5) of a 700 by 500 canvas - the
pixel (70, 250) - in font 42 at 0.05 of the pad. A symbol ROOT draws by
hand - ``#Box``, ``#parallel`` - is lines sized by the pad alone, and so is
drawn at the same pixels whatever face is installed; an accent, a bracket
or a fraction is drawn round text whose size is the face's, and is checked
by where its strokes go against that text. Sizes that are ROOT's own -
``TLatex::GetXsize`` and ``GetYsize`` of ROOT 6.40.04 in a pad of 696 by
472 pixels - are held exactly where they do not depend on the face, and
otherwise to the hundredth of a pixel with Helvetica and a quarter without.
"""

from __future__ import annotations

import pytest

from test_canvas_draw import fills, lines, written
from test_canvas_latex_layout import ALPHA, drawn, form, roots


def strokes(ax):
    """Every line drawn on ``ax``, as its list of pixels."""
    return [line.tolist() for drawn_line in lines(ax) for line in drawn_line.lines]


def test_a_box_is_a_square_outline_drawn_side_by_side():
    assert strokes(drawn("#Box")) == [
        [[71, 250], [81, 250]], [[81, 250], [81, 239]], [[81, 239], [71, 239]],
        [[71, 239], [71, 250]],
    ]  # fmt: skip


def test_parallel_is_two_upright_bars():
    assert strokes(drawn("#parallel")) == [[[73, 255], [73, 232]], [[78, 255], [78, 232]]]


@pytest.mark.parametrize(
    ("text", "width"),
    [("#Box", 11.8), ("#mp", 11.8), ("#backslash", 11.8), ("#minus", 11.8), ("#plus", 11.8),
     ("#hbar", 11.8), ("#parallel", 16.857)],
)  # fmt: skip
def test_a_symbol_drawn_by_hand_is_as_wide_and_tall_as_root_makes_it_whatever_the_face(text, width):
    shape = form(text)
    assert (shape.width, shape.height) == pytest.approx((width, width), abs=0.001)


@pytest.mark.parametrize(
    ("text", "shown"),
    [("#minus", "−"), ("#plus", "+"), ("#backslash", "\\")],  # noqa: RUF001
)
def test_a_lettered_symbol_is_one_character_at_the_formulas_start(text, shown):
    (character,) = drawn(text).texts
    assert (character.get_text(), character.get_position()) == (shown, (70, 250))


def test_minus_plus_is_plus_minus_turned_over():
    (character,) = drawn("#mp").texts
    assert (character.get_text(), character.get_rotation()) == ("±", 180.0)


@pytest.mark.parametrize(("text", "shown"), [("#leq", "≤"), ("#AA", "Å"), ("#forall", "∀")])
def test_a_symbol_is_the_symbol_fonts_character_for_it(text, shown):
    (character,) = drawn(text).texts
    assert character.get_text() == shown


def test_a_sum_is_drawn_bigger_than_its_text_and_lowered_to_straddle_the_baseline():
    texts = written(drawn("#sum x"))
    sign, after = texts["∑"], texts["x"]
    assert sign.get_fontsize() > 1.7 * after.get_fontsize()
    assert sign.get_position()[1] > after.get_position()[1]
    assert form("#sum").over == form("#sum").under


@pytest.mark.parametrize("name", ["acute", "grave", "check", "slash"])
def test_an_accent_stands_a_third_of_the_size_higher_than_its_letter(name):
    shape = form(f"#{name}{{x}}")
    assert (shape.width, shape.height) == (roots(11.0), roots(19.867))


def test_acute_climbs_to_the_right_and_grave_to_the_left():
    (((ax1, ay1), (ax2, ay2)),) = strokes(drawn("#acute{x}"))
    (((gx1, gy1), (gx2, gy2)),) = strokes(drawn("#grave{x}"))
    assert ax2 > ax1 and ay2 < ay1
    assert gx2 < gx1 and gy2 < gy1


def test_check_is_a_trough_over_its_letter():
    (((left, y1), (middle, y2), (right, y3)),) = strokes(drawn("#check{x}"))
    assert left < middle < right and y1 == y3 < y2 < 250


def test_slash_strikes_through_its_letter_from_above_on_the_right_to_below_on_the_left():
    (((x1, y1), (x2, y2)),) = strokes(drawn("#slash{x}"))
    top = written(drawn("#slash{x}"))["x"].get_position()[1]
    assert x1 > x2 and y1 < top < y2


def test_a_dot_is_a_small_filled_square_over_its_letter_and_a_double_dot_two():
    (dot,) = fills(drawn("#dot{x}"))
    corners = dot.get_xy().tolist()
    xs, ys = sorted({x for x, _ in corners}), sorted({y for _, y in corners})
    assert len(corners) == 5 and corners[0] == corners[-1]  # a closed outline of four corners
    assert len(xs) == len(ys) == 2 and xs[1] - xs[0] <= 2 and ys[1] - ys[0] <= 2
    assert 70 < xs[0] and ys[1] < 240  # over the letter's middle, above its top
    assert len(fills(drawn("#ddot{x}"))) == 2


def test_parentheses_are_two_arcs_bowed_away_from_what_they_hold():
    ax = drawn("#left(x#right)")
    left, right = strokes(ax)
    inside = written(ax)["x"].get_position()[0]
    assert len(left) == len(right) == 41
    assert max(x for x, _ in left) <= inside < min(x for x, _ in right)
    assert min(x for x, _ in left) < left[0][0] and max(x for x, _ in right) > right[0][0]
    assert form("#left(x#right)").width == roots(31.594)


@pytest.mark.parametrize(("top", "bottom"), [("a", "bbb"), ("aaa", "b")])
def test_a_fraction_stands_its_numerator_over_its_denominator_on_a_rule(top, bottom):
    ax = drawn(f"#frac{{{top}}}{{{bottom}}}")
    texts = written(ax)
    (((left, rule), (right, level)),) = strokes(ax)
    (nx, ny), (dx, dy) = texts[top].get_position(), texts[bottom].get_position()
    assert ny < rule == level < dy
    assert min(nx, dx) == left == 70  # the narrower centred on the wider
    assert max(nx, dx) > 70 and right > max(nx, dx)
    assert form(f"#frac{{{top}}}{{{bottom}}}").width == max(form(top).width, form(bottom).width)


def test_splitline_stands_one_line_over_the_other_on_the_left_with_no_rule():
    ax = drawn("#splitline{a}{bbb}")
    texts = written(ax)
    (ax_, ay), (bx, by) = texts["a"].get_position(), texts["bbb"].get_position()
    assert (ax_, strokes(ax)) == (bx, []) and ay < by
    assert form("#splitline{a}{b}").width == roots(12.0)


@pytest.mark.parametrize("amount", [1.0, -0.5])
def test_kern_moves_what_it_holds_by_that_much_of_its_width(amount):
    x = form("x")
    assert form(f"#kern[{amount}]{{x}}y").width == pytest.approx(
        (1 + amount) * x.width + form("y").width
    )
    assert form("#kern[1]{x}y").height == roots(17.0)


def test_kern_pushes_what_follows_to_the_right():
    texts = written(drawn("#kern[1]{x}y"))
    assert 80 <= texts["x"].get_position()[0] < texts["y"].get_position()[0]


def test_lower_moves_what_it_holds_down_by_that_much_of_its_height():
    x, lowered = form("x"), form("#lower[0.5]{x}")
    assert (lowered.over, lowered.under) == pytest.approx(
        (x.over + x.height / 2, x.under + x.height / 2)
    )
    assert lowered.height == roots(24.0)
    assert written(drawn("#lower[0.5]{x}"))["x"].get_position()[1] > 250


def test_a_link_draws_only_its_text():
    """``#url[...]{x}`` is a link in ROOT's SVG and PDF; in a picture it is its text."""
    assert list(written(drawn("#url[1]{x}"))) == ["x"]
    assert form("#url[1]{x}") == form("x")


def test_color_draws_what_it_holds_in_that_colour():
    texts = written(drawn("#color[2]{x}y"))
    assert (texts["x"].get_color(), texts["y"].get_color()) == ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert form("#color[2]{x}") == form("x")


def test_bold_leaves_a_font_outside_its_table_as_it_is():
    """Font 2 is of family 0, which ``#bf`` has no bold for: it keeps its face."""
    plain = written(drawn("x", fTextFont=2))["x"].get_fontproperties().get_file()
    bold = written(drawn("#bf{x}", fTextFont=2))["x"].get_fontproperties().get_file()
    assert bold == plain


def test_italic_greek_and_symbols_are_drawn_from_the_italic_symbol_font():
    upright = written(drawn("#alpha"))[ALPHA].get_fontproperties().get_file()
    italic = written(drawn("#it{#alpha}"))[ALPHA].get_fontproperties().get_file()
    assert italic != upright
    assert written(drawn("#it{#leq}"))["≤"]
