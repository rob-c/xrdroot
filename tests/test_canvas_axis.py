"""``TGaxis::PaintAxis`` on a linear axis: its divisions, its ticks and its labels' numbers.

An axis is painted here with :func:`xrdroot.canvas.axis.paint_axis` straight,
on a 700 by 500 canvas, and what it paints is read back in the canvas's
pixels (``y`` down): the axis line first, then a tick per division, and the
labels' text. Every case was drawn by ROOT's own ``TGaxis`` and its labels
and tick pixels compared; the labels are written as ``TLatex``, so a minus is
``#minus`` and the common factor is ``#times10^{n}``.
"""

from __future__ import annotations

from collections import Counter

from xrdroot.canvas.axis import Axis, paint_axis

WIDTH, HEIGHT = 700, 500
NOEXPONENT = frozenset({"noexponent"})


def pixel(u, v):
    """A point of the pad's NDC in the canvas's whole pixels."""
    return round(u * WIDTH), round((1 - v) * HEIGHT)


def painted(wmin, wmax, ndiv=510, chopt="", **members):
    """What ``PaintAxis`` paints for a level axis across the pad, a fifth of the way up."""
    return paint_axis(Axis(0.1, 0.2, 0.9, 0.2, wmin, wmax, ndiv, chopt, **members), pixel)


def texts(out):
    return [label.text for label in out.labels]


def segments(out):
    """The axis's segments in pixels, the axis line first."""
    return [(*pixel(x0, y0), *pixel(x1, y1)) for x0, y0, x1, y1 in out.lines]


def tick_lengths(out):
    """How many ticks of each length, in pixels, the axis has."""
    return Counter(abs(y0 - y1) for _x0, y0, _x1, y1 in segments(out)[1:])


def test_a_third_level_of_divisions_puts_quarter_ticks_within_the_secondary_ones():
    out = painted(0.0, 1.0, 50510)
    assert texts(out) == ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]
    assert tick_lengths(out) == {12: 11, 6: 40, 3: 150}
    assert segments(out)[1:4] == [(70, 388, 70, 400), (73, 397, 73, 400), (76, 397, 76, 400)]


def test_chopt_n_takes_the_divisions_as_they_are_over_the_whole_axis():
    out = painted(0.0, 1.0, 510, "N")
    assert tick_lengths(out) == {12: 11, 6: 40}
    assert segments(out)[1:3] == [(70, 388, 70, 400), (81, 394, 81, 400)]
    assert texts(out)[-1] == "1"


def test_one_division_is_a_tick_at_each_end_and_no_optimising():
    out = painted(0.0, 1.0, 1)
    assert texts(out) == ["0", "1"]
    assert segments(out)[1:] == [(70, 388, 70, 400), (630, 388, 630, 400)]


def test_no_primary_divisions_ticks_the_ends_and_writes_no_labels():
    out = painted(0.0, 1.0, 100)
    assert texts(out) == []
    assert segments(out)[1:] == [(70, 388, 70, 400), (630, 388, 630, 400)]


def test_round_divisions_that_would_leave_the_axis_are_given_up_for_its_own_ends():
    # Optimize puts 6.34 to 6.77 in two halves from 6 to 7; neither is on the axis.
    out = painted(6.34, 6.77, 2)
    assert texts(out) == ["6.34", "6.555", "6.77"]
    assert segments(out)[1:] == [(70, 388, 70, 400), (350, 388, 350, 400), (630, 388, 630, 400)]


def test_a_label_a_rounding_error_below_zero_is_written_zero_not_minus_zero():
    labels = texts(painted(-0.3, 0.7))
    assert labels[:4] == ["#minus0.3", "#minus0.2", "#minus0.1", "0"]
    assert "#minus0" not in labels


def test_labels_far_from_zero_with_small_steps_get_the_decimals_their_steps_need():
    labels = texts(painted(20000.0, 20000.5, 505))
    assert labels == ["20000", "20000.1", "20000.2", "20000.3", "20000.4", "20000.5"]


def test_steps_of_millionths_near_one_are_written_in_full_with_no_exponent():
    labels = texts(painted(1.0, 1.00003))
    assert labels == ["1", "1.000005", "1.00001", "1.000015", "1.00002", "1.000025", "1.00003"]


def test_steps_of_millionths_of_small_labels_are_shown_times_a_power_of_a_thousand():
    labels = texts(painted(0.0, 3e-5))
    assert labels == ["0", "5", "10", "15", "20", "25", "30", "#times10^{#minus6}"]


def test_steps_far_below_the_labels_widen_the_format_until_they_show():
    labels = texts(painted(1e-3, 1.000001e-3))
    assert labels[:3] == ["1", "1.0000001", "1.0000002"]
    assert labels[-2:] == ["1.000001", "#times10^{#minus3}"]


def test_labels_of_millions_are_shown_in_thousands_times_ten_cubed():
    out = painted(0.0, 2e6)
    assert texts(out) == [str(n) for n in range(0, 2001, 200)] + ["#times10^{3}"]
    exponent = out.labels[-1]
    assert exponent.align == 11
    assert exponent.u > 0.9 > out.labels[-2].u - 0.01  # just past the axis's far end
    assert (exponent.font, exponent.angle) == (42, 0.0)


def test_negative_millions_keep_their_minus_sign_before_the_thousands():
    labels = texts(painted(-2e6, 1e6))
    assert labels == ["#minus2000", "#minus1500", "#minus1000", "#minus500", "0", "500", "1000",
                      "#times10^{3}"]  # fmt: skip


def test_labels_of_ten_thousandths_are_shown_in_thousandths_times_ten_to_the_minus_three():
    labels = texts(painted(0.0, 1e-4))
    assert labels == ["0", "0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08",
                      "0.09", "0.1", "#times10^{#minus3}"]  # fmt: skip


def test_no_exponent_writes_the_numbers_in_full_however_long_or_small():
    big = texts(painted(0.0, 2e6, bits=NOEXPONENT))
    assert big == ["0"] + [str(n) for n in range(200000, 2000001, 200000)]
    small = texts(painted(0.0, 3e-5, bits=NOEXPONENT))
    assert small == ["0", "0.000005", "0.00001", "0.000015", "0.00002", "0.000025", "0.00003"]


def test_chopt_dot_keeps_the_point_of_a_whole_label():
    assert texts(painted(0.0, 0.1, 50510, "."))[:2] == ["0.", "0.01"]


def test_an_axis_of_no_range_or_no_divisions_is_its_line_alone():
    for out in (painted(5.0, 5.0), painted(0.0, 5.0, 0)):
        assert segments(out) == [(70, 400, 630, 400)]
        assert out.labels == []


def test_chopt_b_leaves_out_the_axis_line_but_not_its_ticks():
    out = painted(0.0, 10.0, 510, "B")
    assert (70, 400, 630, 400) not in segments(out)
    assert segments(out)[0] == (70, 388, 70, 400)
    assert len(texts(out)) == 11
