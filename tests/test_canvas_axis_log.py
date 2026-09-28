"""``TGaxis::PaintAxis`` on a logarithmic axis, and where an axis's labels and title go.

A logarithmic axis (``chopt`` ``G``) has a long tick and a label at each
decade and short ones at the whole multiples between, fewer of both when the
decades are many. Painted with :func:`xrdroot.canvas.axis.paint_axis` on a
700 by 500 canvas, the ticks are read back in its pixels, and the labels and
the title as the point of the pad's NDC each is anchored at, with its
alignment and angle - which is what ROOT places too, before any font is
measured. ROOT's own ``TGaxis`` drew every case and its labels, ticks and
anchors were compared.
"""

from __future__ import annotations

from collections import Counter

import pytest

from xrdroot.canvas.axis import Axis, paint_axis

WIDTH, HEIGHT = 700, 500


def pixel(u, v):
    """A point of the pad's NDC in the canvas's whole pixels."""
    return round(u * WIDTH), round((1 - v) * HEIGHT)


def level(wmin, wmax, ndiv=510, chopt="G", **members):
    """A level axis across the pad, a fifth of the way up."""
    axis = Axis(0.1, 0.2, 0.9, 0.2, wmin, wmax, ndiv, chopt, label_size=0.04, **members)
    return paint_axis(axis, pixel)


def upright(wmin, wmax, ndiv=510, chopt="G", **members):
    """An axis up the pad's left, a tenth of the way in."""
    axis = Axis(0.1, 0.2, 0.1, 0.9, wmin, wmax, ndiv, chopt, label_size=0.04, **members)
    return paint_axis(axis, pixel)


def texts(out):
    return [label.text for label in out.labels]


def ticks(out):
    """How many ticks of each length the axis has, its line left out."""
    return Counter(
        abs(y0 - y1) + abs(x0 - x1)
        for x0, y0, x1, y1 in ((*pixel(a, b), *pixel(c, d)) for a, b, c, d in out.lines[1:])
    )


def test_a_decade_has_a_long_tick_and_its_multiples_short_ones():
    out = level(1.0, 1000.0)
    assert texts(out) == ["1", "10", "10^{2}", "10^{3}"]
    assert ticks(out) == {12: 4, 6: 24}
    assert [label.u for label in out.labels] == pytest.approx([0.1, 0.3667, 0.6333, 0.9], abs=1e-4)


def test_an_axis_ending_short_of_a_decade_stops_after_its_last_multiple():
    out = level(1.0, 9.5)
    assert texts(out) == ["1"]
    assert ticks(out) == {12: 1, 6: 8}


def test_the_next_decade_past_the_axis_is_neither_ticked_nor_labelled():
    out = level(0.01, 0.95)
    assert texts(out) == ["10^{#minus2}", "10^{#minus1}"]
    assert ticks(out) == {12: 2, 6: 16}


def test_many_decades_are_labelled_only_every_few_and_ticked_only_at_five_times():
    out = level(1e-10, 1e10, 505)
    assert texts(out) == ["10^{#minus8}", "10^{#minus4}", "1", "10^{4}", "10^{8}", "10^{10}"]
    assert ticks(out) == {12: 21, 6: 20}


def test_chopt_u_ticks_a_logarithmic_axis_but_leaves_it_unlabelled():
    out = level(1.0, 1000.0, chopt="GU")
    assert texts(out) == []
    assert ticks(out) == {12: 4, 6: 24}


def test_no_primary_divisions_leaves_a_logarithmic_axis_unlabelled():
    assert texts(level(1.0, 1000.0, 100)) == []


def test_a_logarithmic_axis_reaching_zero_or_below_is_its_line_alone():
    out = level(-1.0, 1000.0)
    assert out.lines == [(0.1, 0.2, 0.9, 0.2)]
    assert out.labels == []


def test_no_exponent_writes_each_decade_as_its_number_a_third_of_a_label_higher():
    plain = level(0.001, 10.0)
    out = level(0.001, 10.0, bits=frozenset({"noexponent"}))
    assert texts(out) == ["0.001", "0.01", "0.1", "1", "10"]
    assert out.labels[0].v == pytest.approx(plain.labels[0].v + 0.33 * 0.04)


def test_an_upright_logarithmic_axis_writes_its_decades_off_the_side_its_ticks_are_not():
    out = upright(1.0, 1000.0)
    assert texts(out) == ["1", "10", "10^{2}", "10^{3}"]
    assert ticks(out) == {15: 4, 7: 24}
    assert [label.u for label in out.labels] == pytest.approx([0.0918] * 4)
    assert [label.v for label in out.labels] == pytest.approx([0.2, 0.4333, 0.6667, 0.9], abs=1e-4)
    assert {label.align for label in out.labels} == {32}


@pytest.mark.parametrize("chopt", ["G=", "G-="])
def test_decades_written_on_their_ticks_side_stand_off_them_the_one_less_far(chopt):
    out = upright(1.0, 1000.0, chopt=chopt)
    # ROOT moves "1" one label height off, and the raised 10^{n} two.
    assert [label.u for label in out.labels] == pytest.approx([0.1528, 0.1928, 0.1928, 0.1928])


def test_ticks_turned_to_the_left_leave_the_decades_just_right_of_the_axis():
    out = upright(1.0, 1000.0, chopt="G+")
    assert [label.u for label in out.labels] == pytest.approx([0.1118] * 4)
    tick = out.lines[1]
    assert tick[0] < 0.1 == tick[2]  # it stands out to the left


def test_an_upright_axis_writes_its_power_of_a_thousand_just_above_its_top():
    out = upright(0.0, 2e6, chopt="")
    assert texts(out)[-2:] == ["2000", "#times10^{3}"]
    exponent = out.labels[-1]
    assert (exponent.u, exponent.align) == (0.1, 11)
    assert exponent.v == pytest.approx(0.9 + 0.1 * 0.04)
