"""``TGaxis::ChangeLabel``: a label restyled, rewritten or erased, counted from either end."""

from __future__ import annotations

from xrdroot.canvas.axis import Axis, paint_axis
from xrdroot.pyroot.graphics import TGaxis


def labels(*changed):
    axis = Axis(0.1, 0.1, 0.9, 0.1, 0.0, 10.0, ndiv=5, changed=changed)
    return paint_axis(axis, lambda u, v: (u * 700, v * 500)).labels


def test_a_changed_label_takes_each_attribute_given_and_keeps_the_rest():
    first, *_, last = labels((1, 30.0, 0.05, 22, 2, 62, "zero"), (-1, -1.0, -1.0, -1, -1, -1, ""))
    assert (first.text, first.angle, first.size, first.align, first.color, first.font) == (
        "zero", 30.0, 0.05, 22, 2, 62)  # fmt: skip
    assert (last.text, last.size, last.font) == ("10", 0.035, 42)


def test_a_label_of_size_zero_is_erased_and_the_first_change_of_a_label_wins():
    texts = [
        label.text for label in labels((-1, -1.0, 0.0, -1, -1, -1, ""), (6, -1, 1, -1, -1, -1, "x"))
    ]
    assert texts == ["0", "2", "4", "6", "8"]


def test_a_gaxis_keeps_one_change_per_label_and_forgets_them_all_at_zero():
    axis = TGaxis(0, 0, 1, 0, 0, 1)
    axis.ChangeLabel(1, -1, 0)
    axis.ChangeLabel(1, 45)
    assert axis.members["_changed_labels"] == [(1, 45.0, -1.0, -1, -1, -1, "")]
    axis.ChangeLabel(0)
    assert axis.members["_changed_labels"] == []
