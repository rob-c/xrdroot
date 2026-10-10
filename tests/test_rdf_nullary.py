"""A ``Define`` of a callable that reads no columns is called once an entry, as ROOT calls it.

Such a callable captures something of its own - a counter, say - and
ROOT calls it entry by entry, the columns defined one after another each
in turn for every entry, so that what one changes the next sees. Here a
batch of entries is computed at once, and these callables, defined one
after another at the same node, are called for each entry in their order.
"""

from __future__ import annotations

import numpy as np

from xrdroot.rdf import RDataFrame


def _counting():
    state = {"i": 0}

    def b1():
        return state["i"]

    def b2():
        j = float(state["i"] * state["i"])
        state["i"] += 1
        return j

    return b1, b2


def test_callables_of_no_columns_are_called_an_entry_at_a_time_each_in_turn():
    b1, b2 = _counting()
    frame = RDataFrame(100).Define("b1", b1).Define("b2", b2)
    found = frame.AsNumpy(["b1", "b2"])
    assert found["b1"].tolist() == list(range(100))
    assert found["b2"].tolist() == [float(i * i) for i in range(100)]
    assert frame.Range(50).Filter("1 == b1 % 2").Count().GetValue() == 25


def test_a_callable_of_no_columns_alone_is_still_called_once_an_entry():
    calls = []

    def tick():
        calls.append(len(calls))
        return len(calls)

    frame = RDataFrame(7).Define("n", tick)
    assert np.array_equal(np.asarray(frame.Take("n").GetValue()), np.arange(1, 8))
    assert len(calls) == 7


def test_a_callable_after_one_that_reads_columns_or_at_another_node_is_called_on_its_own():
    b1, b2 = _counting()
    frame = RDataFrame(5).Define("twice", "rdfentry_ * 2").Define("b1", b1)
    assert frame.AsNumpy(["b1"])["b1"].tolist() == [0, 0, 0, 0, 0]  # nothing counts it on
    apart = RDataFrame(5).Define("b1", b1).Filter("rdfentry_ >= 0").Define("b2", b2)
    assert apart.AsNumpy(["b2"])["b2"].tolist() == [0.0, 1.0, 4.0, 9.0, 16.0]
    words = RDataFrame(3).Define("word", lambda: "entry")
    assert words.Take("word").GetValue() == ["entry"] * 3  # not numbers: kept as a list


def test_a_macros_filter_of_no_columns_is_still_counted_by_the_entry_number():
    import xrdroot.pyroot as ROOT

    calls: list[int] = []

    def keep():
        calls.append(1)
        return len(calls) % 2 == 1

    keep.__module__ = "__cint__"
    assert ROOT.RDataFrame(6).Filter(keep).Count().GetValue() == 3 and len(calls) == 6


def test_a_macros_lambdas_of_no_columns_go_to_the_frame_as_they_are_to_be_called_in_turn():
    import xrdroot.pyroot as ROOT

    b1, b2 = _counting()
    b1.__module__ = b2.__module__ = "__cint__"  # a translated macro's, as ROOT's lambdas are
    frame = ROOT.RDataFrame(10).Define("b1", b1).Define("b2", b2)
    found = frame.AsNumpy(["b1", "b2"])
    assert found["b1"].tolist() == list(range(10))
    assert found["b2"].tolist() == [float(i * i) for i in range(10)]
