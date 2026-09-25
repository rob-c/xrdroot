"""The event loop: lazy, one pass however many results, and the same answer in parallel.

Reads are counted where they happen - each basket fetched from a file - so
"nothing is read until a result is asked for" and "each basket is read once
for every result booked" are checked rather than trusted. The parallel tests
run the same analysis in one process and in several and compare the results
to the last bit, pickled.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import xrdroot
from frames import DATA, write_xyn
from xrdroot import Jagged, RDataFrame, RunGraphs, UnsupportedFeatureError
from xrdroot.rdf import loop
from xrdroot.rdf.frame import (
    DisableImplicitMT,
    EnableImplicitMT,
    GetThreadPoolSize,
    IsImplicitMTEnabled,
)
from xrdroot.tree import Basket

FLAT = [str(DATA / "chain.flat.1.root"), str(DATA / "chain.flat.2.root")]


@pytest.fixture
def baskets(monkeypatch):
    """Every basket read from a file, by where it starts, as it is read."""
    read: list[int] = []
    original = Basket.keyed.__func__

    def counted(cls, source, seek, nbytes, has_offsets):
        read.append(seek)
        return original(cls, source, seek, nbytes, has_offsets)

    monkeypatch.setattr(Basket, "keyed", classmethod(counted))
    return read


@pytest.fixture
def xyn_path(tmp_path):
    return write_xyn(tmp_path / "xyn.root", entries=1000, basket_size=512)


def _branch_baskets(path: str, names: list[str]) -> int:
    with xrdroot.open_root(path) as f:
        return sum(f["tree"][name].num_baskets for name in names)


def test_nothing_is_read_until_a_result_is_asked_for(xyn_path, baskets):
    with RDataFrame("tree", xyn_path, step=100) as df:
        mass = df.Filter("x > 10", "cut").Define("z", "x * y")
        results = [
            mass.Histo1D(("h", "", 10, -1e6, 0), "z"),
            mass.Sum("x"),
            df.Report(),
            df.Count(),
        ]
        assert baskets == [] and not any(result.IsReady() for result in results)
        assert results[1].GetValue() == sum(range(11, 1000))
        assert all(result.IsReady() for result in results)


def test_every_basket_is_read_once_for_every_result_booked(xyn_path, baskets):
    with RDataFrame("tree", xyn_path, step=64) as df:
        booked = [
            df.Sum("x"),
            df.Mean("x"),
            df.Histo1D(("h", "", 10, 0, 1000), "x", "y"),
            df.Take("y"),
        ]
        booked += [df.Filter("x > 500").Define("r", "x / y").Max("r") for _ in range(3)]
        RunGraphs(booked)
    assert len(baskets) == len(set(baskets)) == _branch_baskets(xyn_path, ["x", "y"])


def test_a_column_nothing_reaches_is_not_read(xyn_path, baskets):
    with RDataFrame("tree", xyn_path, step=100) as df:
        assert df.Filter("x < 0").Sum("y").GetValue() == 0.0
    assert len(baskets) == _branch_baskets(xyn_path, ["x"])


def test_frames_over_one_tree_share_one_loop(xyn_path, baskets):
    with xrdroot.open_root(xyn_path) as f:
        tree = f["tree"]
        first, second = RDataFrame(tree, step=100), RDataFrame(tree, step=100)
        results = [first.Sum("x"), second.Filter("x < 10").Sum("x"), first.Count()]
        assert RunGraphs(results) == 1
        assert [result.GetValue() for result in results] == [sum(range(1000)), 45.0, 1000]
        assert (first.GetNRuns(), second.GetNRuns()) == (1, 1)
    assert len(baskets) == _branch_baskets(xyn_path, ["x"])
    assert RunGraphs(results) == 0  # everything is computed already
    other = RDataFrame(3)
    assert RunGraphs([other.Count(), RDataFrame(4).Count(), other.Sum("rdfentry_")]) == 2


class Recorder:
    """A column of entry numbers that remembers the size of every batch it was asked for."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def __call__(self, rdfentry_: np.ndarray) -> np.ndarray:
        self.sizes.append(len(rdfentry_))
        return rdfentry_.astype(np.int64)


def test_a_range_at_the_head_reads_no_further_than_its_end():
    record = Recorder()
    df = RDataFrame(1000, step=10).Define("i", record)
    assert df.Range(25).Take("i").GetValue().tolist() == list(range(25))
    assert record.sizes == [10, 10, 5]


def test_the_loop_stops_once_every_range_is_past_its_end():
    record = Recorder()
    df = RDataFrame(1000, step=10).Define("i", record).Filter("i % 2 == 0")
    assert df.Range(5).Take("i").GetValue().tolist() == [0, 2, 4, 6, 8]
    assert record.sizes == [10]


def test_a_display_stops_the_loop_once_it_has_its_entries():
    record = Recorder()
    shown = RDataFrame(1000, step=10).Define("i", record).Display(["i"], 12).GetValue()
    assert str(shown).count("\n") == 3 + 2 * 12 and record.sizes == [10, 10]


def test_a_named_filter_keeps_the_loop_going_for_the_report():
    record = Recorder()
    df = RDataFrame(100, step=10).Define("i", record)
    df.Filter("i < 50", "half")
    assert df.Range(3).Count().GetValue() == 3 and len(record.sizes) == 10


def test_a_failed_loop_fails_what_it_was_computing_and_nothing_after(xyn_path):
    with RDataFrame("tree", xyn_path) as df:
        broken, fine = df.Define("z", lambda x: 1.0).Sum("z"), df.Count()
        with pytest.raises(ValueError, match="gave a float"):
            fine.GetValue()
        with pytest.raises(ValueError, match="gave a float"):
            broken.GetValue()
        assert df.Count().GetValue() == 1000


def test_implicit_mt_sets_how_many_workers_frames_use():
    try:
        EnableImplicitMT(3)
        assert (IsImplicitMTEnabled(), GetThreadPoolSize(), RDataFrame(5).GetNSlots()) == (
            True,
            3,
            3,
        )
        EnableImplicitMT()
        assert GetThreadPoolSize() >= 1
    finally:
        DisableImplicitMT()
    assert (IsImplicitMTEnabled(), RDataFrame(5).GetNSlots()) == (False, 1)
    assert xrdroot.EnableImplicitMT is EnableImplicitMT


def _analysis(df: RDataFrame) -> list:
    cut = df.Filter("x > 100", "cut").Define("z", "x * y / 7.")
    return [
        cut.Histo1D(("h", "", 50, -1.5e5, 0), "z", "x"),
        cut.Profile1D(("p", "", 20, 0, 1000), "x", "z"),
        cut.Sum("z"),
        cut.Mean("z"),
        cut.StdDev("z"),
        cut.Stats("z", "x"),
        cut.AsNumpy(["z", "n"]),
        cut.Graph("x", "z"),
        df.Report(),
        cut.Reduce(np.add, "z"),
        cut.Aggregate(_sum_batches, _add, "z", 0.0),
        df.Define("slot", "rdfslot_").Max("slot"),
        cut.Histo1D("z"),
    ]


def _sum_batches(acc: float, values: np.ndarray) -> float:
    return acc + float(values.sum())


def _add(a: float, b: float) -> float:
    return a + b


def _array(value):
    return (str(value.dtype), value.tolist())


def _rows(value):
    return (str(value.content.dtype), value.tolist())


def _mapping(value):
    return {key: _plain(item) for key, item in value.items()}


def _sequence(value):
    return [_plain(item) for item in value]


#: How each kind of result is made plain Python, to compare with ``==``.
PLAIN = [
    (
        lambda value: hasattr(value, "cuts"),
        lambda value: [(c.name, c.passed, c.all) for c in value],
    ),
    (
        lambda value: hasattr(value, "members"),
        lambda value: (value.classname, _plain(value.members)),
    ),
    (lambda value: hasattr(value, "GetN"), lambda value: _plain(vars(value))),
    (lambda value: isinstance(value, dict), _mapping),
    (lambda value: isinstance(value, (list, tuple)), _sequence),
    (lambda value: isinstance(value, np.ndarray), _array),
    (lambda value: isinstance(value, Jagged), _rows),
]


def _plain(value):
    """A result as plain Python, every number exactly as it is, to compare with ``==``."""
    for test, made in PLAIN:
        if test(value):
            return made(value)
    return value


def _bits(value) -> bytes:
    return pickle.dumps(_plain(value))


def test_several_workers_give_exactly_what_one_does(xyn_path):
    with RDataFrame("tree", xyn_path, step=97) as serial:
        one = [result.GetValue() for result in _analysis(serial)]
    with RDataFrame("tree", xyn_path, step=97, workers=3) as parallel:
        many = [result.GetValue() for result in _analysis(parallel)]
    assert [_bits(value) for value in one[:-2]] == [_bits(value) for value in many[:-2]]
    assert one[-2] == 0 and many[-2] in (0, 1, 2)  # the slot a worker is: which one it was
    assert _bits(one[-1]) == _bits(many[-1])


def test_a_tree_handed_over_is_sent_to_the_workers_from_its_file(xyn_path):
    with xrdroot.open_root(xyn_path) as f:
        values = [
            RDataFrame(f["tree"], step=300, workers=w).Sum("x * y").GetValue() for w in (1, 2)
        ]
    assert values[0] == values[1] == -sum(i * i for i in range(1000))


def test_a_chain_is_shared_across_workers_file_by_file():
    with RDataFrame("tree", FLAT, workers=2, step=2) as df:
        assert df.Take("I32").GetValue().tolist() == [-i for i in range(10)]
        assert df.Define("s", "Sum(SliI32)").Take("s").GetValue().tolist() == [
            -i * i for i in range(10)
        ]


def test_what_cannot_be_shared_across_workers_is_refused_by_name(xyn_path):
    with RDataFrame("tree", xyn_path, step=100, workers=2) as df:
        with pytest.raises(UnsupportedFeatureError, match="Range counts the entries"):
            df.Range(0, 500).Count().GetValue()
        with pytest.raises(UnsupportedFeatureError, match="Foreach is run for what it does"):
            df.Foreach(print, ["x"])
        with pytest.raises(UnsupportedFeatureError, match=r"could not be sent.*A lambda"):
            df.Define("z", lambda x: x).Sum("z").GetValue()
        assert df.Filter("x < 0").Count().GetValue() == 0  # and the frame goes on working


def test_one_task_is_run_here_however_many_workers_there_are():
    assert (
        RDataFrame(10, workers=4).Define("z", lambda rdfentry_: rdfentry_ * 2).Sum("z").GetValue()
        == 90
    )


class Slots:
    def __init__(self) -> None:
        self.given = [5]

    def get(self) -> int:
        return self.given.pop()


def test_a_worker_installs_its_plan_and_runs_tasks_from_it(xyn_path):
    with RDataFrame("tree", xyn_path) as df:
        count = df.Define("s", "rdfslot_").Sum("s")
        plan = loop.Plan(df._graph.source, [count._action], [], 100, 2)
        loop.install(pickle.dumps(plan), Slots())
        parts, counts = loop.work((0, 10, 0))
        assert (parts, counts) == ([[50]], [])
        assert pickle.loads(pickle.dumps(plan)).workers == 1


def test_errors_in_workers_come_back_as_themselves(xyn_path):
    with RDataFrame("tree", xyn_path, step=100, workers=2) as df:
        with pytest.raises(ValueError, match=r"gives a collection per entry"):
            df.Define("v", "Range(n)").Filter("v > 1").Count().GetValue()
