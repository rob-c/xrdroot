"""Distributed RDataFrame's graph, head nodes and ranges, run on a backend in this process.

The graph a user builds is recorded as nodes, pruned of what nobody holds,
cut into ranges of entries - evenly for empty entries, at cluster boundaries
for trees - and run range by range on xrdroot's own frame, the results of
the ranges added up. A backend that maps and reduces in this process runs
it all here, as ROOT's own unit tests do.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.file import open_root
from xrdroot.pyroot._distrdf import ComputationGraphGenerator, Node, Operation, Proxy, Ranges
from xrdroot.pyroot._distrdf._graph_cache import _ACTIONS_REGISTER, _RDF_REGISTER
from xrdroot.pyroot._distrdf.Backends import Base, Utils
from xrdroot.pyroot._distrdf.DataFrame import RDataFrame
from xrdroot.pyroot._distrdf.HeadNode import (
    EmptySourceHeadNode,
    RDatasetSpecHeadNode,
    RNTupleHeadNode,
    TreeHeadNode,
    get_headnode,
)
from xrdroot.pyroot._distrdf.PythonMergeables import SnapshotResult


class Here(Base.BaseBackend):
    """A backend that maps every range and reduces the results in this process."""

    def __init__(self, partitions: int = 2) -> None:
        self.partitions = partitions

    def ProcessAndMerge(self, ranges, mapper, reducer):
        results = [mapper(each) for each in ranges]
        while len(results) > 1:
            results.append(reducer(results.pop(0), results.pop(0)))
        return results.pop()

    def distribute_unique_paths(self, paths):
        """Nothing to send: everything is here."""

    def optimize_npartitions(self):
        return self.partitions

    def make_dataframe(self, *args, **kwargs):
        return RDataFrame(get_headnode(self, kwargs.pop("npartitions", None), *args))


@pytest.fixture(autouse=True)
def _fresh_caches():
    yield
    _RDF_REGISTER.clear()
    _ACTIONS_REGISTER.clear()


def _frame(*args, npartitions=None, backend=None):
    return RDataFrame(get_headnode(backend or Here(), npartitions, *args))


def _write(path, tree="t", count=10, flush_every=None, column="x"):
    step = flush_every or count
    with xrdroot.create(str(path)) as out:
        made = out.tree(tree, {column: float})
        for start in range(0, count, step):
            made.extend({column: np.arange(start, min(count, start + step), dtype=float)})
            made.flush()
    return str(path)


def test_an_operations_kind_says_how_it_is_run():
    assert isinstance(Operation.create_op("Count"), Operation.Action)
    assert isinstance(Operation.create_op("Snapshot"), Operation.InstantAction)
    assert isinstance(Operation.create_op("Histo1D", model=("h", "h", 2, 0, 1)), Operation.Histo)
    assert repr(Operation.create_op("Define", "x", "1")) == "<Transformation Define>"
    with pytest.raises(ValueError, match="'Take' is either invalid"):
        Operation.create_op("Take")
    with pytest.raises(ValueError, match="without a model"):
        Operation.create_op("Histo1D")


def test_results_of_empty_entries_add_up_across_ranges():
    df = _frame(100, npartitions=3)
    defined = df.Define("x", "rdfentry_ * 1.0")
    count, mean, histo = defined.Count(), defined.Mean("x"), defined.Histo1D(
        ("h", "h", 10, 0, 100), "x"
    )
    assert count.GetValue() == 100 and mean.GetValue() == 49.5
    assert histo.GetEntries() == 100 and isinstance(histo.GetValue(), ROOT.TH1D)
    assert defined.Sum("x").GetValue() == 4950
    assert list(df.GetColumnNames()) == [] and "Histo1D" in dir(df)
    assert repr(df) == "<distributed RDataFrame on Here>"


def test_an_instant_action_runs_at_once_and_a_lazy_one_when_asked(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = _frame(10, npartitions=2).Define("x", "rdfentry_")
    assert sorted(df.AsNumpy(["x"])["x"].tolist()) == list(range(10))
    lazy = df.AsNumpy(["x"], lazy=True)
    assert isinstance(lazy, Proxy.ResultPtrProxy) and len(lazy.GetValue()["x"]) == 10
    options = ROOT.RDF.RSnapshotOptions()
    options.fLazy = True
    later = df.Snapshot("t", "out.root", ["x"], options)
    assert isinstance(later, Proxy.ResultPtrProxy)
    assert later.GetValue().Count().GetValue() == 10
    assert sorted(f for f in os.listdir() if f.startswith("out")) == ["out_0.root", "out_1.root"]
    assert df.Snapshot("t", "now.root").Sum("x").GetValue() == 45
    options.fLazy = False
    assert df.Snapshot("t", "then.root", ["x"], options).Count().GetValue() == 10
    options.fMode = "UPDATE"
    with pytest.raises(ValueError, match="UPDATE mode"):
        df.Snapshot("t", "out.root", ["x"], options)


def test_a_snapshot_of_a_tree_writes_a_file_per_range_and_reads_them_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    files = [_write(f"in{i}.root", count=20, flush_every=5) for i in range(2)]
    snapped = _frame("t", files, npartitions=4).Snapshot("t", "snap.root", ("x",))
    assert sorted(f for f in os.listdir() if f.startswith("snap")) == [
        f"snap_{i}.root" for i in range(4)
    ]
    assert snapped.Count().GetValue() == 40
    assert len(_RDF_REGISTER) == 2


def test_the_graph_is_built_once_in_a_process_and_run_for_each_range():
    df = _frame(100, npartitions=4)
    assert df.Define("x", "1").Count().GetValue() == 100
    assert len(_RDF_REGISTER) == 1 and next(iter(_RDF_REGISTER.values())).GetNRuns() == 4


def test_a_pruned_graph_leaves_out_what_nobody_holds_and_what_is_known():
    df = _frame(10)
    kept = df.Define("x", "1")
    dropped = kept.Filter("x > 0").Count()
    known = kept.Count()
    known.proxied_node.value = 1
    head = df._headnode
    dropped = None  # noqa: F841
    graph = head._generate_graph_dict()
    assert [node.operation.name for node in list(graph.values())[1:]] == ["Define"]
    assert head._get_action_nodes() == []
    assert head.is_prunable() is False


def test_a_node_nobody_holds_with_no_parent_is_pruned_alone():
    node = Node.Node(None)
    node.has_user_references = False
    assert node.is_prunable()


def test_a_node_travels_as_its_operation_and_its_parents_id():
    head = get_headnode(None, None, 5)
    node = Node.Node(head.get_head, 1, Operation.create_op("Define", "x", "1"), head)
    state = node.__getstate__()
    again = Node.Node.__new__(Node.Node)
    again.__setstate__(state)
    assert again.parent_id == 0 and again.operation.args == ["x", "1"]


def test_a_proxy_answers_for_its_node_and_refuses_what_it_has_not():
    df = _frame(5)
    defined = df.Define("x", "1")
    with pytest.raises(AttributeError, match="'Define' object has no attribute 'nothing'"):
        defined.nothing  # noqa: B018
    with pytest.raises(AttributeError, match="'RDataFrame' object has no attribute 'nothing'"):
        df.nothing  # noqa: B018
    with pytest.raises(AttributeError):
        df.__nothing__  # noqa: B018
    assert defined.GetColumnType("x") == "int"
    result = defined.Count()
    assert result.GetValue() == 5 and result.__index__() == 5
    with pytest.raises(UnsupportedFeatureError, match="no Vary"):
        result.create_variations()
    with pytest.raises(AttributeError, match="no attribute 'GetKeys'"):
        Proxy.ResultMapProxy(result.proxied_node).GetKeys  # noqa: B018
    half = Proxy.ResultPtrProxy.__new__(Proxy.ResultPtrProxy)
    half.__del__()  # a proxy whose making failed holds no node to let go of


def test_variations_are_refused_in_a_task():
    graph = {0: Node.Node(None, 0), 1: Node.Node(None, 1, Operation.create_op("VariationsFor"))}
    graph[1].parent_id = 0
    with pytest.raises(UnsupportedFeatureError, match="no Vary"):
        ComputationGraphGenerator.generate_computation_graph(graph, object(), 0)


def test_a_snapshot_given_two_arguments_is_given_the_rest_to_be_lazy():
    made = ComputationGraphGenerator._create_lazy_op_if_needed(
        Operation.create_op("Snapshot", "t", "f.root"), 3
    )
    assert made.args[:3] == ["t", "f_3.root", ""] and made.args[3].fLazy


def test_trees_are_cut_at_cluster_boundaries_and_every_entry_is_read_once(tmp_path):
    files = [_write(tmp_path / f"f{i}.root", count=10, flush_every=5) for i in range(3)]
    for partitions in (1, 2, 5, 7):
        df = _frame("t", files, npartitions=partitions)
        assert df.Count().GetValue() == 30
        assert df.Sum("x").GetValue() == 135
    head = df._headnode
    assert isinstance(head, TreeHeadNode) and head.maintreename == "t"
    assert head.inputfiles == files and head.subtreenames == ["t"] * 3


def test_a_tree_handed_over_and_one_named_with_default_columns(tmp_path):
    path = _write(tmp_path / "a.root")
    head = get_headnode(None, 1, "t", path, ["x"])
    assert head.inputfiles == [path] and head.defaultbranches == ["x"]
    with ROOT.TFile(path) as file:
        tree = file.Get("t")
        handed = get_headnode(Here(), 2, tree, ["x"])
        assert handed.defaultbranches == ["x"] and handed.friendinfo is None
        assert RDataFrame(handed).Sum("x").GetValue() == 45


def test_a_chain_with_friends_sends_them_with_every_task(tmp_path):
    main = _write(tmp_path / "a.root", count=10, flush_every=5)
    friend = _write(tmp_path / "f.root", "f", count=10, column="y")
    chain, other = ROOT.TChain("t"), ROOT.TChain("f")
    chain.Add(main)
    other.Add(friend)
    chain.AddFriend(other, "fr")
    df = _frame(chain, npartitions=2)
    assert df._headnode.friendinfo is not None
    assert df.Sum("fr.y").GetValue() == 45


def test_a_friend_with_an_index_is_refused(tmp_path):
    chain, other = ROOT.TChain("t"), ROOT.TChain("f")
    chain.Add(_write(tmp_path / "a.root"))
    other.Add(_write(tmp_path / "f.root", "f"))
    other.BuildIndex("x")
    chain.AddFriend(other)
    with pytest.raises(ValueError, match="Friend tree 'f' has a TTreeIndex"):
        get_headnode(None, None, chain)


def test_arguments_no_frame_takes_are_refused():
    for wrong in [(), (10, "file.root")]:
        with pytest.raises(TypeError, match="not accepted by any RDataFrame constructor"):
            get_headnode(None, None, *wrong)


def test_empty_entries_are_cut_evenly_and_never_into_more_ranges_than_entries():
    ranges = Ranges.get_balanced_ranges(10, 4, None)
    assert [(r.start, r.end) for r in ranges] == [(0, 3), (3, 6), (6, 8), (8, 10)]
    head = get_headnode(Here(), 7, 5)
    assert isinstance(head, EmptySourceHeadNode)
    head.exec_id = None
    with pytest.warns(UserWarning, match="greater than number of entries 5"):
        assert len(head._build_ranges()) == 5
    with pytest.raises(RuntimeError, match="zero entries"):
        get_headnode(Here(), 2, 0)._build_ranges()


def test_the_backends_partitions_are_taken_unless_the_user_gave_some():
    head = get_headnode(Here(5), None, 20)
    head.npartitions = 5
    assert head.npartitions == 5
    given = get_headnode(Here(5), 2, 20)
    given.npartitions = 5
    assert given.npartitions == 2


def test_what_the_tasks_processed_is_checked_against_the_dataset(tmp_path):
    path = _write(tmp_path / "a.root")
    head = get_headnode(Here(), 1, "t", path)
    entries = Ranges.TaskTreeEntries(10, {f"{path}/t": 10})
    with pytest.raises(RuntimeError, match="returned no values"):
        head._handle_returned_values(Base.TaskResult(None, entries))
    wrong = Ranges.TaskTreeEntries(10, {"other/t": 10})
    with pytest.raises(RuntimeError, match="are not the same"):
        head._handle_returned_values(Base.TaskResult([], wrong))
    short = Ranges.TaskTreeEntries(5, {f"{path}/t": 10})
    with pytest.raises(RuntimeError, match="has 10 entries, but 5 were processed"):
        head._handle_returned_values(Base.TaskResult([], short))


def test_a_task_with_nothing_in_its_range_sends_nothing_back(tmp_path):
    path = _write(tmp_path / "a.root", count=10)  # one cluster
    df = _frame("t", [path], npartitions=3)
    assert df.Count().GetValue() == 10


def test_a_reducer_adds_a_tasks_results_to_one_that_had_none():
    entries = Ranges.TaskTreeEntries(3, {"a/t": 3})
    nothing = Base.TaskResult(None, Ranges.TaskTreeEntries(0, {"b/t": 0}))
    merged = Base.distrdf_reducer(nothing, Base.TaskResult(["kept"], entries))
    assert merged.mergeables == ["kept"] and merged.entries_in_trees.processed_entries == 3
    assert Base.merge_values(["kept"], None) == ["kept"]


def test_a_dataset_spec_is_cut_like_a_chain_of_its_samples(tmp_path):
    first = _write(tmp_path / "a.root", count=10, flush_every=5)
    second = _write(tmp_path / "b.root", "u", count=10, flush_every=5)
    spec = ROOT.RDF.Experimental.RDatasetSpec()
    spec.AddSample(("one", "t", first)).AddSample(("two", "u", second))
    df = _frame(spec, npartitions=3)
    assert isinstance(df._headnode, RDatasetSpecHeadNode)
    assert df.Sum("x").GetValue() == 90
    spec.WithGlobalRange((1, 5))
    with pytest.warns(UserWarning, match="global range is not supported"):
        get_headnode(None, None, spec)


def test_rntuples_are_cut_into_ranges_and_their_graph_built_for_each(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = xrdroot.RDataFrame(30).Define("x", "rdfentry_")
    frame.Snapshot("n", "a.root", ["x"], rntuple=True).close()
    df = _frame("n", ["a.root", "a.root"], npartitions=2)
    assert isinstance(df._headnode, RNTupleHeadNode)
    assert df.Count().GetValue() == 60
    with open_root("a.root") as file:
        with pytest.raises(UnsupportedFeatureError, match="by its name and the files"):
            get_headnode(None, None, file["n"])


def test_an_rntuples_task_with_nothing_in_its_range_sends_nothing_back(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    xrdroot.RDataFrame(5).Define("x", "1").Snapshot("n", "a.root", ["x"], rntuple=True).close()
    assert _frame("n", "a.root", npartitions=3).Count().GetValue() == 5


def test_the_snapshot_result_names_its_files_and_merges_their_lists():
    result = SnapshotResult("t", ["a_0.root"])
    result.Merge(SnapshotResult("t", ["a_1.root"]))
    assert repr(result) == "<SnapshotResult 't' in ['a_0.root', 'a_1.root']>"
    again = Utils.clone_action(SnapshotResult("t", ["plain_0"], _Booked()), 2)
    assert again.filenames == ["plain_2.root"]
    assert Utils.clone_action(SnapshotResult("t", ["a_0.root"], _Booked()), 1).filenames == [
        "a_1.root"
    ]


class _Booked:
    """A booked ``Snapshot`` that only knows to be booked again."""

    def __init__(self):
        from xrdroot.rdf.distributed import Window
        from xrdroot.rdf.sources import Empty

        frame = xrdroot.RDataFrame(Window(Empty(1), 0, 1)).Define("x", "1")
        self._inner = frame.Snapshot("t", os.devnull, ["x"], lazy=True)


def test_a_merged_value_is_presented_as_pyroot_presents_it():
    from xrdroot.pyroot._distrdf.PythonMergeables import MergeableValue

    class Sum:
        def __init__(self, value):
            self.value = value

        def Merge(self, other):
            self.value += other.value

        def GetValue(self):
            return self.value

    merged = MergeableValue(Sum(1), str)
    merged.Merge(MergeableValue(Sum(2), str))
    assert merged.GetValue() == "3" and repr(merged).startswith("<MergeableValue")


def test_the_mapper_declares_the_code_every_task_needs_first(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Base.setup_mapper(lambda: None, "int distrdf_declared_in_task = 4;")
    assert ROOT.distrdf_declared_in_task == 4
