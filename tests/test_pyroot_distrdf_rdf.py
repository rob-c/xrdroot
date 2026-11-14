"""What distributed RDataFrame needs of PyROOT's ``RDF``: options, dataset specs, tree facts.

``RSnapshotOptions``, ``RDF.Experimental.RDatasetSpec`` and the questions a
head node asks of a tree - its files, its trees' names, its friends - are
plain PyROOT that a local frame uses too.
"""

from __future__ import annotations

import warnings
from array import array

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.pyroot.rdf import treeutils
from xrdroot.pyroot.rdf.experimental import RDatasetSpec, RMetaData, RSample
from xrdroot.pyroot.rdf.snapshot import ESnapshotOutputFormat, RSnapshotOptions, keywords_of
from xrdroot.rdf.distributed import clusters_and_entries


def _write(path, tree="t", values=range(10), column="x"):
    with xrdroot.create(str(path)) as out:
        out[tree] = {column: np.asarray(list(values), dtype=np.int32)}
    return str(path)


def test_snapshot_options_say_what_xrdroots_snapshot_is_told():
    assert keywords_of(None) == {}
    options = ROOT.RDF.RSnapshotOptions()
    assert keywords_of(options) == {"mode": "RECREATE", "lazy": False, "auto_flush": 0,
                                    "rntuple": False}  # fmt: skip
    assert "lazy=False" in repr(options)
    options = RSnapshotOptions("update", 5, 1, 10, 99, True,
                               outputFormat=ESnapshotOutputFormat.kRNTuple)  # fmt: skip
    assert keywords_of(options) == {"mode": "UPDATE", "lazy": True, "auto_flush": 10,
                                    "rntuple": True, "compression": "zstd"}  # fmt: skip
    options.fCompressionLevel = 0
    assert keywords_of(options)["compression"] is None


def test_a_snapshot_takes_roots_options_and_a_tuple_of_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame = ROOT.RDataFrame(25).Define("x", "rdfentry_").Define("y", "2 * x")
    options = ROOT.RDF.RSnapshotOptions()
    options.fAutoFlush, options.fLazy = 10, True
    booked = frame.Snapshot("t", "a.root", ("x",), options)
    assert not booked.IsReady()
    assert list(booked.GetValue().GetColumnNames()) == ["x"]
    assert clusters_and_entries("t", "a.root") == ([0, 10, 20, 25], 25)
    every = frame.Snapshot("t", "b.root", "")
    assert sorted(every.GetColumnNames()) == ["x", "y"]
    written = frame.Snapshot("t", "c.root", "y", compression=None)
    assert list(written.GetColumnNames()) == ["y"]
    for each in (booked.GetValue(), every, written):
        each.close()


def test_the_rdf_namespace_has_the_models_and_namespaces_distributed_rdf_needs():
    assert ROOT.RDF.THnDModel("h", "h", 2, [2, 2], [0, 0], [1, 1])[2] == 2
    assert ROOT.RDF.THnSparseDModel() == ()
    assert ROOT.RDF.Experimental.RDatasetSpec is RDatasetSpec
    assert repr(ROOT.RDF.Experimental) == "<namespace ROOT::RDF::Experimental>"
    assert "Distributed" in dir(ROOT.RDF.Experimental)
    assert ROOT.RDF.Distributed.Dask.RDataFrame is not None
    with pytest.warns(FutureWarning, match="ROOT.RDF.Distributed"):
        assert ROOT.RDF.Experimental.Distributed.RunGraphs is not None
    with pytest.raises(AttributeError, match=r"RDF\.Experimental\.Vary"):
        ROOT.RDF.Experimental.Vary  # noqa: B018


def test_metadata_gives_each_value_as_asked_or_a_default():
    meta = RMetaData()
    meta.Add("lumi", "2.5")
    meta.Add("n", 3)
    assert (meta.GetD("lumi"), meta.GetI("n"), meta.GetS("n")) == (2.5, 3, "3")
    assert meta.GetI("missing", 7) == 7 and meta.Dump("n") == "3"
    with pytest.raises(RuntimeError, match="No key with name missing"):
        meta.GetS("missing")


def test_a_sample_names_its_tree_once_for_every_file_or_once_each():
    sample = RSample("s", "t", ["a.root", "b.root"])
    assert sample.GetTreeNames() == ["t", "t"] and sample.GetSampleName() == "s"
    assert isinstance(sample.GetMetaData(), RMetaData)
    with pytest.raises(ValueError, match="2 tree names for 3 files"):
        RSample("s", ["t", "u"], ["a", "b", "c"])


def test_a_dataset_spec_is_read_as_a_chain_of_its_samples(tmp_path):
    first = _write(tmp_path / "a.root", "t", range(5))
    second = _write(tmp_path / "b.root", "u", range(5, 10))
    spec = RDatasetSpec().AddSample(("one", "t", first)).AddSample(RSample("two", "u", second))
    assert spec.GetSampleNames() == ["one", "two"] and spec.GetTreeNames() == ["t", "u"]
    assert spec.GetFileNameGlobs() == [first, second] and len(spec.GetSamples()) == 2
    assert ROOT.RDataFrame(spec).Sum("x").GetValue() == 45
    spec.WithGlobalRange((2, 7))
    assert (spec.GetEntryRangeBegin(), spec.GetEntryRangeEnd()) == (2, 7)
    assert ROOT.RDataFrame(spec).Take["int"]("x").GetValue().tolist() == [2, 3, 4, 5, 6]
    spec.WithGlobalRange((8, None))
    assert ROOT.RDataFrame(spec).Count().GetValue() == 2


def test_a_dataset_spec_reads_its_friends_beside_it(tmp_path):
    main = _write(tmp_path / "a.root", "t", range(4))
    friend = _write(tmp_path / "f.root", "f", range(10, 14), "y")
    spec = RDatasetSpec().AddSample(("", "t", main)).WithGlobalFriends("f", friend, "fr")
    info = spec.GetFriendInfo()
    assert [tuple(pair) for pair in info.fFriendNames] == [("f", "fr")]
    assert ROOT.RDataFrame(spec).Sum("fr.y").GetValue() == 46
    unnamed = RDatasetSpec().WithGlobalFriends(["f"], [friend]).GetFriendInfo()
    assert unnamed.fFriendNames[0].second == "f"


def test_a_trees_files_names_and_friends_are_what_distributed_rdf_sends(tmp_path):
    main = _write(tmp_path / "a.root", "t", range(4))
    friend = _write(tmp_path / "f.root", "f", range(4), "y")
    chain = ROOT.TChain("t")
    chain.Add(main)
    friends = ROOT.TChain("f")
    friends.Add(friend)
    chain.AddFriend(friends, "fr")
    assert treeutils.file_names(chain) == [main] and treeutils.tree_names(chain) == ["t"]
    info = treeutils.friend_info(chain)
    assert bool(info) and repr(info) == "<RFriendInfo of ['fr']>"
    assert repr(info.fFriendNames[0]) == "('f', 'fr')"
    assert (info.fFriendFileNames, info.fFriendChainSubNames) == ([[friend]], [["f"]])
    assert treeutils.indexed_friend(chain) is None
    again = treeutils.chain_of(["t"], [main], info)
    assert ROOT.RDataFrame(again).Sum("fr.y").GetValue() == 6


def test_a_tree_from_a_file_is_named_by_its_file_and_its_friend_by_no_chain(tmp_path):
    main = _write(tmp_path / "a.root", "t", range(4))
    friend = _write(tmp_path / "f.root", "f", range(4), "y")
    with ROOT.TFile(main) as file, ROOT.TFile(friend) as other:
        tree = file.Get("t")
        tree.AddFriend(other.Get("f"))
        assert treeutils.file_names(tree) == [main] and treeutils.tree_names(tree) == ["t"]
        info = treeutils.friend_info(tree)
        assert info.fFriendFileNames == [[friend]] and info.fFriendChainSubNames == [[]]
        assert not treeutils.friend_info(other.Get("f"))


def test_a_tree_only_in_memory_has_no_file_to_send():
    tree = ROOT.TTree("m", "m")
    x = array("i", [0])
    tree.Branch("x", x, "x/I")
    tree.Fill()
    with pytest.raises(RuntimeError, match="in memory and in no file"):
        treeutils.file_names(tree)


def test_a_friend_indexed_by_build_index_is_named(tmp_path):
    main = _write(tmp_path / "a.root", "t", range(4))
    friend = _write(tmp_path / "f.root", "f", range(4))
    chain, other = ROOT.TChain("t"), ROOT.TChain("f")
    chain.Add(main)
    other.Add(friend)
    other.BuildIndex("x")
    chain.AddFriend(other)
    assert treeutils.indexed_friend(chain) == "f"


def test_a_results_value_is_presented_as_pyroot_presents_it():
    from xrdroot.pyroot.rdf.frame import presented

    histogram = xrdroot.Histogram.book("h", (2, 0, 1))
    assert isinstance(presented(histogram), ROOT.TH1D)
    assert presented(3) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert isinstance(presented(histogram, style=False), ROOT.TH1D)
