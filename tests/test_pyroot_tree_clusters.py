"""``FlushBaskets`` ending clusters, an empty tree written by ``WriteObject``, chains as friends."""

from __future__ import annotations

from array import array

import xrdroot.pyroot as ROOT
from xrdroot.file import open_root


def _filled(n: int, flush_every: int) -> ROOT.TTree:
    t = ROOT.TTree("t", "clustered")
    x = array("i", [0])
    t.Branch("x", x, "x/I")
    for i in range(n):
        x[0] = i
        t.Fill()
        if (i + 1) % flush_every == 0:
            t.FlushBaskets()
    t.FlushBaskets()  # nothing new since the last: no empty cluster
    return t


def test_flush_baskets_ends_a_cluster_where_it_is_called(tmp_path):
    path = str(tmp_path / "clusters.root")
    f = ROOT.TFile(path, "RECREATE")
    _filled(10, 4).Write()
    f.Close()
    with open_root(path) as opened:
        ranges = opened["t"].clustering.ranges()
        assert [stop - start for start, stop in ranges] == [4, 4, 2]
        assert list(opened["t"].arrays()["x"]) == list(range(10))


def test_a_tree_of_no_branches_is_written_when_write_object_asks(tmp_path):
    path = str(tmp_path / "empty.root")
    with ROOT.TFile(path, "RECREATE") as f:
        f.WriteObject(ROOT.TTree("e", "empty"), "e")
        f.WriteObject(_filled(3, 10), "full")
    df = ROOT.RDataFrame("e", path)
    assert df.Count().GetValue() == 0
    assert ROOT.RDataFrame("full", path).Count().GetValue() == 3


def test_reset_and_a_circular_tree_forget_the_clusters_with_the_entries(tmp_path):
    t = _filled(4, 2)  # the last flush finds nothing new to end a cluster with
    t.Reset()
    assert t.GetEntries() == 0
    circular = ROOT.TTree("c", "")
    x = array("i", [0])
    circular.Branch("x", x, "x/I")
    circular.SetCircular(4)
    for i in range(6):
        x[0] = i
        circular.Fill()
        circular.FlushBaskets()
    assert circular.GetEntries() < 6


def test_a_chain_of_trees_of_names_of_their_own_is_a_friend_read_as_a_patchwork(tmp_path):
    for name, tree in (("a.root", "ta"), ("b.root", "tb")):
        f = ROOT.TFile(str(tmp_path / name), "RECREATE")
        t = ROOT.TTree(tree, "")
        y = array("i", [0])
        t.Branch("y", y, "y/I")
        for i in range(2):
            y[0] = i
            t.Fill()
        t.Write()
        f.Close()
    f = ROOT.TFile(str(tmp_path / "main.root"), "RECREATE")
    _filled(4, 10).Write()
    f.Close()
    main = ROOT.TChain("t")
    main.Add(str(tmp_path / "main.root"))
    friend = ROOT.TChain()
    friend.Add(str(tmp_path / "a.root") + "?#ta")
    friend.Add(str(tmp_path / "b.root") + "?#tb")
    main.AddFriend(friend, "fr")
    assert ROOT.RDataFrame(main).Count().GetValue() == 4
