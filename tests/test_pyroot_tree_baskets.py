"""A tree's baskets: the size ROOT gives a branch, and the clusters ``SetAutoFlush`` makes.

ROOT closes a basket when the next entry would not fit, keeps at least room for the
basket's own key (``100 + strlen(branch name)``, which is why ``SetBasketSize(100)`` on
``branch`` answers 106), and - with ``SetAutoFlush(n)`` - sends every part-filled basket
out at each ``n``-entry cluster end. ``kOnlyFlushAtCluster`` closes baskets *only* there,
which is ROOT's one-basket-per-cluster mode and the reason ROOT's own test wants 2.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def filled(count: int = 1000, only_at_cluster: bool = False, auto_flush: int = 0) -> tuple:
    """ROOT's own testTTreeCluster fixture: ``count`` doubles into one branch."""
    holder = ROOT.TFile("baskets.root", "RECREATE")
    tree = ROOT.TTree("t", "t")
    if only_at_cluster:
        tree.SetBit(ROOT.TTree.kOnlyFlushAtCluster)
    if auto_flush:
        tree.SetAutoFlush(auto_flush)
    data = np.zeros(1)
    branch = tree.Branch("branch", data)
    for entry in range(count):
        data[0] = entry
        tree.Fill()
        if entry == 100:
            branch.SetBasketSize(100)
    holder.Write()
    return holder, branch


def test_a_branchs_basket_keeps_room_for_its_own_key_as_roots_does() -> None:
    holder = ROOT.TFile("size.root", "RECREATE")
    tree = ROOT.TTree("t", "t")
    branch = tree.Branch("branch", np.zeros(1))
    expect((branch.GetBasketSize(), 32000))  # fmt: skip
    branch.SetBasketSize(100)
    expect((branch.GetBasketSize(), 106), (branch.GetBasketSize(), 100 + len("branch")))
    branch.SetBasketSize(50000)  # above the room the key needs: kept as given
    expect((branch.GetBasketSize(), 50000))  # fmt: skip
    holder.Write()
    holder.Close()


def test_a_tree_being_read_keeps_the_basket_size_the_file_was_written_with() -> None:
    holder, _ = filled(10)
    holder.Close()
    back = ROOT.TFile("baskets.root")
    read = back.Get("t").GetBranch("branch")
    read.SetBasketSize(100)  # nothing to size: a tree read answers what its file says
    expect((read.GetBasketSize(), 32000))  # fmt: skip
    back.Close()


def test_the_autoflush_closes_a_baskets_worth_at_every_cluster_end() -> None:
    holder, branch = filled(1000, auto_flush=500)
    expect((branch.GetWriteBasket() > 2, True))  # the size rule closes most of them
    holder.Close()


def test_one_basket_per_cluster_closes_nothing_on_size() -> None:
    """ROOT's own numbers: 1000 entries in two 500-entry clusters are 2 baskets of 106 bytes."""
    holder, branch = filled(1000, only_at_cluster=True, auto_flush=500)
    expect((branch.GetWriteBasket(), 2), (branch.GetBasketSize(), 106))
    holder.Close()

    back = ROOT.TFile("baskets.root")
    read = back.Get("t").GetBranch("branch")
    tree_back = back.Get("t")
    expect((read.GetWriteBasket(), 2), (read.GetBasketSize(), 106), (tree_back.GetEntries(), 1000))
    back.Close()


def test_the_autoflush_holds_for_a_column_of_rows_and_a_scalar_alike() -> None:
    """Two columns pick their baskets at their own rate, and the clusters end for both."""
    holder = ROOT.TFile("mixed.root", "RECREATE")
    tree = ROOT.TTree("t", "t")
    tree.SetAutoFlush(10)
    wide = tree.Branch("wide", np.zeros(8, dtype=np.float64))
    narrow = tree.Branch("narrow", np.zeros(1))
    for entry in range(30):
        wide.SetAddress(np.full(8, float(entry)))
        narrow.SetAddress(np.array([float(entry)]))
        tree.Fill()
    tree.FlushBaskets()
    holder.Write()
    expect((wide.GetWriteBasket() >= 3, True), (narrow.GetWriteBasket() >= 3, True))
    holder.Close()


def test_a_tree_being_read_is_told_nothing_of_a_writers_clusters() -> None:
    """A tree read from a file has no writer to tell, and the calls answer quietly."""
    holder, _ = filled(10, auto_flush=5)
    holder.Close()
    back = ROOT.TFile("baskets.root")
    read = back.Get("t")
    read.SetAutoFlush(500)  # nothing to set: the file's layout is the answer
    read.SetBit(ROOT.TTree.kOnlyFlushAtCluster)  # and this asks the same nothing
    read.SetBit(ROOT.TTree.kCircular)  # a bit that is not the writer's
    expect((read.TestBit(ROOT.TTree.kOnlyFlushAtCluster), True))
    read.SetBit(ROOT.TTree.kOnlyFlushAtCluster, False)
    back.Close()


def test_the_autoflush_of_zero_leaves_the_basket_size_to_decide() -> None:
    holder = ROOT.TFile("zero.root", "RECREATE")
    tree = ROOT.TTree("t", "t")
    tree.SetAutoFlush(0)  # ROOT's own choice: the baskets' size alone
    branch = tree.Branch("branch", np.zeros(1))
    for _ in range(20):
        tree.Fill()
    holder.Write()
    expect((branch.GetWriteBasket(), 1))  # fmt: skip
    holder.Close()
