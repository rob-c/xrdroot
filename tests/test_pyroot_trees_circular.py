"""``TTree::SetCircular``: the newest entries kept, and the bytes ROOT counts for them.

Every number here is ROOT 6.40's own, printed by the tutorial
``io/tree/tree114_circular.C`` or by ``TBranch::GetTotalSize`` and
``TTree::Print`` for trees filled the same way.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.pyroot.trees import TTree

#: What ``tree114_circular.C`` prints under ROOT 6.40.
TUTORIAL = """\
******************************************************************************
*Tree    :T         : test circular buffers                                  *
*Entries :    18977 : Total =          420932 bytes  File  Size =          0 *
*        :          : Tree compression factor =   1.00                       *
******************************************************************************
*Br    0 :px        : px/F                                                   *
*Entries :    18977 : Total  Size=      76529 bytes  One basket in memory    *
*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    1 :py        : px/F                                                   *
*Entries :    18977 : Total  Size=      76529 bytes  One basket in memory    *
*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    2 :pz        : px/F                                                   *
*Entries :    18977 : Total  Size=      76529 bytes  One basket in memory    *
*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    3 :random    : random/D                                               *
*Entries :    18977 : Total  Size=     152469 bytes  One basket in memory    *
*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    4 :i         : i/s                                                    *
*Entries :    18977 : Total  Size=      38565 bytes  One basket in memory    *
*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
"""


def test_a_circular_tree_keeps_its_newest_entries_and_prints_roots_bytes(capsys):
    px, py, pz = (np.zeros(1, np.float32) for _ in range(3))
    random, i = np.zeros(1), np.zeros(1, np.uint16)
    t = TTree("T", "test circular buffers")
    t.Branch("px", px, "px/F")
    t.Branch("py", py, "px/F")
    t.Branch("pz", pz, "px/F")
    t.Branch("random", random, "random/D")
    t.Branch("i", i, "i/s")
    t.SetCircular(20000)
    for entry in range(65000):
        i[0] = entry
        t.Fill()
    t.Print()
    assert capsys.readouterr().out == TUTORIAL
    # 20000 is passed by one, 18000 are kept, and so on 22 times more.
    assert t.GetEntries() == 18977
    t.GetEntry(0)
    assert int(t.i) == 65000 - 18977


def _four(fills: int, circular: int, name: str = "T") -> TTree:
    x = np.zeros(1, np.float32)
    t = TTree(name, "t")
    for branch in ("a", "bb", "ccc", "dddd"):
        t.Branch(branch, x, f"{branch}/F")
    t.SetCircular(circular)
    for _ in range(fills):
        t.Fill()
    return t


def _totals(t: TTree) -> list[int]:
    return [branch.tot_bytes + branch.streamed for branch in t._layout()]


@pytest.mark.parametrize(
    ("fills", "name", "branches", "tree"),
    [
        (0, "T", [469, 473, 477, 481], 2204),
        (1, "T", [619, 625, 631, 637], 2792),
        (10, "T", [635, 641, 647, 653], 2856),
        (1, "Tree", [625, 631, 637, 643], 2819),
    ],
)
def test_the_bytes_of_a_circular_tree_are_roots_for_any_names_and_entries(
    fills, name, branches, tree
):
    t = _four(fills, 5, name)
    layout = t._layout()
    assert _totals(t) == branches
    assert sum(branch.tot_bytes for branch in layout) + t._record(layout) == tree


def test_a_circular_tree_of_arrays_and_integers_counts_every_value():
    k, arr = np.zeros(1, np.int32), np.zeros(3, np.float32)
    t = TTree("Tree5", "a much longer title")
    t.Branch("px", k, "px/I")
    t.Branch("arr", arr, "arr[3]/F")
    t.SetCircular(50)
    for _ in range(100):
        t.Fill()
    assert t.GetEntries() == 46 and _totals(t) == [813, 1193]


def test_a_circular_tree_of_other_branches_keeps_the_writers_counts(capsys):
    n, v = np.zeros(1, np.int32), np.zeros(4)
    t = TTree("T", "t")
    t.Branch("n", n, "n/I")
    t.Branch("v", v, "v[n]/D")
    t.Branch("pair", np.zeros(2, np.float32), "a/F:b/F")
    t.SetCircular(3)
    for entry in range(5):
        n[0] = entry % 4
        t.Fill()
    assert t.GetEntries() == 3 and t._record(t._layout()) is None
    t.SetCircular(0)
    for _ in range(5):
        t.Fill()
    assert t.GetEntries() == 8 and t._record(t._layout()) is None
