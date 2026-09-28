"""``TTree::Branch`` and ``Fill``: a tree filled through addresses, as ``tree1.C`` fills one.

Every test fills a tree the way ROOT's tutorials do - bind an address to a
branch, change what is at the address, ``Fill`` - and reads the file back
with :func:`xrdroot.open_root`, so what is checked is what was written.
"""

from __future__ import annotations

import array
import ctypes

import numpy as np
import pytest

import xrdroot
from xrdroot.pyroot.stl import std
from xrdroot.pyroot.trees import TNtuple, TNtupleD, TTree, hooks


class Cell:
    """What the macro translator makes of ``Float_t x;``: a value to read and set."""

    def __init__(self, value, dtype=None):
        self.value = value
        if dtype is not None:
            self.dtype = dtype


def _plain(values):
    """A column as plain lists, whatever it was read back as."""
    return values.tolist() if hasattr(values, "tolist") else list(values)


def _written(tmp_path, tree, name="t"):
    path = tmp_path / "out.root"
    with xrdroot.create(str(path)) as f:
        tree.SetDirectory(f)
        assert tree.Write() > 0
    with xrdroot.open_root(str(path)) as f:
        return f[name].arrays()


def test_every_kind_of_address_is_read_at_the_moment_fill_is_called(tmp_path):
    px, py, ev = ctypes.c_float(), np.zeros(1, "f"), array.array("i", [0])
    n, arr, vec, text = np.zeros(1, "i"), np.zeros(4, "d"), std.vector["float"](), std.string()
    label, cell = bytearray(16), Cell(0.0)
    t = TTree("t", "a simple tree")
    for name, where, leaflist in [
        ("px", px, "px/F"),
        ("py", py, "py/F"),
        ("ev", ev, "ev/I"),
        ("n", n, "n/I"),
        ("arr", arr, "arr[n]/D"),
        ("label", label, "label/C"),
        ("w", cell, "w/D"),
    ]:
        t.Branch(name, where, leaflist)
    t.Branch("vec", vec)
    t.Branch("text", text)
    for i in range(6):
        px.value, py[0], ev[0], n[0], cell.value = i * 1.5, -i, i, i % 3, i / 4
        arr[: i % 3] = np.arange(i % 3) + i
        vec.assign(range(i % 2 + 1))
        text.assign(f"e{i}")
        label[:3] = f"l{i}\0".encode()[:3]
        assert t.Fill() > 0
    assert (t.GetEntries(), len(t), bool(TTree("empty", ""))) == (6, 6, True)
    got = _written(tmp_path, t)
    assert {name: _plain(values) for name, values in got.items()} == {
        "px": [0, 1.5, 3, 4.5, 6, 7.5],
        "py": [0, -1, -2, -3, -4, -5],
        "ev": [0, 1, 2, 3, 4, 5],
        "n": [0, 1, 2, 0, 1, 2],
        "arr": [[], [1], [2, 3], [], [4], [5, 6]],
        "nvec": [1, 2, 1, 2, 1, 2],
        "vec": [[0], [0, 1]] * 3,
        "label": ["l0", "l1", "l2", "l3", "l4", "l5"],
        "w": [0, 0.25, 0.5, 0.75, 1.0, 1.25],
        "text": ["e0", "e1", "e2", "e3", "e4", "e5"],
    }
    assert got["py"].dtype == np.float32


def test_a_leaf_list_of_several_leaves_reads_each_member_of_what_it_was_given(tmp_path):
    class Point(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_int)]

    class Event:
        def __init__(self):
            self.e, self.t, self.hits = 0.0, 0.0, np.zeros(2, "i")

    record = np.zeros(1, dtype=[("a", "i4"), ("b", "f8")])
    flat, point, event, cells = np.zeros(3, "f"), Point(), Event(), Cell(0.0)
    t = TTree("t", "")
    t.Branch("rec", record, "a/I:b/D")
    t.Branch("xyz", flat, "u:v:z")
    t.Branch("pt", point, "x/F:y/I")
    t.Branch("ev", event, "e/D:t:hits[2]/I")
    holder = type("Holder", (), {"q": cells, "r": 0})()
    t.Branch("h", holder, "q/D:r/I")
    for i in range(3):
        record["a"], record["b"], flat[:] = i, i / 2, [i, 2 * i, 3 * i]
        point.x, point.y, event.e, event.t, event.hits[:] = i, -i, i + 0.5, i * 2.0, [i, i]
        cells.value, holder.r = i * 10.0, i + 1
        t.Fill()
    got = _written(tmp_path, t)
    assert got["a"].tolist() == [0, 1, 2] and got["b"].tolist() == [0, 0.5, 1.0]
    assert got["u"].tolist() == [0, 1, 2] and got["z"].tolist() == [0, 3, 6]
    assert got["z"].dtype == np.float32 and got["x"].tolist() == [0, 1, 2]
    assert got["y"].tolist() == [0, -1, -2] and got["t"].dtype == np.float64
    assert got["hits"].tolist() == [[0, 0], [1, 1], [2, 2]]
    assert got["q"].tolist() == [0, 10, 20] and got["r"].tolist() == [1, 2, 3]
    branch = t.GetBranch("rec")
    assert branch.GetTitle() == "a/I:b/D" and len(branch.GetListOfLeaves()) == 2


def test_a_branch_without_a_leaf_list_takes_its_type_from_its_address(tmp_path):
    t = TTree("t", "")
    t.Branch("d", np.zeros(1))
    t.Branch("i", ctypes.c_int(3))
    t.Branch("l", np.zeros(1, "l"))
    t.Branch("three", np.zeros(3, "h"))
    t.Branch("flag", Cell(True))
    t.Branch("count", Cell(2))
    t.Branch("x", Cell(0.5))
    t.Branch("typed", Cell(1, dtype="float32"))
    t.Fill()
    got = _written(tmp_path, t)
    kinds = {name: values.dtype.str[1:] for name, values in got.items()}
    assert kinds == {
        "d": "f8",
        "i": "i4",
        "l": "i8",
        "three": "i2",
        "flag": "b1",
        "count": "i4",
        "x": "f8",
        "typed": "f4",
    }
    assert got["three"].shape == (1, 3) and got["i"].tolist() == [3]


def test_what_cannot_be_read_in_place_is_refused_by_name():
    t = TTree("t", "")
    with pytest.raises(TypeError, match="a float, and a tree reads and fills only"):
        t.Branch("x", 1.5, "x/F")
    with pytest.raises(ValueError, match="not one contiguous, writable block"):
        t.Branch("x", np.zeros((3, 2))[:, 0], "x/F")
    with pytest.raises(TypeError, match="does not say its type"):
        t.Branch("x", Cell(object()))
    with pytest.raises(TypeError, match="does not say its type"):
        t.Branch("x", np.zeros(1, "c16"))
    with pytest.raises(TypeError, match="a dict, and a tree reads"):
        t.Branch("x", {}, "x/F:y/F")
    with pytest.raises(AttributeError, match="has no y"):
        t.Branch("x", type("P", (), {"x": 0.0})(), "x/F:y/F")


def test_a_leaf_list_that_is_not_one_is_refused_with_what_is_wrong_with_it():
    t = TTree("t", "")
    for leaflist, message in [
        ("x/Q", "not a type ROOT has a letter for"),
        ("x[/F", "is not a leaf"),
        ("x/F:x/F", "names one leaf twice"),
        ("x[3][n]/F", "every size after that is a number"),
        ("x[m]/F", "counted by 'm'"),
    ]:
        with pytest.raises(ValueError, match=message):
            t.Branch("x", np.zeros(4), leaflist)
    t.Branch("f", np.zeros(1), "f/D")
    with pytest.raises(ValueError, match="has to be an integer leaf"):
        t.Branch("y", np.zeros(4), "y[f]/D")
    t.Branch("n", np.zeros(1, "i"), "n/I")
    with pytest.raises(ValueError, match="fixed size inside its count"):
        t.Branch("y", np.zeros(4), "y[n][2]/D")
    with pytest.raises(ValueError, match="already has a branch called 'n'"):
        t.Branch("n", np.zeros(1, "i"), "n/I")
    with pytest.raises(ValueError, match="which this tree has already"):
        t.Branch("pair", np.zeros(2), "n/D:f/D")


def test_a_counter_that_says_more_than_its_array_holds_refuses_the_whole_entry():
    n, x = np.zeros(1, "i"), np.zeros(2, "f")
    t = TTree("t", "")
    t.Branch("n", n, "n/I")
    t.Branch("x", x, "x[n]/F")
    n[0] = 5
    with pytest.raises(ValueError, match="room for 2"):
        t.Fill()
    assert t.GetEntries() == 0
    n[0] = 1
    t.Fill()
    with pytest.raises(ValueError, match="already has 1 entries"):
        t.Branch("late", np.zeros(1), "late/D")


def test_a_tree_in_memory_is_written_where_set_directory_puts_it_and_not_before(tmp_path):
    x = np.zeros(1)
    t = TTree("t", "memory")
    t.Branch("x", x, "x/D")
    t.Fill()
    assert t.GetDirectory() is None and t.GetCurrentFile() is None
    with pytest.raises(ValueError, match="in memory, in no file"):
        t.Write()
    path = tmp_path / "two.root"
    with xrdroot.create(str(path)) as f:
        t.SetDirectory(f)
        assert t.AutoSave("SaveSelf") > 0
        x[0] = 2.0
        t.Fill()
        t.Write("renamed")
    with xrdroot.open_root(str(path)) as f:
        assert f["t"].arrays()["x"].tolist() == [0.0]
        assert f["renamed"].arrays()["x"].tolist() == [0.0, 2.0]


def test_a_tree_goes_into_the_directory_current_when_it_is_made(tmp_path, monkeypatch):
    kept = []

    class Directory:
        def __init__(self, out):
            self._xrd = out

        def Append(self, obj):
            kept.append(obj)

    path = tmp_path / "here.root"
    with xrdroot.create(str(path)) as f:
        monkeypatch.setattr(hooks, "directory", lambda: Directory(f))
        t = TTree("t", "")
        t.Branch("x", np.ones(1), "x/D")
        t.Fill()
        assert kept == [t]
        t.Write()
    with xrdroot.open_root(str(path)) as f:
        assert f["t"].arrays()["x"].tolist() == [1.0]


def test_reset_forgets_the_entries_and_keeps_the_branches_and_the_rest_do_nothing():
    x = np.zeros(1)
    t = TTree("t", "")
    t.Branch("x", x, "x/D")
    t.Fill()
    t.Reset()
    assert t.GetEntries() == 0 and t.GetBranch("x") is not None
    t.Fill()
    assert t.GetEntries() == 1
    for setting in (
        t.SetAutoSave,
        t.SetAutoFlush,
        t.SetMaxTreeSize,
        t.SetCircular,
        t.StartViewer,
    ):
        assert setting(10) is None
    t.SetBasketSize("x", 100)
    t.OptimizeBaskets()
    t.Refresh()
    assert t.SetCacheSize(10) == 0 and t.AddBranchToCache("x") == 0


def test_an_ntuple_is_filled_with_its_values_in_order_or_one_sequence_of_them(tmp_path):
    ntuple = TNtuple("ntuple", "data from ascii file", "x:y:z")
    ntuple.Fill(1, 2, 3)
    ntuple.Fill(np.array([4.0, 5.0, 6.0]))
    ntuple.Fill([7, 8, 9])
    assert ntuple.GetNvar() == 3 and ntuple.ClassName() == "TNtuple"
    assert ntuple.InheritsFrom("TTree") and not ntuple.InheritsFrom("TH1")
    with pytest.raises(ValueError, match="has 3 variables, and Fill was given 2 values"):
        ntuple.Fill(1, 2)
    ntuple.GetEntry(1)
    assert ntuple.GetArgs().tolist() == [4, 5, 6] and ntuple.GetArgs().dtype == np.float32
    got = _written(tmp_path, ntuple, "ntuple")
    assert got["y"].tolist() == [2, 5, 8] and got["x"].dtype == np.float32
    double = TNtupleD("d", "", "a:b")
    assert double.GetArgs().tolist() == [0, 0]
    double.Fill(0.1, 0.2)
    assert _written(tmp_path, double, "d")["a"].dtype == np.float64
    with xrdroot.open_root(str(tmp_path / "out.root")) as f:
        assert f.key("d").classname == "TNtupleD"  # written as what it is, as ROOT writes it


def test_an_ntuple_is_written_as_a_tntuple_not_a_ttree(tmp_path):
    ntuple = TNtuple("ntuple", "Demo ntuple", "px:py")
    ntuple.Fill(1, 2)
    _written(tmp_path, ntuple, "ntuple")
    with xrdroot.open_root(str(tmp_path / "out.root")) as f:
        assert f.key("ntuple").classname == "TNtuple"


def test_read_file_takes_its_branches_from_the_descriptor_or_the_first_line(tmp_path):
    text = tmp_path / "basic.dat"
    text.write_text("# a comment\n1 2 3\n\n4 5 6\n")
    t = TTree("t", "")
    assert t.ReadFile(str(text), "x:y/I:z") == 2
    assert _written(tmp_path, t)["y"].tolist() == [2, 5]
    headed = tmp_path / "headed.csv"
    headed.write_text("YEAR/I:T/F:NAME/C\n1775,-7.4,a\n1776,1.5,bc\n")
    t = TTree("t", "")
    assert t.ReadFile(str(headed), "", ",") == 2
    got = _written(tmp_path, t)
    assert got["YEAR"].tolist() == [1775, 1776] and got["NAME"] == ["a", "bc"]
    ntuple = TNtuple("n", "", "a:b")
    assert ntuple.ReadFile(str(text)) == 2
    assert _written(tmp_path, ntuple, "n")["b"].tolist() == [2, 5]


def test_a_tree_read_from_a_file_is_not_filled(tmp_path):
    from xrdroot.pyroot.trees import wrap

    t = TTree("t", "")
    t.Branch("x", np.zeros(1), "x/D")
    t.Fill()
    _written(tmp_path, t)
    with xrdroot.open_root(str(tmp_path / "out.root")) as f:
        read = wrap(f["t"])
        with pytest.raises(TypeError, match="was read from a file, and Fill adds"):
            read.Fill()
        with pytest.raises(TypeError, match="was read from a file"):
            wrap(f["t"], "TNtuple").Fill(1.0)
