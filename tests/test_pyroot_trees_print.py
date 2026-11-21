"""``Print``, ``Show``, ``Scan`` and ``Draw``: what ROOT prints and returns, to the character.

The layouts are ``TTree::Print``'s and ``TBranch::Print``'s ``printf``
strings - ``*Br%5d :%-9s : %-54s``, ``*Entries :%9lld : Total  Size=%11lld
bytes  File Size  = %10lld *`` and the rest - every line 78 characters, a
long title broken at a colon onto a line starting ``*         |``, as ROOT's
docs show for ``staff.C``'s one branch of nine leaves. ``Show`` is
``TTree::Show``'s, down to the line it breaks after an array's first value.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from xrdroot.pyroot.trees import TTree, hooks
from xrdroot.pyroot.trees.layout import BranchInfo
from xrdroot.pyroot.trees.printing import branch_lines

STAFF = "Category/I:Flag:Age:Service:Children:Grade:Step:Hrweek:Cost"


def _tree(entries: int = 5) -> TTree:
    x, n, a = np.zeros(1), np.zeros(1, "i"), np.zeros(25, "f")
    t = TTree("T", "a tree to print")
    t.Branch("x", x, "x/D")
    t.Branch("n", n, "n/I")
    t.Branch("a", a, "a[n]/F")
    for i in range(entries):
        x[0], n[0], a[:] = i * 1.25, min(i * 6, 25), np.arange(25) + i
        t.Fill()
    return t


def _branch(number: int, name: str, title: str) -> list[str]:
    """The four lines ROOT prints for a branch of a tree not yet written, as patterns."""
    return [
        re.escape(f"*Br{number:5d} :{name:<9} : {title}".ljust(77) + "*"),
        r"\*Entries :        5 : Total  Size= +\d+ bytes  One basket in memory    \*",
        re.escape("*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *"),
        re.escape("*" + "." * 76 + "*"),
    ]


def test_print_lays_the_tree_and_every_branch_out_in_roots_table(capsys):
    _tree().Print()
    lines = capsys.readouterr().out.splitlines()
    patterns = [
        re.escape("*" * 78),
        re.escape("*Tree    :T         : a tree to print".ljust(77) + "*"),
        r"\*Entries :        5 : Total = +\d+ bytes  File  Size =          0 \*",
        re.escape("*        :          : Tree compression factor =   1.00" + " " * 23 + "*"),
        re.escape("*" * 78),
        *_branch(0, "x", "x/D"),
        *_branch(1, "n", "n/I"),
        *_branch(2, "a", "a[n]/F"),
    ]
    assert len(lines) == len(patterns)
    assert [bool(re.fullmatch(p, line)) for p, line in zip(patterns, lines, strict=False)] == [
        True
    ] * len(lines)
    assert {len(line) for line in lines} == {78}


def test_a_tree_never_written_adds_its_own_record_to_the_header(capsys):
    """ROOT streams the tree object itself and adds no baskets until it is written."""
    from xrdroot.pyroot.trees.memory import TREE_RECORD

    TTree("T", "a tree to print").Print()
    lines = capsys.readouterr().out.splitlines()
    total = TREE_RECORD + len("T") + len("a tree to print")
    assert lines[2] == f"*Entries :        0 : Total = {total:15d} bytes  File  Size = {0:10d} *"
    assert len(lines[2]) == 78


def test_a_branch_whose_record_is_not_known_leaves_the_total_to_the_baskets(capsys):
    """A branch of a variable number of values is not one the record can be counted for."""
    t = _tree()
    t.Print()
    lines = capsys.readouterr().out.splitlines()
    baskets = sum(branch.tot_bytes for branch in t._layout())
    assert lines[2] == f"*Entries :        5 : Total = {baskets:15d} bytes  File  Size = {0:10d} *"


def test_a_tree_written_prints_its_baskets_on_file(capsys, tmp_path):
    import xrdroot

    t = _tree()
    with xrdroot.create(str(tmp_path / "p.root")) as f:
        t.SetDirectory(f)
        t.Write()
    t.Print()
    lines = capsys.readouterr().out.splitlines()
    assert re.fullmatch(
        r"\*Entries :        5 : Total  Size= +\d+ bytes  File Size  = +\d+ \*", lines[6]
    )
    assert re.fullmatch(
        r"\*Baskets :        1 : Basket Size= +32000 bytes  Compression= +\d\.\d\d     \*", lines[7]
    )


def test_a_tree_read_with_its_key_counts_the_key_in_its_totals(capsys, tmp_path):
    import xrdroot
    from xrdroot.pyroot.trees import wrap

    t = _tree()
    with xrdroot.create(str(tmp_path / "k.root")) as f:
        t.SetDirectory(f)
        t.Write()
    with xrdroot.open_root(str(tmp_path / "k.root")) as f:
        key = f.key("T")
        wrap(f["T"]).Print()
        without = capsys.readouterr().out.splitlines()[2]
        wrap(f["T"], "TTree", key).Print()
        with_key = capsys.readouterr().out.splitlines()[2]
    assert without != with_key
    assert with_key.endswith(f"File  Size = {int(without.split()[-2]) + key.nbytes:10d} *")


def test_print_with_a_name_prints_only_the_branches_it_matches(capsys):
    _tree().Print("n*")
    out = capsys.readouterr().out
    assert "*Br    1 :n " in out and ":x " not in out
    _tree().Print("all")
    assert capsys.readouterr().out.count("*Br ") == 3


def test_a_long_title_is_broken_at_a_colon_as_tbranch_print_breaks_it():
    info = BranchInfo("staff", STAFF, "", [None, None], 3354, 154237, 32316, 3, 32000)
    lines = branch_lines(info, 0)
    assert lines[0] == (
        "*Br    0 :staff     : Category/I:Flag:Age:Service:Children:Grade:Step:Hrweek:*\n"
        "*         | Cost                                                             *"
    )
    assert (
        lines[1] == "*Entries :     3354 : Total  Size=     154237 bytes  File Size  =      32316 *"
    )
    assert (
        lines[2] == "*Baskets :        3 : Basket Size=      32000 bytes  Compression=   4.77     *"
    )


def test_a_title_that_is_the_name_prints_the_leaf_type_instead(capsys):
    from xrdroot.pyroot.trees import TNtuple

    ntuple = TNtuple("ntuple", "Demo ntuple", "px:py")
    ntuple.Fill(1, 2)
    ntuple.GetBranch("px").Print()
    assert (
        capsys.readouterr().out.splitlines()[0] == "*Br    0 :px        : Float_t".ljust(77) + "*"
    )
    empty = TTree("empty", "")
    empty.Print()
    assert capsys.readouterr().out.count("\n") == 5


def test_show_prints_one_entry_breaking_an_array_after_its_first_value(capsys):
    t = _tree()
    t.Show(4)
    assert capsys.readouterr().out == (
        "======> EVENT:4\n"
        " x               = 5\n"
        " n               = 24\n"
        " a               = 4, \n"
        "                  5, 6, 7, 8, 9, \n"
        "                  10, 11, 12, 13, 14, \n"
        "                  15, 16, 17, 18, 19, \n"
        "                  20, 21, 22, 23\n"
    )
    t.GetEntry(0)
    t.Show()
    assert (
        capsys.readouterr().out == "======> EVENT:0\n x               = 0\n n               = 0\n"
    )
    fresh = _tree()
    fresh.SetBranchStatus("a", 0)
    fresh.Show()
    assert capsys.readouterr().out.splitlines()[0] == "======> EVENT:0"


def test_show_prints_integers_ten_to_a_line_and_text_whole(capsys):
    from xrdroot.pyroot.stl import std

    hits, name = np.zeros(12, "i"), std.string("")
    t = TTree("t", "")
    t.Branch("hits", hits, "hits[12]/I")
    t.Branch("name", name)
    hits[:] = np.arange(12)
    name.assign("a name")
    t.Fill()
    t.Show(0)
    assert capsys.readouterr().out.splitlines()[1:] == [
        " hits            = 0, ",
        "                  1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ",
        "                  11",
        " name            = a name",
    ]


def test_scan_prints_roots_table_and_returns_how_many_it_selected(capsys):
    t = _tree()
    assert t.Scan("x:n", "n > 6") == 3
    out = capsys.readouterr().out
    assert "*    Row   *         x *         n *" in out and "==> 3 selected entries" in out
    assert t.Scan("x", "", "colsize=4 precision=2", 2, 1) == 2
    assert "*        1 *  1.2 *" in capsys.readouterr().out


def test_draw_returns_the_number_selected_and_leaves_what_it_filled_in_the_directory(monkeypatch):
    drawn, objects = [], {}
    monkeypatch.setattr(hooks, "draw", lambda obj, option: drawn.append((obj, option)))
    monkeypatch.setattr(hooks, "registry", lambda: objects)
    monkeypatch.setattr(hooks, "wrap", lambda obj: ("wrapped", obj))
    t = _tree(10)
    assert t.Draw("x", "n > 12") == 7
    assert (drawn[-1][0][0], drawn[-1][1], "htemp" in objects) == ("wrapped", "", True)
    assert (t.Draw("x>>hx(10, 0, 20)", "", "goff"), len(drawn), objects["hx"].name) == (10, 1, "hx")
    assert (t.Draw("x>>+hx", "", "goff"), objects["hx"].entries) == (10, 20)


def test_draw_takes_aliases_weights_ranges_and_estimates(monkeypatch):
    objects = {}
    monkeypatch.setattr(hooks, "registry", lambda: objects)
    t = _tree(10)
    t.SetAlias("twice", "2 * x")
    assert (t.GetAlias("twice"), t.GetAlias("nope")) == ("2 * x", None)
    assert t.Draw("twice", "twice > 10", "goff", 5, 3) == 3
    t.SetWeight(2.0)
    assert (t.GetWeight(), t.Draw("x", "", "goff")) == (2.0, 10)
    assert objects["htemp"].sum(flow=True) == pytest.approx(20)
    t.SetEstimate(50)
    t.SetEstimate(-1)
    assert (t.GetEstimate(), t.SetScanField(10)) == (50, None)


def test_the_attributes_a_tree_draws_with_are_kept_for_graphics():
    t = _tree()
    t.SetMarkerColor(2)
    t.SetFillStyle(3001)
    assert (t.GetMarkerColor(), t.GetFillStyle(), t.GetLineWidth()) == (2, 3001, 1)
    t.SetNameTitle("renamed", "and retitled")
    assert (t.GetName(), t.GetTitle()) == ("renamed", "and retitled")


def test_print_and_show_of_a_file_root_wrote_are_what_root_6_40_prints(capsys):
    """Every line but the tree's own total, which counts ROOT's in-memory record."""
    import pathlib

    import xrdroot
    from xrdroot.pyroot.trees import wrap

    data = pathlib.Path(__file__).parent / "data"
    expected = (data / "small-flat-tree.root-6.40.txt").read_text().splitlines()
    with xrdroot.open_root(str(data / "small-flat-tree.root")) as f:
        tree = wrap(f["tree"], "TTree", f.key("tree"))
        tree.Print()
        tree.Show(3)
    printed = capsys.readouterr().out.splitlines()
    assert printed[:2] + printed[3:] == expected[:2] + expected[3:]
    assert printed[2].endswith("File  Size =      10452 *")


def test_a_member_of_one_string_per_object_is_printed_as_the_names_it_holds():
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(name="N.fName", text=False, vector=False, classname="")
    lines = show_lines(0, [(leaf, np.array(["N_0_0", "N_0_1"]))])
    assert lines[-1] == " N.fName         = N_0_0, N_0_1"


def test_a_member_of_whole_objects_is_printed_as_the_class_or_the_pointer_it_is():
    """ROOT prints the type the member was declared with: ``TRef`` for a ``TRef``.

    ``roottest/root/io/tree``'s ``.ref`` files say ``TRef`` where this test used to pin
    ``->``: a ``TRef`` writes neither its class nor a pointer, so what it is has to be read
    from the members it does write.
    """
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(name="r1.fRef", text=False, vector=False, classname="")
    lines = show_lines(0, [(leaf, [{"TObject": {"fUniqueID": 1}, "fPID": 0}])])
    assert lines[-1] == " r1.fRef         = TRef"
    array = [{"TObject": {"fUniqueID": 1}, "fLowerBound": 0, "fPID": 0, "fUIDs": [1]}]
    assert show_lines(0, [(leaf, array)])[-1] == " r1.fRef         = TRefArray"
    named = [{"classname": "TNamed", "TObject": {"fUniqueID": 1}}]
    assert show_lines(0, [(leaf, named)])[-1] == " r1.fRef         = TNamed"
    # A record that names neither a class nor a reference is the pointer ROOT would show.
    lines = show_lines(0, [(leaf, [{"TObject": {"fUniqueID": 1}}])])
    assert lines[-1] == " r1.fRef         = ->"


def test_a_leaf_that_is_text_prints_the_text_it_holds():
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(name="N.fType", text=True, vector=False, classname="")
    lines = show_lines(0, [(leaf, np.array([1, 2]))])
    assert lines[-1] == " N.fType         = [1 2]"


def test_a_member_of_a_record_that_is_one_string_per_object_prints_them_joined():
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(name="r1", branch="r1", text=False, vector=False, classname="")
    lines = show_lines(0, [(leaf, {"fName": np.array(["N_0_0", "N_0_1"])})])
    assert lines[-1] == " fName           = N_0_0, N_0_1"


def test_a_collection_branch_prints_how_many_objects_it_holds_and_nothing_else():
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(name="N_", branch="N", text=False, vector=False, classname="")
    record = {"TObject": {"fUniqueID": 1}, "fName": np.array(["N_0_0", "N_0_1"])}
    assert show_lines(0, [(leaf, record)]) == ["======> EVENT:0", " N               = 2"]
    # The first member that is one value per object gives the count, whatever it holds; one
    # object's own members - a base's - say nothing about it.
    other = SimpleNamespace(name="r1_", branch="r1", text=False, vector=False, classname="")
    counts = [show_lines(0, [(other, {"fName": np.array(["a"])})])[-1]]
    counts.append(show_lines(0, [(other, {"fRef": np.zeros((3, 2), "f")})])[-1])
    counts.append(show_lines(0, [(other, {"f": "text"})])[-1])
    counts.append(show_lines(0, [(other, {"e": 2})])[-1])
    assert counts == [
        " r1              = 1",
        " r1              = 3",
        " r1              = 0",
        " r1              = 1",
    ]


def test_a_collections_members_are_printed_without_a_line_break_between_them():
    """``TBranchElement::PrintValue`` prints a split member value after value."""
    from xrdroot.pyroot.trees.printing import show_lines

    leaf = SimpleNamespace(
        name="N.fUniqueID",
        branch="N.fUniqueID",
        text=False,
        vector=False,
        classname="TLeafElement",
    )
    lines = show_lines(0, [(leaf, np.arange(9, 0, -1))])
    assert lines == ["======> EVENT:0", " N.fUniqueID     = 9, 8, 7, 6, 5, 4, 3, 2, 1"]
