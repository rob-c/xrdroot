"""``Print``, ``Show``, ``Scan`` and ``Draw``: what ROOT prints and returns, character for character.

The layouts are ``TTree::Print``'s and ``TBranch::Print``'s ``printf``
strings - ``*Br%5d :%-9s : %-54s``, ``*Entries :%9lld : Total  Size=%11lld
bytes  File Size  = %10lld *`` and the rest - every line 78 characters, a
long title broken at a colon onto a line starting ``*         |``, as ROOT's
docs show for ``staff.C``'s one branch of nine leaves. ``Show`` is
``TTree::Show``'s, down to the line it breaks after an array's first value.
"""

from __future__ import annotations

import re

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


def test_print_lays_the_tree_and_every_branch_out_in_roots_table(capsys):
    _tree().Print()
    lines = capsys.readouterr().out.splitlines()
    assert all(len(line) == 78 for line in lines)
    assert lines[0] == "*" * 78
    assert lines[1] == "*Tree    :T         : a tree to print".ljust(77) + "*"
    assert re.fullmatch(r"\*Entries :        5 : Total = +\d+ bytes  File  Size = +\d+ \*", lines[2])
    assert re.fullmatch(r"\*        :          : Tree compression factor = +\d+\.\d\d +\*", lines[3])
    assert lines[5] == "*Br    0 :x         : x/D".ljust(77) + "*"
    assert re.fullmatch(
        r"\*Entries :        5 : Total  Size= +\d+ bytes  One basket in memory    \*", lines[6]
    )
    assert lines[7] == "*Baskets :        0 : Basket Size=      32000 bytes  Compression=   1.00     *"
    assert lines[2].endswith("File  Size =          0 *") and "factor =   1.00 " in lines[3]
    assert lines[8] == "*" + "." * 76 + "*"
    assert lines[13].startswith("*Br    2 :a         : a[n]/F ")
    assert len(lines) == 5 + 3 * 4


def test_a_tree_written_prints_its_baskets_on_file(capsys, tmp_path):
    import xrdroot

    t = _tree()
    with xrdroot.create(str(tmp_path / "p.root")) as f:
        t.SetDirectory(f)
        t.Write()
    t.Print()
    lines = capsys.readouterr().out.splitlines()
    assert re.fullmatch(r"\*Entries :        5 : Total  Size= +\d+ bytes  File Size  = +\d+ \*", lines[6])
    assert re.fullmatch(r"\*Baskets :        1 : Basket Size= +32000 bytes  Compression= +\d\.\d\d     \*", lines[7])


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
    assert lines[1] == "*Entries :     3354 : Total  Size=     154237 bytes  File Size  =      32316 *"
    assert lines[2] == "*Baskets :        3 : Basket Size=      32000 bytes  Compression=   4.77     *"


def test_a_title_that_is_the_name_prints_the_leaf_type_instead(capsys):
    from xrdroot.pyroot.trees import TNtuple

    ntuple = TNtuple("ntuple", "Demo ntuple", "px:py")
    ntuple.Fill(1, 2)
    ntuple.GetBranch("px").Print()
    assert capsys.readouterr().out.splitlines()[0] == "*Br    0 :px        : Float_t".ljust(77) + "*"
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
    assert capsys.readouterr().out == "======> EVENT:0\n x               = 0\n n               = 0\n"
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
    assert drawn[-1][0][0] == "wrapped" and drawn[-1][1] == "" and "htemp" in objects
    assert t.Draw("x>>hx(10, 0, 20)", "", "goff") == 10 and len(drawn) == 1
    assert objects["hx"].name == "hx"
    assert t.Draw("x>>+hx", "", "goff") == 10 and objects["hx"].entries == 20
    t.SetAlias("twice", "2 * x")
    assert t.GetAlias("twice") == "2 * x" and t.GetAlias("nope") is None
    assert t.Draw("twice", "twice > 10", "goff", 5, 3) == 3
    t.SetWeight(2.0)
    assert t.GetWeight() == 2.0 and t.Draw("x", "", "goff") == 10
    assert objects["htemp"].sum(flow=True) == pytest.approx(20)
    t.SetEstimate(50)
    t.SetEstimate(-1)
    assert t.GetEstimate() == 50
    assert t.SetScanField(10) is None


def test_the_attributes_a_tree_draws_with_are_kept_for_graphics():
    t = _tree()
    t.SetMarkerColor(2)
    t.SetFillStyle(3001)
    assert t.GetMarkerColor() == 2 and t.GetFillStyle() == 3001 and t.GetLineWidth() == 1
    t.SetNameTitle("renamed", "and retitled")
    assert (t.GetName(), t.GetTitle()) == ("renamed", "and retitled")
