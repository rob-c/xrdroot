"""``TObject``, ``TNamed``, ``TClass`` and the ``TAtt`` mixins, by ROOT's names."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import hooks
from xrdroot.pyroot.core.objects import TAttLine, TAttMarker, TAttText, attribute_home


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


class Drawn(ROOT.TNamed, TAttLine, TAttMarker, TAttText):
    """A named object with drawing attributes of its own, as a graphics class has."""


def test_a_tobject_is_named_and_titled_by_its_class():
    thing = ROOT.TObject()
    expect(
        (thing.GetName(), "TObject"),
        (thing.GetTitle(), "Basic ROOT object"),
        (thing.ClassName(), "TObject"),
        (thing.IsA().GetName(), "TObject"),
        (ROOT.TObject.Class(), ROOT.TClass("TObject")),
        (thing.GetIconName(), "TObject"),
        (thing.GetObjectInfo(0, 0), ""),
        (bool(not thing.IsFolder()), True),
        (bool(not thing.IsZombie()), True),
        (thing.Hash(), hash("TObject")),
        (bool(thing.FindObject("x") is None), True),
    )


def test_inherits_from_follows_the_class_hierarchy():
    h = ROOT.TH1F("h", "", 1, 0, 1)
    expect(
        (bool(h.InheritsFrom("TH1")), True),
        (bool(h.InheritsFrom(ROOT.TClass("TNamed"))), True),
        (bool(h.InheritsFrom("TH1F")), True),
        (bool(not h.InheritsFrom("TGraph")), True),
        (bool(ROOT.TClass("TH1F").InheritsFrom("TObject")), True),
        (bool(not ROOT.TClass("TNoSuch").InheritsFrom("TObject")), True),
    )


def test_a_class_makes_its_objects_by_name():
    made = ROOT.TClass.GetClass("TNamed").New()
    expect(
        (bool(isinstance(made, ROOT.TNamed)), True),
        (repr(ROOT.TClass("TH1D")), "<TClass TH1D>"),
        (len({ROOT.TClass("A"), ROOT.TClass("A")}), 1),
        (bool(ROOT.TClass("A") != "A"), True),
    )
    with pytest.raises(TypeError, match="has no class TNoSuch"):
        ROOT.TClass("TNoSuch").New()


def test_bits_are_set_tested_reset_and_inverted():
    thing = ROOT.TObject()
    thing.SetBit(ROOT.kCanDelete)
    expect(
        (bool(thing.TestBit(ROOT.kCanDelete)), True),
        (thing.TestBits(3), 1),
    )
    thing.SetBit(ROOT.kCanDelete, False)
    assert not thing.TestBit(ROOT.kCanDelete)
    thing.InvertBit(4)
    assert thing.TestBit(4)
    thing.ResetBit(4)
    assert not thing.TestBit(4)
    thing.SetUniqueID(7)
    assert thing.GetUniqueID() == 7


def test_a_tnamed_keeps_its_name_and_title_and_compares_by_name():
    a, b = ROOT.TNamed("a", "first"), ROOT.TNamed("b", "second")
    a.SetNameTitle("c", "third")
    expect(
        ((a.GetName(), a.GetTitle()), ("c", "third")),
        (a.Compare(b), 1),
        (b.Compare(a), -1),
        (a.Compare(a), 0),
        (a.Sizeof(), len("c") + len("third")),
    )
    copied = ROOT.TNamed(a)
    expect(
        (copied.GetName(), "c"),
        (bool(copied is not a), True),
    )


def test_a_copy_constructor_copies_a_tobject():
    first = ROOT.TNamed("n", "t")
    second = ROOT.TObject(first)
    assert second.__dict__["_name"] == "n"


def test_clone_shares_nothing_and_takes_a_new_name():
    a = ROOT.TNamed("a", "t")
    b = a.Clone("b")
    expect(
        (b.GetName(), "b"),
        (a.GetName(), "a"),
        (ROOT.TObject().Clone().ClassName(), "TObject"),
    )


def test_print_and_ls_are_roots_obj_lines(capsys):
    thing = ROOT.TNamed("n", "the title")
    thing.Print()
    thing.ls("noaddr")
    thing.ls()
    lines = capsys.readouterr().out.splitlines()
    expect(
        (lines[0], "OBJ: TNamed\tn\tthe title"),
        (lines[1], "OBJ: TNamed\tn\tthe title : 0"),
        (bool(lines[2].startswith("OBJ: TNamed\tn\tthe title : 0 at: 0x")), True),
    )


def test_dump_prints_every_member(capsys):
    ROOT.TNamed("n", "t").Dump()
    assert "_name" in capsys.readouterr().out


def test_draw_goes_through_the_hook_the_graphics_installs():
    thing = ROOT.TNamed("n", "")
    thing.Draw("same")
    assert hooks.DRAWN[-1] == (thing, "same")
    seen = []
    ROOT.set_draw_hook(lambda obj, option: seen.append(option))
    copy = thing.DrawClone("L")
    thing.AppendPad("P")
    thing.Paint()
    thing.Pop()
    expect(
        (seen, ["L", "P"]),
        (bool(copy is not thing), True),
    )


def test_writing_with_no_file_open_is_roots_error(capsys):
    expect(
        (ROOT.TNamed("n", "").Write(), 0),
        (bool("is not associated with a file" in capsys.readouterr().err), True),
    )


def test_save_as_writes_a_file_of_its_own(capsys, tmp_path):
    ROOT.TH1D("h", "", 2, 0, 1).SaveAs("h.root")
    expect(
        (bool("ROOT file h.root has been created" in capsys.readouterr().out), True),
        (ROOT.TFile("h.root").Get("h").GetNbinsX(), 2),
    )


def test_delete_forgets_an_object_in_memory():
    h = ROOT.TH1D("h", "", 2, 0, 1)
    assert ROOT.gROOT.FindObject("h") is h
    h.Delete()
    assert ROOT.gROOT.FindObject("h") is None
    ROOT.TNamed("x", "").Delete()
    h.RecursiveRemove(h)


def test_an_objects_messages_name_its_class(capsys):
    thing = ROOT.TNamed("n", "")
    thing.Error("Fn", "bad %d", 1)
    thing.Warning("Fn", "odd")
    thing.Info("Fn", "fyi")
    err = capsys.readouterr().err.splitlines()
    assert err == ["Error in <TNamed::Fn>: bad 1", "Warning in <TNamed::Fn>: odd",
                   "Info in <TNamed::Fn>: fyi"]  # fmt: skip


def test_the_repr_is_pyroots():
    assert repr(ROOT.TNamed("n", "")).startswith("<cppyy.gbl.TNamed object at 0x")


def test_attributes_are_kept_on_an_object_without_members_of_its_own():
    thing = Drawn("n", "")
    expect(
        (thing.GetLineColor(), 1),
        (thing.GetMarkerSize(), 1.0),
    )
    thing.SetLineColor(ROOT.kRed + 1)
    thing.SetMarkerSize(2)
    thing.SetTextSize(0.04)
    thing.SetLineColorAlpha(ROOT.kBlue, 0.5)
    expect(
        (thing.GetLineColor(), ROOT.kBlue),
        (attribute_home(thing, "TAttLine")["fLineAlpha"], 0.5),
        (thing.GetMarkerSize(), 2.0),
        (thing.GetTextSize(), 0.04),
    )
    other = Drawn("m", "")
    thing.CopyTAttLine(other)
    assert other.GetLineColor() == ROOT.kBlue
    thing.ResetAttLine()
    assert thing.GetLineColor() == 1


def test_attributes_of_a_histogram_are_its_members():
    h = ROOT.TH1F("h", "", 1, 0, 1)
    h.SetFillColor(ROOT.kYellow)
    h.SetLineWidth(3)
    expect(
        (h._xrd._core["TAttFill"]["fFillColor"], ROOT.kYellow),
        (h._xrd._core["TAttLine"]["fLineWidth"], 3),
    )
