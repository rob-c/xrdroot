"""``TObject``, ``TNamed``, ``TClass`` and the ``TAtt`` mixins, by ROOT's names."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import hooks
from xrdroot.pyroot.core.objects import TAttLine, TAttMarker, TAttText, attribute_home


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


class Drawn(ROOT.TNamed, TAttLine, TAttMarker, TAttText):
    """A named object with drawing attributes of its own, as a graphics class has."""


def test_a_tobject_is_named_and_titled_by_its_class():
    thing = ROOT.TObject()
    assert thing.GetName() == "TObject" and thing.GetTitle() == "Basic ROOT object"
    assert thing.ClassName() == "TObject" and thing.IsA().GetName() == "TObject"
    assert ROOT.TObject.Class() == ROOT.TClass("TObject")
    assert thing.GetIconName() == "TObject" and thing.GetObjectInfo(0, 0) == ""
    assert not thing.IsFolder() and not thing.IsZombie()
    assert thing.Hash() == hash("TObject") and thing.FindObject("x") is None


def test_inherits_from_follows_the_class_hierarchy():
    h = ROOT.TH1F("h", "", 1, 0, 1)
    assert h.InheritsFrom("TH1") and h.InheritsFrom(ROOT.TClass("TNamed"))
    assert h.InheritsFrom("TH1F") and not h.InheritsFrom("TGraph")
    assert ROOT.TClass("TH1F").InheritsFrom("TObject")
    assert not ROOT.TClass("TNoSuch").InheritsFrom("TObject")


def test_a_class_makes_its_objects_by_name():
    made = ROOT.TClass.GetClass("TNamed").New()
    assert isinstance(made, ROOT.TNamed)
    assert repr(ROOT.TClass("TH1D")) == "<TClass TH1D>"
    assert len({ROOT.TClass("A"), ROOT.TClass("A")}) == 1
    assert ROOT.TClass("A") != "A"
    with pytest.raises(TypeError, match="has no class TNoSuch"):
        ROOT.TClass("TNoSuch").New()


def test_bits_are_set_tested_reset_and_inverted():
    thing = ROOT.TObject()
    thing.SetBit(ROOT.kCanDelete)
    assert thing.TestBit(ROOT.kCanDelete) and thing.TestBits(3) == 1
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
    assert (a.GetName(), a.GetTitle()) == ("c", "third")
    assert a.Compare(b) == 1 and b.Compare(a) == -1 and a.Compare(a) == 0
    assert a.Sizeof() == len("c") + len("third")
    copied = ROOT.TNamed(a)
    assert copied.GetName() == "c" and copied is not a


def test_a_copy_constructor_copies_a_tobject():
    first = ROOT.TNamed("n", "t")
    second = ROOT.TObject(first)
    assert second.__dict__["_name"] == "n"


def test_clone_shares_nothing_and_takes_a_new_name():
    a = ROOT.TNamed("a", "t")
    b = a.Clone("b")
    assert b.GetName() == "b" and a.GetName() == "a"
    assert ROOT.TObject().Clone().ClassName() == "TObject"


def test_print_and_ls_are_roots_obj_lines(capsys):
    thing = ROOT.TNamed("n", "the title")
    thing.Print()
    thing.ls("noaddr")
    thing.ls()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "OBJ: TNamed\tn\tthe title"
    assert lines[1] == "OBJ: TNamed\tn\tthe title : 0"
    assert lines[2].startswith("OBJ: TNamed\tn\tthe title : 0 at: 0x")


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
    assert seen == ["L", "P"] and copy is not thing


def test_writing_with_no_file_open_is_roots_error(capsys):
    assert ROOT.TNamed("n", "").Write() == 0
    assert "is not associated with a file" in capsys.readouterr().err


def test_save_as_writes_a_file_of_its_own(capsys, tmp_path):
    ROOT.TH1D("h", "", 2, 0, 1).SaveAs("h.root")
    assert "ROOT file h.root has been created" in capsys.readouterr().out
    assert ROOT.TFile("h.root").Get("h").GetNbinsX() == 2


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
    assert thing.GetLineColor() == 1 and thing.GetMarkerSize() == 1.0
    thing.SetLineColor(ROOT.kRed + 1)
    thing.SetMarkerSize(2)
    thing.SetTextSize(0.04)
    thing.SetLineColorAlpha(ROOT.kBlue, 0.5)
    assert (
        thing.GetLineColor() == ROOT.kBlue
        and attribute_home(thing, "TAttLine")["fLineAlpha"] == 0.5
    )
    assert thing.GetMarkerSize() == 2.0 and thing.GetTextSize() == 0.04
    other = Drawn("m", "")
    thing.CopyTAttLine(other)
    assert other.GetLineColor() == ROOT.kBlue
    thing.ResetAttLine()
    assert thing.GetLineColor() == 1


def test_attributes_of_a_histogram_are_its_members():
    h = ROOT.TH1F("h", "", 1, 0, 1)
    h.SetFillColor(ROOT.kYellow)
    h.SetLineWidth(3)
    assert h._xrd._core["TAttFill"]["fFillColor"] == ROOT.kYellow
    assert h._xrd._core["TAttLine"]["fLineWidth"] == 3
