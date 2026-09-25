"""``TFile``, ``TDirectoryFile`` and ``TKey``: files read, made, added to and listed as ROOT's."""

from __future__ import annotations

import pathlib

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import files

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_file_made_keeps_what_is_booked_after_it_and_writes_it(capsys):
    out = ROOT.TFile("out.root", "RECREATE", "the title")
    assert out.GetPath() == "out.root:/" and ROOT.gDirectory == out and out.GetOption() == "CREATE"
    assert out.IsWritable() and out.IsOpen() and not out.IsZombie() and bool(out)
    h = ROOT.TH1F("hpx", "px", 10, 0, 1)
    h.Fill(0.5)
    sub = out.mkdir("sub", "the sub")
    assert sub.GetPath() == "out.root:/sub" and ROOT.gDirectory == out
    sub.cd()
    inner = ROOT.TH1D("inner", "in", 2, 0, 1)
    assert inner.GetDirectory() is sub and sub.GetFile() is out
    out.cd()
    assert out.Write() == 2 and h.Write() == 1
    out.ls()
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["TFile**\t\tout.root\tthe title", " TFile*\t\tout.root\tthe title"]
    assert lines[2].startswith("  OBJ: TH1F\thpx\tpx : 0 at: 0x")
    assert lines[3] == "  TDirectoryFile*\t\tsub\tthe sub" and lines[4].startswith(
        "   OBJ: TH1D\tinner"
    )
    assert lines[5:] == [
        "   KEY: TH1D\tinner;1\tin",
        "  KEY: TDirectoryFile\tsub;1\tthe sub",
        "  KEY: TH1F\thpx;2\tpx [current cycle]",
        "  KEY: TH1F\thpx;1\tpx [backup cycle]",
    ]
    assert out.Get("hpx") is h and out.Get("sub/inner") is inner and out.Get("nothing") is None
    out.Close()
    assert out.GetEND() == out.GetSize() > 0
    assert ROOT.gDirectory == ROOT.gROOT and not out.IsOpen() and out.GetSize() > 0
    out.Close()


def test_a_file_read_hands_back_roots_classes_and_keeps_its_histograms(capsys):
    source = ROOT.TFile.Open(str(DATA / "gauss-h1.root"))
    assert source.GetOption() == "READ" and not source.IsWritable()
    key = source.GetListOfKeys().At(0)
    name = key.GetName()
    h = source.Get(name)
    assert h.InheritsFrom("TH1") and source.Get(name) is h and h.GetDirectory() is source
    assert getattr(source, name) is h and source.GetNkeys() == source.GetListOfKeys().GetSize()
    with pytest.raises(AttributeError, match="has no object"):
        source.not_there  # noqa: B018
    with pytest.raises(AttributeError):
        source._private  # noqa: B018
    assert key.ReadObj() is not None and key.GetCycle() == 1 and key.GetClassName() == h.ClassName()
    assert (
        key.GetNbytes() > 0 and key.GetObjlen() > 0 and key.GetKeylen() > 0 and key.GetSeekKey() > 0
    )
    assert key.GetDatime().GetYear() > 2000 and not key.IsFolder() and key.GetMotherDir() is source
    key.Print()
    assert capsys.readouterr().out.startswith(f"TKey Name = {name}, Title = ")
    assert source.FindKey(name) is key and source.GetKey("nothing") is None
    assert source.GetVersion() > 0 and source.GetCompressionSettings() >= 0
    assert source.GetCompressionLevel() >= 0 and source.GetCompressionAlgorithm() >= 0
    source.SetCompressionLevel(5)
    source.SetCompressionSettings(101)
    source.Flush()
    source.Close()


def test_directories_on_file_are_walked_as_root_walks_them(capsys):
    source = ROOT.TFile(str(DATA / "dirs-6.14.00.root"))
    below = source.Get("dir1")
    assert below.ClassName() == "TDirectoryFile" and below.GetPath().endswith(
        "dirs-6.14.00.root:/dir1"
    )
    assert source.GetDirectory("dir1") is below and source.GetDirectory("dir1/..") is source
    assert source.GetDirectory(f"{source.GetName()}:/dir1") is below
    assert source.GetDirectory("nowhere") is None and source.Get("nowhere/h") is None
    assert below.cd() and ROOT.gDirectory == below
    below.ls("-d")
    source.ls("-m")
    assert "KEY: TDirectoryFile\tdir11;1" in capsys.readouterr().out
    assert below.mkdir("new") is None and "Can not create directory" in capsys.readouterr().err
    assert below.WriteTObject(ROOT.TNamed("n", "")) == 0 and below.Write() == 0
    assert "not writable" in capsys.readouterr().err
    source.Close()
    assert ROOT.gDirectory == ROOT.gROOT


def test_a_listing_can_be_narrowed_by_a_wildcard(capsys):
    source = ROOT.TFile(str(DATA / "graphs.root"))
    source.ls("t*")
    lines = capsys.readouterr().out.splitlines()
    assert all("KEY" not in line or "\tt" in line for line in lines)
    source.Close()


def test_a_file_is_added_to_and_read_back(capsys):
    made = ROOT.TFile("up.root", "RECREATE")
    ROOT.TH1D("first", "one", 2, 0, 1)
    made.Write()
    made.Close()
    added = ROOT.TFile("up.root", "UPDATE")
    assert added.IsWritable() and added.GetListOfKeys().GetSize() == 1
    first = added.Get("first")
    ROOT.TH1D("second", "two", 2, 0, 1)
    added.Write()
    added.Close()
    back = ROOT.TFile("up.root")
    assert [key.GetName() for key in back.GetListOfKeys()] == ["first", "second"] or len(
        back.GetListOfKeys()
    ) >= 2
    assert first.GetName() == "first"
    back.Close()


def test_modes_refuse_as_root_refuses(capsys):
    assert ROOT.TFile.Open("missing.root") is None and ROOT.TFile("missing.root").IsZombie()
    ROOT.TFile("there.root", "RECREATE").Close()
    assert ROOT.TFile("there.root", "NEW").IsZombie()
    pathlib.Path("junk.root").write_text("not a root file at all, but long enough to read" * 3)
    assert ROOT.TFile("junk.root").IsZombie()
    err = capsys.readouterr().err
    assert "file missing.root does not exist" in err and "file there.root already exists" in err
    assert "file junk.root is not a ROOT file" in err
    assert ROOT.TFile("fresh.root", "UPDATE").IsWritable()
    zombie = ROOT.TFile("missing.root")
    zombie.Close()
    assert not zombie


def test_a_file_is_a_context_manager_and_closes_at_the_end():
    with ROOT.TFile("ctx.root", "RECREATE") as made:
        ROOT.TH1D("h", "", 1, 0, 1)
        made.Write()
    assert not made.IsOpen() and made not in files._OPEN
    ROOT.TFile("left.root", "RECREATE")
    files._close_all()
    assert files._OPEN == []


def test_a_remote_name_is_taken_to_be_there():
    assert (
        files._exists("root://host//file.root") and files._exists("file:///no/such.root") is False
    )
    assert not files._local("root://host//f.root") and files._local("file:///x")


def test_objects_of_classes_with_no_wrapper_come_back_named_by_their_class():
    source = ROOT.TFile(str(DATA / "tformula.root"))
    other = source.Get("fconv")
    assert other.ClassName() == "TF1Convolution" and other.GetName() == "TF1Convolution"
    assert other.GetTitle() == "" and isinstance(other, files.TOther)
    named = files.TOther("TThing", {"TNamed": {"fName": "n", "fTitle": "t"}})
    assert (named.GetName(), named.GetTitle()) == ("n", "t")
    assert files.TOther().GetName() == "TObject"
    source.Close()


def test_a_memory_directory_below_a_file_is_one_on_file_too():
    made = ROOT.TFile("deep.root", "RECREATE")
    deep = made.mkdir("a/b")
    assert deep.GetPath() == "deep.root:/a/b" and made.mkdir("a") is None
    assert made.mkdir("a", "", True).GetName() == "a" and made.GetDirectory("a/b") is deep
    deep.cd()
    ROOT.TH1D("h", "", 1, 0, 1)
    made.Write()
    made.Close()
    back = ROOT.TFile("deep.root")
    assert back.Get("a/b/h").GetNbinsX() == 1
    back.Close()
