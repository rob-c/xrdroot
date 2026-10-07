"""``TFile``, ``TDirectoryFile`` and ``TKey``: files read, made, added to and listed as ROOT's."""

from __future__ import annotations

import pathlib

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import files

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_file_made_keeps_what_is_booked_after_it_and_writes_it(capsys):
    out = ROOT.TFile("out.root", "RECREATE", "the title")
    expect(
        (out.GetPath(), "out.root:/"),
        (ROOT.gDirectory, out),
        (out.GetOption(), "CREATE"),
        (bool(out.IsWritable()), True),
        (bool(out.IsOpen()), True),
        (bool(not out.IsZombie()), True),
        (bool(bool(out)), True),
    )
    h = ROOT.TH1F("hpx", "px", 10, 0, 1)
    h.Fill(0.5)
    sub = out.mkdir("sub", "the sub")
    expect(
        (sub.GetPath(), "out.root:/sub"),
        (ROOT.gDirectory, out),
    )
    sub.cd()
    inner = ROOT.TH1D("inner", "in", 2, 0, 1)
    expect(
        (bool(inner.GetDirectory() is sub), True),
        (bool(sub.GetFile() is out), True),
    )
    out.cd()
    expect(
        (out.Write(), 2),
        (h.Write(), 1),
    )
    out.ls()
    lines = capsys.readouterr().out.splitlines()
    expect(
        (lines[:2], ["TFile**\t\tout.root\tthe title", " TFile*\t\tout.root\tthe title"]),
        (bool(lines[2].startswith("  OBJ: TH1F\thpx\tpx : 0 at: 0x")), True),
        (lines[3], "  TDirectoryFile*\t\tsub\tthe sub"),
        (bool(lines[4].startswith("   OBJ: TH1D\tinner")), True),
        (
            lines[5:],
            [
                "   KEY: TH1D\tinner;1\tin",
                "  KEY: TDirectoryFile\tsub;1\tthe sub",
                "  KEY: TH1F\thpx;2\tpx [current cycle]",
                "  KEY: TH1F\thpx;1\tpx [backup cycle]",
            ],
        ),
        (bool(out.Get("hpx") is h), True),
        (bool(out.Get("sub/inner") is inner), True),
        (bool(out.Get("nothing") is None), True),
    )
    out.Close()
    expect(
        (bool(out.GetEND() == out.GetSize() > 0), True),
        (ROOT.gDirectory, ROOT.gROOT),
        (bool(not out.IsOpen()), True),
        (bool(out.GetSize() > 0), True),
    )
    out.Close()


def test_a_file_read_hands_back_roots_classes_and_keeps_its_histograms(capsys):
    source = ROOT.TFile.Open(str(DATA / "gauss-h1.root"))
    expect(
        (source.GetOption(), "READ"),
        (bool(not source.IsWritable()), True),
    )
    key = source.GetListOfKeys().At(0)
    name = key.GetName()
    h = source.Get(name)
    expect(
        (bool(h.InheritsFrom("TH1")), True),
        (bool(source.Get(name) is h), True),
        (bool(h.GetDirectory() is source), True),
        (bool(getattr(source, name) is h), True),
        (source.GetNkeys(), source.GetListOfKeys().GetSize()),
    )
    with pytest.raises(AttributeError, match="has no object"):
        source.not_there  # noqa: B018
    with pytest.raises(AttributeError):
        source._private  # noqa: B018
    expect(
        (bool(key.ReadObj() is not None), True),
        (key.ReadObject["TH1"]().GetName(), key.GetName()),
        (key.GetCycle(), 1),
        (key.GetClassName(), h.ClassName()),
        (bool(key.GetNbytes() > 0), True),
        (bool(key.GetObjlen() > 0), True),
        (bool(key.GetKeylen() > 0), True),
        (bool(key.GetSeekKey() > 0), True),
        (bool(key.GetDatime().GetYear() > 2000), True),
        (bool(not key.IsFolder()), True),
        (bool(key.GetMotherDir() is source), True),
    )
    key.Print()
    expect(
        (bool(capsys.readouterr().out.startswith(f"TKey Name = {name}, Title = ")), True),
        (bool(source.FindKey(name) is key), True),
        (bool(source.GetKey("nothing") is None), True),
        (bool(source.GetVersion() > 0), True),
        (bool(source.GetCompressionSettings() >= 0), True),
        (bool(source.GetCompressionLevel() >= 0), True),
        (bool(source.GetCompressionAlgorithm() >= 0), True),
    )
    source.SetCompressionLevel(5)
    source.SetCompressionSettings(101)
    source.Flush()
    source.Close()


def test_directories_on_file_are_walked_as_root_walks_them(capsys):
    source = ROOT.TFile(str(DATA / "dirs-6.14.00.root"))
    below = source.Get("dir1")
    expect(
        (below.ClassName(), "TDirectoryFile"),
        (bool(below.GetPath().endswith("dirs-6.14.00.root:/dir1")), True),
        (bool(source.GetDirectory("dir1") is below), True),
        (bool(source.GetDirectory("dir1/..") is source), True),
        (bool(source.GetDirectory(f"{source.GetName()}:/dir1") is below), True),
        (bool(source.GetDirectory("nowhere") is None), True),
        (bool(source.Get("nowhere/h") is None), True),
        (bool(below.cd()), True),
        (ROOT.gDirectory, below),
    )
    below.ls("-d")
    source.ls("-m")
    expect(
        (bool("KEY: TDirectoryFile\tdir11;1" in capsys.readouterr().out), True),
        (bool(below.mkdir("new") is None), True),
        (bool("Can not create directory" in capsys.readouterr().err), True),
        (below.WriteTObject(ROOT.TNamed("n", "")), 0),
        (below.Write(), 0),
        (bool("not writable" in capsys.readouterr().err), True),
    )
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
    expect(
        (bool(added.IsWritable()), True),
        (added.GetListOfKeys().GetSize(), 1),
    )
    first = added.Get("first")
    ROOT.TH1D("second", "two", 2, 0, 1)
    added.Write()
    added.Close()
    back = ROOT.TFile("up.root")
    expect(
        (
            bool(
                [key.GetName() for key in back.GetListOfKeys()] == ["first", "second"]
                or len(back.GetListOfKeys()) >= 2
            ),
            True,
        ),
        (first.GetName(), "first"),
    )
    back.Close()


def test_modes_refuse_as_root_refuses(capsys):
    expect(
        (bool(ROOT.TFile.Open("missing.root") is None), True),
        (bool(ROOT.TFile("missing.root").IsZombie()), True),
    )
    ROOT.TFile("there.root", "RECREATE").Close()
    assert ROOT.TFile("there.root", "NEW").IsZombie()
    pathlib.Path("junk.root").write_text("not a root file at all, but long enough to read" * 3)
    assert ROOT.TFile("junk.root").IsZombie()
    err = capsys.readouterr().err
    expect(
        (bool("file missing.root does not exist" in err), True),
        (bool("file there.root already exists" in err), True),
        (bool("file junk.root is not a ROOT file" in err), True),
        (bool(ROOT.TFile("fresh.root", "UPDATE").IsWritable()), True),
    )
    zombie = ROOT.TFile("missing.root")
    zombie.Close()
    assert not zombie


def test_a_file_is_a_context_manager_and_closes_at_the_end():
    with ROOT.TFile("ctx.root", "RECREATE") as made:
        ROOT.TH1D("h", "", 1, 0, 1)
        made.Write()
    expect(
        (bool(not made.IsOpen()), True),
        (bool(made not in files._OPEN), True),
    )
    ROOT.TFile("left.root", "RECREATE")
    files._close_all()
    assert files._OPEN == []


def test_a_remote_name_is_taken_to_be_there():
    expect(
        (bool(files._exists("root://host//file.root")), True),
        (bool(files._exists("file:///no/such.root") is False), True),
        (bool(not files._local("root://host//f.root")), True),
        (bool(files._local("file:///x")), True),
    )


def test_objects_of_classes_with_no_wrapper_come_back_named_by_their_class():
    source = ROOT.TFile(str(DATA / "tformula.root"))
    other = source.Get("fconv")
    expect(
        (other.ClassName(), "TF1Convolution"),
        (other.GetName(), "TF1Convolution"),
        (other.GetTitle(), ""),
        (bool(isinstance(other, files.TOther)), True),
    )
    named = files.TOther("TThing", {"TNamed": {"fName": "n", "fTitle": "t"}})
    expect(
        ((named.GetName(), named.GetTitle()), ("n", "t")),
        (files.TOther().GetName(), "TObject"),
    )
    source.Close()


def test_a_memory_directory_below_a_file_is_one_on_file_too():
    made = ROOT.TFile("deep.root", "RECREATE")
    deep = made.mkdir("a/b")
    expect(
        (deep.GetPath(), "deep.root:/a/b"),
        (bool(made.mkdir("a") is None), True),
        (made.mkdir("a", "", True).GetName(), "a"),
        (bool(made.GetDirectory("a/b") is deep), True),
    )
    deep.cd()
    ROOT.TH1D("h", "", 1, 0, 1)
    made.Write()
    made.Close()
    back = ROOT.TFile("deep.root")
    assert back.Get("a/b/h").GetNbinsX() == 1
    back.Close()


def test_get_object_fills_the_pointer_it_is_handed_as_the_translator_hands_one():
    """FitHistoInFile.C's ``f->GetObject("histo", histo)``: the pointer is set, or nulled."""

    class Pointer:
        value: object = "unset"

    out = ROOT.TFile("objects.root", "RECREATE")
    ROOT.TH1D("histo", "t", 2, 0, 1).Fill(0.5)
    out.Write()
    out.Close()
    back = ROOT.TFile("objects.root")
    found, missing = Pointer(), Pointer()
    assert back.GetObject("histo", found) is found.value
    assert (found.value.GetEntries(), back.GetObject("nothing", missing), missing.value) == (
        1,
        None,
        None,
    )
    assert back.GetObject("histo", "not a pointer").GetName() == "histo"


def test_a_file_is_copied_here_and_a_missing_one_is_said_so(tmp_path, capsys) -> None:
    source = tmp_path / "a.root"
    source.write_bytes(b"0123")
    assert ROOT.TFile.Cp(str(source), str(tmp_path / "b.root"))
    assert (tmp_path / "b.root").read_bytes() == b"0123"
    assert "[TFile::Cp] Total 0.00 MB" in capsys.readouterr().err
    assert ROOT.TFile.Cp(str(source), str(tmp_path / "c.root"), False)
    assert not ROOT.TFile.Cp(str(tmp_path / "none.root"), str(tmp_path / "d.root"))
    assert "Error in <TFile::Cp>: cannot open source file" in capsys.readouterr().err


def test_a_file_is_written_with_roots_compression_unless_told_another() -> None:
    plain = ROOT.TFile("plain.root", "RECREATE")
    assert (plain.GetCompressionSettings(), plain.GetCompressionAlgorithm()) == (101, 1)
    plain.SetCompressionLevel(6)
    assert plain.GetCompressionSettings() == 106
    plain.SetCompressionSettings(207)
    assert (plain.GetCompressionSettings(), plain.GetCompressionLevel()) == (207, 7)
    plain.Close()
    raw = ROOT.TFile("raw.root", "RECREATE", "", 0)
    assert raw.GetCompressionSettings() == 0
    raw.Close()
    assert ROOT.TFile("plain.root").GetCompressionSettings() == 207


def test_a_file_on_a_macros_stack_is_closed_as_its_scope_ends() -> None:
    from xrdroot.cint import translate
    from xrdroot.cint.execute import run_source

    source = 'void t() { { TFile f("stack.root", "RECREATE"); } TFile::Open("stack.root"); }'
    assert "f._destruct()" in translate(source, "t.C")
    run_source(source, "t.C", root=ROOT)
    assert ROOT.TFile("stack.root").IsOpen()
