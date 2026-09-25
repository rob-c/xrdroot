"""``gROOT`` and ``gDirectory``: the lists ROOT keeps, memory, and C++ handed to the translator."""

from __future__ import annotations

import sys
import types

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core import directories, troot


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_groot_is_the_top_directory_pyroot():
    assert ROOT.gROOT.GetName() == "PyROOT" and ROOT.gROOT.GetTitle() == "The ROOT of EVERYTHING"
    assert ROOT.gROOT.GetPath() == "PyROOT:/" and ROOT.gDirectory.GetPath() == "PyROOT:/"
    assert ROOT.gDirectory == ROOT.gROOT and ROOT.gDirectory == ROOT.gDirectory
    assert hash(ROOT.gDirectory) == id(ROOT.gDirectory) and repr(ROOT.gDirectory) == repr(
        ROOT.gROOT
    )
    assert ROOT.gROOT._xrd is __import__("xrdroot").gROOT


def test_groot_says_it_is_in_batch_mode_unless_told_otherwise():
    assert ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(False)
    assert not ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetWebDisplay("off")
    assert not ROOT.gROOT.IsWebDisplay()


def test_groot_names_the_release_it_follows():
    assert ROOT.gROOT.GetVersion() == "6.40.04" and ROOT.gROOT.GetVersionInt() == 64004
    assert ROOT.gROOT.GetVersionCode() == (6 << 16) + (40 << 8) + 4
    assert ROOT.gROOT.GetGitCommit() == "" and ROOT.gROOT.GetConfigFeatures() == "pyroot"


def test_the_tutorial_directory_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROOT_TUTORIAL_DIR", "/tut")
    assert ROOT.gROOT.GetTutorialDir() == "/tut" and ROOT.gROOT.GetTutorialsDir() == "/tut"
    monkeypatch.delenv("ROOT_TUTORIAL_DIR")
    monkeypatch.setenv("ROOTSYS", "/opt/root")
    assert ROOT.gROOT.GetTutorialDir() == "/opt/root/tutorials"


def test_every_list_groot_keeps_is_a_tlist():
    for name in ("Files", "Canvases", "Functions", "Styles", "Specials", "Globals", "Browsers",
                 "Geometries", "Colors", "Types", "Cleanups", "ClosedObjects", "DataSets",
                 "MappedFiles", "Sockets", "Tasks"):  # fmt: skip
        assert isinstance(getattr(ROOT.gROOT, f"GetListOf{name}")(), ROOT.TList)


def test_find_object_looks_in_the_lists_then_the_current_directory():
    h = ROOT.TH1D("h", "", 1, 0, 1)
    f = ROOT.TF1("myfunc", "x", 0, 1)
    assert ROOT.gROOT.FindObject("h") is h and ROOT.gROOT.FindObjectAny("myfunc") is f
    assert ROOT.gROOT.FindObject(h) is h and ROOT.gROOT.FindObject("nothing") is None
    out = ROOT.TFile("f.root", "RECREATE")
    assert ROOT.gROOT.FindObject("f.root") is out and ROOT.gROOT.FindObject("h") is None


def test_get_function_makes_a_standard_function_when_first_asked():
    gaus = ROOT.gROOT.GetFunction("gaus")
    assert gaus.GetNpar() == 3 and ROOT.gROOT.GetFunction("gaus") is gaus
    assert ROOT.gROOT.GetFunction("not_a_function") is None


def test_force_style_is_noted():
    ROOT.gROOT.ForceStyle()
    assert ROOT.gROOT.GetForceStyle()


def test_reset_forgets_memory_and_goes_back_to_it():
    ROOT.TH1D("h", "", 1, 0, 1)
    ROOT.TFile("f.root", "RECREATE")
    ROOT.gROOT.Reset()
    assert ROOT.gDirectory == ROOT.gROOT and ROOT.gROOT.FindObject("h") is None
    ROOT.gROOT.CloseFiles()
    ROOT.gROOT.EndOfProcessCleanups()
    assert ROOT.gROOT.GetListOfFiles().IsEmpty()


def test_the_small_session_questions_have_roots_answers():
    ROOT.gROOT.SetMacroPath("/m")
    assert ROOT.gROOT.GetMacroPath() == "/m" and ROOT.gROOT.GetDirLevel() == 0
    assert ROOT.gROOT.GetFile() is None and ROOT.gROOT.GetApplication() is None
    assert not ROOT.gROOT.IsInterrupted() and ROOT.gROOT.GetSelectedPad() is None
    ROOT.gROOT.Time()
    ROOT.gROOT.RefreshBrowsers()


def test_cpp_is_refused_by_name_without_the_translator(monkeypatch):
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(UnsupportedFeatureError, match=r"xrdroot\.cint"):
        ROOT.gROOT.ProcessLine("int x = 1;")
    with pytest.raises(UnsupportedFeatureError, match=r"gInterpreter\.Declare"):
        ROOT.gInterpreter.Declare("int f();")


def test_cpp_goes_to_the_translator_when_it_is_installed(monkeypatch):
    seen = []
    fake = types.ModuleType("xrdroot.cint")
    fake.process_line = lambda text: seen.append(text) or 7
    monkeypatch.setitem(sys.modules, "xrdroot.cint", fake)
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    try:
        assert ROOT.gROOT.ProcessLine("1+1") == 7 and ROOT.gROOT.ProcessLineSync("a") == 7
        assert ROOT.gROOT.ProcessLineFast("b") == 7 and ROOT.gROOT.Macro("m.C") == 7
        assert ROOT.gROOT.LoadMacro("m.C") == 0 and ROOT.gInterpreter.Declare("int y;")
        assert ROOT.gInterpreter.ProcessLine("c") == 7 and ROOT.gInterpreter.Calc("d") == 7
    finally:
        troot.set_line_processor(None)
    assert seen == ["1+1", "a", "b", ".x m.C", ".L m.C", "int y;", "c", "d"]
    assert ROOT.gInterpreter.Load("lib") == 0 and ROOT.gInterpreter.GenerateDictionary("x") == 0
    ROOT.gInterpreter.AddIncludePath("-I.")


def test_a_translator_module_without_process_line_is_still_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "xrdroot.cint", types.ModuleType("xrdroot.cint"))
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    with pytest.raises(UnsupportedFeatureError):
        ROOT.gROOT.ProcessLine("1")


def test_memory_directories_nest_and_are_walked(capsys):
    top = ROOT.gROOT.mkdir("a/b", "the b")
    assert top.GetName() == "b" and top.GetPath() == "PyROOT:/a/b"
    assert ROOT.gROOT.mkdir("a") is None and ROOT.gROOT.mkdir("a", "", True).GetName() == "a"
    assert ROOT.gROOT.GetDirectory("a/b") is top and ROOT.gROOT.GetDirectory("a/../a/./b") is top
    assert ROOT.gROOT.GetDirectory("zz") is None and ROOT.gROOT.GetDirectory("a/zz/q") is None
    assert top.cd() and ROOT.gDirectory.GetName() == "b" and top.GetMother().GetName() == "a"
    ROOT.gDirectory.pwd()
    assert ROOT.gROOT.cd("a") and not ROOT.gROOT.cd("zz")
    out = capsys.readouterr()
    assert out.out == "PyROOT:/a/b\n" and "Unknown directory zz" in out.err
    assert top.GetMotherDir().GetName() == "a" and top.GetFile() is None and not top.IsWritable()


def test_a_directory_keeps_what_is_made_in_it_and_replaces_by_name(capsys):
    ROOT.gROOT.mkdir("d").cd()
    first = ROOT.TH1D("h", "", 1, 0, 1)
    second = ROOT.TH1D("h", "", 1, 0, 1)
    here = ROOT.gDirectory.__real__()
    assert here.Get("h") is second and here.Get("h;1") is second
    assert ROOT.gROOT.Get("d/h") is second and ROOT.gROOT.Get("zz/h") is None
    assert "Replacing existing TH1D: h (Potential memory leak)." in capsys.readouterr().err
    assert first.GetDirectory() is None and second.GetDirectory() is here
    here.Add(ROOT.TNamed("n", ""))
    assert ROOT.gROOT.FindObjectAny("n").GetName() == "n" and here.GetObject("n").GetName() == "n"
    here.Delete("n")
    assert here.FindObject("n") is None and len(here.GetList()) == 1
    here.ReadAll()
    here.SaveSelf()
    here.DeleteAll()
    assert here.GetList().IsEmpty() and here.GetListOfKeys().IsEmpty()


def test_a_memory_directory_lists_its_objects_one_level_in(capsys):
    ROOT.TH1D("h1", "one", 1, 0, 1)
    ROOT.TH1D("x2", "two", 1, 0, 1)
    ROOT.gROOT.ls()
    ROOT.gDirectory.ls("-mh*")
    ROOT.gROOT.ls("noaddr")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(" OBJ: TH1D\th1\tone : 0 at: 0x") and lines[1].startswith(
        " OBJ: TH1D\tx2"
    )
    assert len(lines) == 3 and lines[2].startswith(" OBJ: TH1D\th1")


def test_writing_a_memory_directory_writes_nothing(capsys):
    ROOT.TH1D("h", "", 1, 0, 1)
    assert ROOT.gROOT.Write() == 0 and ROOT.gROOT.WriteObject(ROOT.TNamed("n", ""), "m") == 0
    assert "The object (m) has not been written" in capsys.readouterr().err
    ROOT.gROOT.Close()
    assert ROOT.gROOT.GetList().IsEmpty()


def test_a_context_goes_back_to_the_directory_it_was_made_in():
    below = ROOT.gROOT.mkdir("below")
    with ROOT.TDirectory.TContext(below):
        assert ROOT.gDirectory == below
    assert ROOT.gDirectory == ROOT.gROOT
    with ROOT.TDirectory.TContext() as context:
        below.cd()
        assert context is not None
    assert ROOT.gDirectory == ROOT.gROOT
    directories.forget(ROOT.TNamed("n", ""))
