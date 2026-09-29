"""``gROOT`` and ``gDirectory``: the lists ROOT keeps, memory, and C++ handed to the translator."""

from __future__ import annotations

import sys
import types

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core import directories, troot


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_groot_is_the_top_directory_pyroot():
    expect(
        (ROOT.gROOT.GetName(), "PyROOT"),
        (ROOT.gROOT.GetTitle(), "The ROOT of EVERYTHING"),
        (ROOT.gROOT.GetPath(), "PyROOT:/"),
        (ROOT.gDirectory.GetPath(), "PyROOT:/"),
        (ROOT.gDirectory, ROOT.gROOT),
        (ROOT.gDirectory, ROOT.gDirectory),
        (hash(ROOT.gDirectory), id(ROOT.gDirectory)),
        (repr(ROOT.gDirectory), repr(ROOT.gROOT)),
        (bool(ROOT.gROOT._xrd is __import__("xrdroot").gROOT), True),
    )


def test_groot_says_it_is_in_batch_mode_unless_told_otherwise():
    assert ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(False)
    assert not ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetWebDisplay("off")
    assert not ROOT.gROOT.IsWebDisplay()


def test_groot_names_the_release_it_follows():
    expect(
        (ROOT.gROOT.GetVersion(), "6.40.04"),
        (ROOT.gROOT.GetVersionInt(), 64004),
        (ROOT.gROOT.GetVersionCode(), (6 << 16) + (40 << 8) + 4),
        (ROOT.gROOT.GetGitCommit(), ""),
        (ROOT.gROOT.GetConfigFeatures(), "pyroot"),
    )


def test_the_tutorial_directory_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("ROOT_TUTORIAL_DIR", "/tut")
    expect(
        (ROOT.gROOT.GetTutorialDir(), "/tut"),
        (ROOT.gROOT.GetTutorialsDir(), "/tut"),
    )
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
    expect(
        (bool(ROOT.gROOT.FindObject("h") is h), True),
        (bool(ROOT.gROOT.FindObjectAny("myfunc") is f), True),
        (bool(ROOT.gROOT.FindObject(h) is h), True),
        (bool(ROOT.gROOT.FindObject("nothing") is None), True),
    )
    out = ROOT.TFile("f.root", "RECREATE")
    expect(
        (bool(ROOT.gROOT.FindObject("f.root") is out), True),
        (bool(ROOT.gROOT.FindObject("h") is None), True),
    )


def test_get_function_makes_a_standard_function_when_first_asked():
    gaus = ROOT.gROOT.GetFunction("gaus")
    expect(
        (gaus.GetNpar(), 3),
        (bool(ROOT.gROOT.GetFunction("gaus") is gaus), True),
        (bool(ROOT.gROOT.GetFunction("not_a_function") is None), True),
    )


def test_force_style_is_noted():
    ROOT.gROOT.ForceStyle()
    assert ROOT.gROOT.GetForceStyle()


def test_reset_forgets_memory_and_goes_back_to_it():
    ROOT.TH1D("h", "", 1, 0, 1)
    ROOT.TFile("f.root", "RECREATE")
    ROOT.gROOT.Reset()
    expect(
        (ROOT.gDirectory, ROOT.gROOT),
        (bool(ROOT.gROOT.FindObject("h") is None), True),
    )
    ROOT.gROOT.CloseFiles()
    ROOT.gROOT.EndOfProcessCleanups()
    assert ROOT.gROOT.GetListOfFiles().IsEmpty()


def test_the_small_session_questions_have_roots_answers():
    ROOT.gROOT.SetMacroPath("/m")
    expect(
        (ROOT.gROOT.GetMacroPath(), "/m"),
        (ROOT.gROOT.GetDirLevel(), 0),
        (bool(ROOT.gROOT.GetFile() is None), True),
        (bool(ROOT.gROOT.GetApplication() is None), True),
        (bool(not ROOT.gROOT.IsInterrupted()), True),
        (bool(ROOT.gROOT.GetSelectedPad() is None), True),
    )
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
    monkeypatch.setitem(sys.modules, "xrdroot.cint.execute", types.ModuleType("execute"))
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    try:
        expect(
            (ROOT.gROOT.ProcessLine("1+1"), 7),
            (ROOT.gROOT.ProcessLineSync("a"), 7),
            (ROOT.gROOT.ProcessLineFast("b"), 7),
            (ROOT.gROOT.Macro("m.C"), 7),
            (ROOT.gROOT.LoadMacro("m.C"), 0),
            (bool(ROOT.gInterpreter.Declare("int y;")), True),
            (ROOT.gInterpreter.ProcessLine("c"), 7),
            (ROOT.gInterpreter.Calc("d"), 7),
        )
    finally:
        troot.set_line_processor(None)
    expect(
        (seen, ["1+1", "a", "b", ".x m.C", ".L m.C", "int y;", "c", "d"]),
        (ROOT.gInterpreter.Load("lib"), 0),
        (ROOT.gInterpreter.GenerateDictionary("x"), 0),
    )
    ROOT.gInterpreter.AddIncludePath("-I.")


def test_a_declaration_puts_what_it_declares_in_the_namespace(monkeypatch):
    def declared():
        return 42

    declared.__module__ = "__cint__"
    execute = types.ModuleType("xrdroot.cint.execute")
    execute.run_source = lambda code, file, call: {"answer": declared, "_hidden": declared, "np": 1}
    execute.process_line = lambda text: 7
    monkeypatch.setitem(sys.modules, "xrdroot.cint", types.ModuleType("xrdroot.cint"))
    monkeypatch.setitem(sys.modules, "xrdroot.cint.execute", execute)
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    monkeypatch.setattr(troot, "_PRELUDE", {})  # the translator's own names, learnt once
    try:
        expect(
            (bool(ROOT.gInterpreter.Declare("int answer() { return 42; }")), True),
            (bool(ROOT.gInterpreter.Declare("int again() { return 42; }")), True),
            (ROOT.answer(), 42),
            (bool("_hidden" not in ROOT.__dict__), True),
            (ROOT.gROOT.ProcessLine("x"), 7),
        )
    finally:
        troot.set_line_processor(None)
        ROOT.__dict__.pop("answer", None)


def test_the_translator_is_not_looked_for_without_it(monkeypatch):
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: None)
    assert troot._translator("run_source") is None


def test_a_translator_module_without_process_line_is_still_refused(monkeypatch):
    monkeypatch.setitem(sys.modules, "xrdroot.cint", types.ModuleType("xrdroot.cint"))
    monkeypatch.setitem(sys.modules, "xrdroot.cint.execute", types.ModuleType("execute"))
    monkeypatch.setattr(troot.importlib.util, "find_spec", lambda name: True)
    with pytest.raises(UnsupportedFeatureError):
        ROOT.gROOT.ProcessLine("1")


def test_memory_directories_nest_and_are_walked(capsys):
    top = ROOT.gROOT.mkdir("a/b", "the b")
    expect(
        (top.GetName(), "b"),
        (top.GetPath(), "PyROOT:/a/b"),
        (bool(ROOT.gROOT.mkdir("a") is None), True),
        (ROOT.gROOT.mkdir("a", "", True).GetName(), "a"),
        (bool(ROOT.gROOT.GetDirectory("a/b") is top), True),
        (bool(ROOT.gROOT.GetDirectory("a/../a/./b") is top), True),
        (bool(ROOT.gROOT.GetDirectory("zz") is None), True),
        (bool(ROOT.gROOT.GetDirectory("a/zz/q") is None), True),
        (bool(top.cd()), True),
        (ROOT.gDirectory.GetName(), "b"),
        (top.GetMother().GetName(), "a"),
    )
    ROOT.gDirectory.pwd()
    expect(
        (bool(ROOT.gROOT.cd("a")), True),
        (bool(not ROOT.gROOT.cd("zz")), True),
    )
    out = capsys.readouterr()
    expect(
        (out.out, "PyROOT:/a/b\n"),
        (bool("Unknown directory zz" in out.err), True),
        (top.GetMotherDir().GetName(), "a"),
        (bool(top.GetFile() is None), True),
        (bool(not top.IsWritable()), True),
    )


def test_a_directory_keeps_what_is_made_in_it_and_replaces_by_name(capsys):
    ROOT.gROOT.mkdir("d").cd()
    first = ROOT.TH1D("h", "", 1, 0, 1)
    second = ROOT.TH1D("h", "", 1, 0, 1)
    here = ROOT.gDirectory.__real__()
    expect(
        (bool(here.Get("h") is second), True),
        (bool(here.Get("h;1") is second), True),
        (bool(ROOT.gROOT.Get("d/h") is second), True),
        (bool(ROOT.gROOT.Get("zz/h") is None), True),
        (
            bool("Replacing existing TH1D: h (Potential memory leak)." in capsys.readouterr().err),
            True,
        ),
        (bool(first.GetDirectory() is None), True),
        (bool(second.GetDirectory() is here), True),
    )
    here.Add(ROOT.TNamed("n", ""))
    expect(
        (ROOT.gROOT.FindObjectAny("n").GetName(), "n"),
        (here.GetObject("n").GetName(), "n"),
    )
    here.Delete("n")
    expect(
        (bool(here.FindObject("n") is None), True),
        (len(here.GetList()), 1),
    )
    here.ReadAll()
    here.SaveSelf()
    here.DeleteAll()
    expect(
        (bool(here.GetList().IsEmpty()), True),
        (bool(here.GetListOfKeys().IsEmpty()), True),
    )


def test_a_memory_directory_lists_its_objects_one_level_in(capsys):
    ROOT.TH1D("h1", "one", 1, 0, 1)
    ROOT.TH1D("x2", "two", 1, 0, 1)
    ROOT.gROOT.ls()
    ROOT.gDirectory.ls("-mh*")
    ROOT.gROOT.ls("noaddr")
    lines = capsys.readouterr().out.splitlines()
    expect(
        (bool(lines[0].startswith(" OBJ: TH1D\th1\tone : 0 at: 0x")), True),
        (bool(lines[1].startswith(" OBJ: TH1D\tx2")), True),
        (len(lines), 3),
        (bool(lines[2].startswith(" OBJ: TH1D\th1")), True),
    )


def test_writing_a_memory_directory_writes_nothing(capsys):
    ROOT.TH1D("h", "", 1, 0, 1)
    expect(
        (ROOT.gROOT.Write(), 0),
        (ROOT.gROOT.WriteObject(ROOT.TNamed("n", ""), "m"), 0),
        (bool("The object (m) has not been written" in capsys.readouterr().err), True),
    )
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
