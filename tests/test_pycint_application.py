"""``gApplication`` and the ROOT namespace's library loading, with and without the translator."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyrootsupport import fresh
from xrdroot import pyroot as ROOT
from xrdroot.pycint.execute import called, process_line
from xrdroot.pycint.library import loading
from xrdroot.pycint.runtime import ROOT as HOST
from xrdroot.pycint.runtime.application import TApplication, gApplication, started
from xrdroot.pycxx import cache
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    monkeypatch.setattr(ROOT.gSystem, "_libraries", [])
    saved, names = dict(DECLARED), set(vars(ROOT))
    loading.forget_all()
    try:
        yield from fresh(tmp_path)
    finally:
        loading.forget_all()
        DECLARED.clear()
        DECLARED.update(saved)
        troot.set_line_processor(None)  # as a fresh namespace has it: found when first needed
        for name in set(vars(ROOT)) - names:  # what gInterpreter.Declare made ROOT's
            if getattr(getattr(ROOT, name), "__module__", None) == "__cint__":
                delattr(ROOT, name)


def test_g_application_is_the_t_rint_root_runs_a_macro_in() -> None:
    with started(["root.exe", "-b", "m.C"]) as application:
        assert application is gApplication is HOST.gApplication
        assert (application.GetName(), application.ClassName()) == ("TRint", "TRint")
        assert (application.Argc(), application.Argv(2), application.Argv()) == (
            3, "m.C", ["root.exe", "-b", "m.C"])  # fmt: skip
        assert application.IsRunning() and ROOT.gROOT.GetName() == "Rint"
        application.SetReturnFromRun(True)
        application.Run()
        application.InitializeGraphics()
        assert application.ProcessLine("6 * 7") == 42
    assert ROOT.gROOT.GetName() == "PyROOT"
    assert HOST.TApplication is HOST.TRint is TApplication
    assert TApplication().Argv() == ["root.exe"]
    with pytest.raises(SystemExit) as leaving:
        gApplication.Terminate(3)
    assert leaving.value.code == 3


def test_what_is_no_macro_call_is_refused() -> None:
    with pytest.raises(ValueError, match="does not name a macro"):
        called("")


def test_without_the_translator_a_library_loads_as_nothing_and_a_macro_compiles_as_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module in ("xrdroot.pycint.library.loading", "xrdroot.pycint.library.aclic"):
        monkeypatch.setitem(sys.modules, module, None)  # not installed
    assert ROOT.gSystem.Load("libAnything") == 0
    assert ROOT.gSystem.GetLibraries() == "libAnything"
    assert ROOT.gSystem.CompileMacro("m.C", "k") == 1
    with pytest.raises(AttributeError, match="ROOT has NotDeclared"):
        _ = ROOT.NotDeclared
    ROOT.gSystem.Unload("libAnything")
    assert ROOT.gSystem.GetLibraries() == ""


def test_what_cpp_declared_is_a_name_of_the_root_namespace() -> None:
    process_line(".L " + str(_write("decl.C", "int declared_here() { return 11; }")))
    assert ROOT.declared_here() == 11
    assert (ROOT.gInterpreter.Load("libNope"), ROOT.gInterpreter.Load("libTree")) == (-1, 0)


def _write(name: str, text: str) -> Path:
    path = Path(name)
    path.write_text(text)
    return path


def test_a_library_unloaded_loads_anew_and_autoloading_can_be_turned_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrdroot.pycint.library import build_dictionary, build_library

    assert ROOT.gSystem.Load("libPhysics") == 0
    ROOT.gSystem.Unload("libPhysics.so")
    assert ROOT.gSystem.Load("libPhysics") == 0
    _write("A.h", "int fromA() { return 5; }")
    build_library("A", [Path("A.h")])
    assert ROOT.gSystem.Load("libA") == 0
    ROOT.gSystem.Unload("libA")
    assert "libA" not in ROOT.gSystem.GetLibraries()
    _write("B.h", "int fromB() { return 6; }")
    build_dictionary("B", [Path("B.h")])
    interpreter = ROOT.gInterpreter
    assert interpreter.SetClassAutoloading(0) == 1 and interpreter.SetClassAutoLoading(0) == 0
    with pytest.raises(AttributeError):
        _ = ROOT.fromB
    interpreter.SetClassAutoloading(1)
    assert interpreter.AutoLoad("fromB") == 1 and ROOT.fromB() == 6
    assert (interpreter.AutoParse("x"), interpreter.SetClassAutoparsing(0)) == (0, 1)
    assert (interpreter.LoadLibraryMap(), interpreter.UnloadLibraryMap("libB")) == (0, 0)
    monkeypatch.setitem(sys.modules, "xrdroot.pycint.library.loading", None)
    assert interpreter.AutoLoad("fromB") == 0


def test_the_command_aclic_would_build_with_is_kept_for_a_macro_to_edit() -> None:
    before = ROOT.gSystem.GetMakeSharedLib()
    assert "$SharedLib" in before
    ROOT.gSystem.SetMakeSharedLib(before + " -lm")
    assert ROOT.gSystem.GetMakeSharedLib().endswith(" -lm")
    ROOT.gSystem.SetMakeSharedLib(before)


def test_a_function_named_like_a_python_builtin_is_found_by_its_cpp_name() -> None:
    process_line(".L " + str(_write("comp.C", "int compile(int n) { return n + 1; }")))
    assert ROOT.compile(1) == 2
    _write("use.C", "int use() { return compile(2); }")
    assert process_line(".x use.C") == 3
