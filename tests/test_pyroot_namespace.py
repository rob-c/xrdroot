"""``import xrdroot.pyroot as ROOT``: the namespace, ROOT's enumerations and printf's formats."""

from __future__ import annotations

import importlib

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.core import cformat


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_the_namespace_holds_roots_classes_by_their_names():
    for name in ("TH1F", "TFile", "TGraph", "TF1", "TMath", "Math", "gROOT", "gRandom", "kRed"):
        assert name in ROOT.__all__
        assert hasattr(ROOT, name)


def test_a_name_root_has_and_this_does_not_is_refused_by_that_name():
    with pytest.raises(AttributeError, match=r"TNoSuchThing; xrdroot\.pyroot does not yet"):
        ROOT.TNoSuchThing  # noqa: B018


def test_a_listed_module_that_is_not_installed_is_passed_over(monkeypatch):
    monkeypatch.setattr(ROOT, "SUBMODULES", ["core", "no_such_module"])
    namespace: dict = {}
    names = ROOT._gather(namespace)
    assert "TH1D" in names
    assert "TH1D" in namespace


def test_the_core_gathers_every_family_again_the_same():
    core = importlib.import_module("xrdroot.pyroot.core")
    namespace: dict = {}
    assert set(core._gather(namespace)) <= set(core.__all__)


def test_colours_are_ints_and_add_as_roots_do():
    assert ROOT.kRed + 2 == 634
    assert ROOT.kBlue - 9 == 591
    assert ROOT.kOrange == 800
    assert ROOT.kWhite == 0
    assert ROOT.kTRUE is True
    assert ROOT.kFALSE is False


def test_the_styles_and_palettes_are_roots_numbers():
    assert (ROOT.kSolid, ROOT.kDashed, ROOT.kDotted) == (1, 2, 3)
    assert ROOT.kFullCircle == 20
    assert ROOT.kOpenSquare == 25
    assert ROOT.kBird == 57
    assert ROOT.kRainbow == ROOT.kRainBow == 55
    assert ROOT.kCividis == 113
    assert ROOT.kFSolid == 1
    assert ROOT.kFHatched1 == 3004
    assert ROOT.kCanDelete == 1
    assert ROOT.kOverwrite == 2


def test_form_fills_a_format_as_printf_does():
    assert ROOT.Form("h%d", 3) == "h3"
    assert ROOT.Form("%5.2lf|%lld|%lu|%zu", 3.14159, 7, 8, 9) == " 3.14|7|8|9"
    assert ROOT.Form("%s=%g", ROOT.TString("x"), 0.5) == "x=0.5"
    assert ROOT.Form("%d%%", True) == "1%"
    assert ROOT.Form("plain") == "plain"
    assert ROOT.Form("%p", 255) == "0xff"


def test_printf_prints_without_a_newline_of_its_own(capsys):
    cformat.printf("a=%d\n", 1)
    cformat.printf("b")
    assert capsys.readouterr().out == "a=1\nb"


def test_printf_with_its_newline_is_rooot_printf(capsys):
    ROOT.Printf("%s and %d", "x", 2)
    assert capsys.readouterr().out == "x and 2\n"


def test_messages_go_to_standard_error_in_roots_words(capsys):
    ROOT.Info("here", "value %d", 3)
    ROOT.Warning("There::fn", "careful")
    ROOT.Error("", "plain %s", "text")
    ROOT.SysError("sys", "oops")
    ROOT.Break("brk", "stop")
    err = capsys.readouterr().err.splitlines()
    assert err == [
        "Info in <here>: value 3",
        "Warning in <There::fn>: careful",
        "Error: plain text",
        "SysError in <sys>: oops",
        "Break in <brk>: stop",
    ]


def test_messages_below_the_ignore_level_are_not_printed(capsys):
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
    ROOT.Info("here", "hidden")
    ROOT.Warning("here", "shown")
    assert capsys.readouterr().err == "Warning in <here>: shown\n"


def test_a_fatal_message_ends_the_program(capsys):
    with pytest.raises(SystemExit):
        ROOT.Fatal("f", "the end")
    assert "Fatal in <f>: the end" in capsys.readouterr().err


def test_ownership_and_addresses_are_what_python_keeps():
    thing = ROOT.TNamed("n", "t")
    ROOT.SetOwnership(thing, False)
    assert ROOT.addressof(thing) == id(thing)
    assert ROOT.nullptr is None
