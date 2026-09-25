"""What ``gROOT`` hands to the graphics part of the kit: styles, canvases, and names on pads."""

from __future__ import annotations

import sys
import types

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


@pytest.fixture
def graphics(monkeypatch):
    """A stand-in for ``xrdroot.pyroot.graphics``, with the names gROOT reaches for."""
    seen: list = []
    package = types.ModuleType("xrdroot.pyroot.graphics")
    package.set_style = seen.append
    package.get_style = lambda name: f"style {name}"
    pads = types.ModuleType("xrdroot.pyroot.graphics.pads")
    pads.CANVASES = [ROOT.TNamed("c1", "")]
    pads.find_anywhere = lambda name: f"found {name}"
    canvas = types.ModuleType("xrdroot.pyroot.graphics.canvas")
    canvas.default_canvas = lambda: "c1"
    for module in (package, pads, canvas):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    real = troot.importlib.util.find_spec
    monkeypatch.setattr(
        troot.importlib.util,
        "find_spec",
        lambda name: True if name.startswith("xrdroot.pyroot.graphics") else real(name),
    )
    return seen


@pytest.fixture
def no_graphics(monkeypatch):
    real = troot.importlib.util.find_spec
    monkeypatch.setattr(
        troot.importlib.util,
        "find_spec",
        lambda name: None if name.startswith("xrdroot.pyroot.graphics") else real(name),
    )


def test_styles_are_the_graphics_when_it_is_there(graphics):
    ROOT.gROOT.SetStyle("Plain")
    assert graphics == ["Plain"] and ROOT.gROOT.GetStyle("Plain") == "style Plain"


def test_canvases_are_the_graphics_list(graphics):
    listed = ROOT.gROOT.GetListOfCanvases()
    assert isinstance(listed, ROOT.TList) and listed.At(0).GetName() == "c1"
    assert ROOT.gROOT.MakeDefCanvas() == "c1"


def test_a_name_found_nowhere_else_is_looked_for_on_the_pads(graphics):
    assert ROOT.gROOT.FindObject("somewhere") == "found somewhere"
    h = ROOT.TH1D("h", "", 1, 0, 1)
    assert ROOT.gROOT.FindObject("h") is h


def test_a_graphics_package_without_a_name_is_as_good_as_none(graphics, monkeypatch):
    monkeypatch.delattr(sys.modules["xrdroot.pyroot.graphics"], "get_style")
    assert troot._graphics("get_style") is None and troot._graphics("pads", "nothing") is None


def test_without_the_graphics_the_lists_are_groots_own(no_graphics, capsys):
    assert ROOT.gROOT.GetListOfCanvases().IsEmpty() and ROOT.gROOT.FindObject("x") is None
    chosen = []
    style = ROOT.TNamed("Plain", "")
    style.cd = lambda: chosen.append("Plain")
    ROOT.gROOT.GetListOfStyles().Add(style)
    ROOT.gROOT.SetStyle("Plain")
    ROOT.gROOT.SetStyle("Nope")
    assert chosen == ["Plain"] and ROOT.gROOT.GetStyle("Plain") is style
    assert "Error in <TROOT::SetStyle>: Unknown style:Nope" in capsys.readouterr().err
    with pytest.raises(UnsupportedFeatureError, match="MakeDefCanvas"):
        ROOT.gROOT.MakeDefCanvas()
