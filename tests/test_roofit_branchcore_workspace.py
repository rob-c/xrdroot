"""The RooFit workspace's new corners: embedded data, generic objects, sets made and removed.

The ``Print`` of a workspace holding a ``RooHistFunc``'s dataset, a
histogram and a constant parameter with an error, and the answers of
``defineSet``, ``removeSet``, ``argSet`` and ``embeddedData``, as ROOT 6.40
gave them for the same calls through PyROOT.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from xrdroot.roofit.collections import RooArgSet
from xrdroot.roofit.data.datahist import RooDataHist
from xrdroot.roofit.messages import service
from xrdroot.roofit.pdfs.histpdf import RooHistFunc
from xrdroot.roofit.variables import RooRealVar
from xrdroot.roofit.workspace import RooWorkspace


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


class Histogram:
    """A ``TH1F`` as a workspace sees one: a named thing that can be cloned."""

    def __init__(self, name: str) -> None:
        self.name = name

    def GetName(self) -> str:
        return self.name

    def ClassName(self) -> str:
        return "TH1F"

    def Clone(self) -> Histogram:
        return Histogram(self.name)


class Note:
    """A named thing that cannot be cloned: kept as it is."""

    def GetName(self) -> str:
        return "note"

    def ClassName(self) -> str:
        return "TNamed"


def filled() -> tuple[RooWorkspace, Histogram, RooRealVar]:
    w = RooWorkspace("w", "w")
    x = RooRealVar("x", "x", 0, 10)
    x.setBins(2)
    w.Import(RooHistFunc("hf", "hf", RooArgSet(x), RooDataHist("dh", "dh", RooArgSet(x))))
    histogram = Histogram("h")
    w.Import(histogram)
    lumi = RooRealVar("Lumi", "Lumi", 1, 0, 2)
    lumi.setError(0.1)
    lumi.setConstant(True)
    w.Import(lumi)
    w.saveSnapshot("snap", RooArgSet(lumi))
    return w, histogram, lumi


def test_a_workspace_lists_embedded_data_and_generic_objects_as_root_does(capsys: Any) -> None:
    """The dataset a ``RooHistFunc`` holds and a histogram imported are listed in sections of
    their own; a constant parameter with an error prints as ``Lumi=1 +/- 0.1[C]``."""
    w, _, _ = filled()
    w.defineSet("t", "x,Lumi")
    w.SetName("w2")
    w.SetTitle("t2")
    capsys.readouterr()
    w.Print()
    assert capsys.readouterr().out.endswith(
        "embedded datasets (in pdfs and functions)\n"
        "-----------------------------------------\n"
        "RooDataHist::dh(x)\n"
        "\n"
        "parameter snapshots\n"
        "-------------------\n"
        "snap = (Lumi=1 +/- 0.1[C])\n"
        "\n"
        "named sets\n"
        "----------\n"
        "t:(x,Lumi)\n"
        "\n"
        "generic objects\n"
        "---------------\n"
        "TH1F::h\n"
        "\n"
    )
    assert (w.GetName(), w.GetTitle()) == ("w2", "t2")


def test_a_generic_object_is_imported_as_a_copy_when_it_can_be_cloned() -> None:
    """``import(TObject&)`` keeps a clone - or the object itself, if it has no ``Clone``."""
    w, histogram, _ = filled()
    note = Note()
    assert w.Import(note) is False
    assert w.obj("h") is not histogram and w.obj("h").GetName() == "h"
    assert w.obj("note") is note


def test_embedded_data_is_the_data_the_nodes_hold_not_the_workspaces_own() -> None:
    """``embeddedData("dh")`` is the ``RooHistFunc``'s dataset; ``data`` knows nothing of it."""
    w, _, _ = filled()
    assert w.embeddedData("dh").GetName() == "dh"
    assert w.data("dh") is None and w.embeddedData("nothing") is None


def test_define_set_imports_what_is_missing_and_remove_set_says_of_one_absent(
    capsys: Any,
) -> None:
    """``defineSet(name, set, true)`` imports the members first; ``removeSet`` of a set that
    is not there is ``true``, with ROOT's error."""
    w, _, _ = filled()
    m = RooRealVar("m", "m", 1, 0, 2)
    capsys.readouterr()
    assert w.defineSet("s", RooArgSet(m, w.var("x")), True) is False
    assert w.set("s").find("m") is not None and w.var("m") is not None
    assert (w.removeSet("s"), w.removeSet("s")) == (False, True)
    assert capsys.readouterr().out == (
        "[#1] INFO:ObjectHandling -- RooWorkspace::import(w) importing RooRealVar::m\n"
        "[#0] ERROR:InputArguments -- RooWorkspace::removeSet(w) ERROR a set with name s does "
        "not exist\n"
    )


def test_arg_set_is_the_named_nodes_of_the_workspace() -> None:
    """``argSet("x,Lumi")`` gathers the workspace's own copies of the nodes named."""
    w, _, _ = filled()
    found = w.argSet("x,Lumi")
    assert [one.GetName() for one in found] == ["x", "Lumi"]
    assert found.find("Lumi") is w.var("Lumi")
