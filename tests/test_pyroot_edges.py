"""The corners of the session classes: stopwatches restarted, directories nested, and so on."""

from __future__ import annotations

import math

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pyroot.core import refs


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def test_a_running_stopwatch_started_again_keeps_running(capsys):
    watch = ROOT.TStopwatch()
    watch.Start(False)
    watch.Continue()
    watch.Stop()
    watch.Start()
    watch.Print()
    expect(
        (bool(capsys.readouterr().out.startswith("Real time 0:00:00, CP time ")), True),
        (watch.Counter(), 1),
    )


def test_appending_an_object_already_kept_says_nothing(capsys):
    h = ROOT.TH1D("h", "", 1, 0, 1)
    ROOT.gROOT.Append(h, True)
    expect(
        (capsys.readouterr().err, ""),
        (ROOT.gROOT.GetList().GetSize(), 1),
    )


def test_find_object_any_looks_into_every_directory_below():
    below = ROOT.gROOT.mkdir("a/b")
    below.Append(ROOT.TNamed("deep", ""))
    expect(
        (ROOT.gROOT.FindObjectAny("deep").GetName(), "deep"),
        (bool(ROOT.gROOT.GetDirectory("a").FindObjectAny("nothing") is None), True),
    )


def test_a_directory_standing_alone_is_its_own_top():
    assert ROOT.TDirectory("d", "t").GetPath() == "d:/"


def test_prepend_path_name_hands_back_a_plain_join():
    assert ROOT.gSystem.PrependPathName("d", "f") == "d/f"


def test_the_root_namespace_hands_other_names_to_the_top():
    expect(
        (bool(ROOT.ROOT.TH1F is ROOT.TH1F), True),
        (repr(ROOT.ROOT), "<namespace ROOT>"),
        (bool(ROOT.ROOT.GetThreadPoolSize() >= 1), True),
    )
    with pytest.raises(AttributeError):
        ROOT.ROOT.__wrapped__  # noqa: B018


def test_genvector_edges():
    expect(
        (ROOT.Math.RhoEtaPhiVector(ROOT.Math.XYZVector(1, 0, 0)).Rho(), pytest.approx(1)),
        (ROOT.Math.XYZVector(1, 2, 3).GetCoordinates(), (1, 2, 3)),
    )
    far = ROOT.Math.PtEtaPhiMVector(1, 0, 3.0, 0)
    near = ROOT.Math.PtEtaPhiMVector(1, 0, -3.0, 0)
    assert ROOT.Math.VectorUtil.DeltaPhi(far, near) == pytest.approx(2 * math.pi - 6)


def test_references_are_filled_however_they_were_made():
    import ctypes

    holder, cell, array = ctypes.c_double(0), type("Cell", (), {"value": 0})(), np.zeros(2)
    refs.store(holder, 1.5)
    refs.store(cell, 2.5)
    refs.store(array, 3.5)
    refs.store(None, 1.0)
    expect(
        ((holder.value, cell.value, array[0]), (1.5, 2.5, 3.5)),
        (refs.load(holder), 1.5),
        (refs.load(array), 3.5),
    )
    with pytest.raises(TypeError, match="not somewhere to put a number"):
        refs.store(3, 1.0)
