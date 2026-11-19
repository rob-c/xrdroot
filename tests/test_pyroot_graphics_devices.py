"""``TPostScript``, ``TPDF`` and ``TSVG``: a pad written to a file by a device."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import fresh_session, gaussian  # noqa: F401
from xrdroot.pyroot.graphics import postscript


@pytest.mark.parametrize(
    "device, suffix", [(ROOT.TPostScript, "ps"), (ROOT.TPDF, "pdf"), (ROOT.TSVG, "svg")]
)
def test_a_device_writes_the_pad_it_was_opened_on_in_the_format_its_class_names(
    tmp_path, device, suffix
):
    """ROOT's file devices are the page the pad is drawn into; here they are its ``SaveAs``."""
    path = tmp_path / f"c.{suffix}"
    canvas = ROOT.TCanvas("c", "c", 300, 200)
    gaussian().Draw()
    canvas.cd()
    page = device(str(path), 111)
    canvas.Draw()
    assert page.GetName() == str(path)
    page.Close()
    assert path.stat().st_size > 0
    assert postscript.OPEN == []


def test_a_name_without_a_suffix_is_given_the_format_s(tmp_path):
    ROOT.TCanvas("c", "c", 300, 200)
    gaussian().Draw()
    ROOT.TPDF(str(tmp_path / "c")).Close()
    assert (tmp_path / "c.pdf").stat().st_size > 0


def test_an_unnamed_device_is_named_after_its_class(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ROOT.TCanvas("c", "c", 300, 200)
    ROOT.TPDF().Close()
    assert (tmp_path / "TPDF.pdf").stat().st_size > 0


def test_a_device_is_closed_once_and_a_second_close_writes_no_second_page(tmp_path):
    path = tmp_path / "c.ps"
    ROOT.TCanvas("c", "c", 300, 200)
    device = ROOT.TPostScript(str(path))
    device.Close()
    written = path.read_bytes()
    device.Close()
    assert path.read_bytes() == written


def test_a_device_opened_before_any_pad_exists_writes_the_pad_current_at_its_close(tmp_path):
    """A macro opens its device first, as in ROOT, and draws after; the last pad is the page."""
    path = tmp_path / "c.ps"
    device = ROOT.TPostScript(str(path))
    gaussian().Draw()
    device.Close()
    assert path.stat().st_size > 0


def test_a_device_opened_with_no_pad_and_closed_with_none_writes_nothing(tmp_path):
    path = tmp_path / "c.ps"
    ROOT.TPostScript(str(path)).Close()
    assert not path.exists()


def test_a_device_left_open_is_closed_when_the_session_ends(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ROOT.TCanvas("c", "c", 300, 200)
    gaussian().Draw()
    device = ROOT.TSVG("left.svg")
    assert postscript.OPEN == [device]
    postscript.close_devices()
    assert (tmp_path / "left.svg").stat().st_size > 0


def test_a_device_is_a_page_of_the_format_its_class_names(tmp_path):
    """What is written is the format, not a picture of another one: a check of the bytes."""
    csvg = tmp_path / "c.svg"
    ROOT.gStyle.SetOptStat(0)
    ROOT.TCanvas("c", "c", 200, 150)
    ROOT.TLine(0.1, 0.1, 0.9, 0.9).Draw()
    ROOT.TSVG(str(csvg), 111, True).Close()
    assert b"<svg" in csvg.read_bytes()[:400]
