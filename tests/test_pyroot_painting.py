"""What painting a live pad adds, as ROOT's does: titles, stats boxes, fit boxes, legends."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootgraphics import Wrapped, fresh_session, gaussian, graph  # noqa: F401
from xrdroot import Function, Histogram
from xrdroot.pyroot.graphics import decorations, snapshot


def _classes(pad):
    return [getattr(obj, "classname", type(obj).__name__) for obj, _ in snapshot.prepare(pad)]


def _stats_lines(pad):
    box = pad.GetPrimitive("stats")
    return [line.GetTitle() for line in box.GetListOfLines()]


def test_a_histogram_is_painted_with_a_frame_title_and_stats_box_from_gstyle():
    c = ROOT.TCanvas("c", "c")
    gaussian().Draw()
    assert _classes(c) == ["TFrame", "TH1D", "TPaveText", "TPaveStats"]
    c.Update()
    assert _stats_lines(c)[:2] == ["h", "Entries = 1000"]
    assert [line.split(" =")[0] for line in _stats_lines(c)] == ["h", "Entries", "Mean", "Std Dev"]
    title = c.GetPrimitive("title")
    assert title.GetListOfLines()[0].GetTitle() == "A Gaussian"
    x1, x2 = title.GetX1NDC(), title.GetX2NDC()
    assert (x1 + x2) / 2 == pytest.approx(0.5) and title.GetY2NDC() == pytest.approx(0.995)
    stats = c.GetPrimitive("stats")
    assert (stats.GetX2NDC(), stats.GetY2NDC()) == (0.98, 0.935)
    assert stats.GetY1NDC() == pytest.approx(0.935 - 0.16)


def test_opt_stat_and_opt_title_say_what_is_painted():
    c = ROOT.TCanvas("c", "c")
    ROOT.gStyle.SetOptStat("nemrou")
    ROOT.gStyle.SetOptTitle(0)
    gaussian().Draw()
    c.Update()
    assert len(_stats_lines(c)) == 6 and c.GetPrimitive("title") is None
    ROOT.gStyle.SetOptStat(0)
    c.Update()
    assert c.GetPrimitive("stats") is None
    ROOT.gStyle.SetOptStat(1)  # 1 is ROOT's shorthand for 1111
    c.Update()
    assert len(_stats_lines(c)) == 4


def test_a_stats_box_keeps_its_place_and_its_own_options_across_updates():
    c = ROOT.TCanvas("c", "c")
    gaussian().Draw()
    c.Update()
    stats = c.GetPrimitive("stats")
    stats.SetX1NDC(0.1)
    stats.SetOptStat(11)
    c.Update()
    assert c.GetPrimitive("stats") is stats and stats.GetX1NDC() == 0.1
    assert _stats_lines(c) == ["h", "Entries = 1000"]


def test_histograms_drawn_same_have_no_stats_box_and_sames_have_theirs():
    c = ROOT.TCanvas("c", "c")
    gaussian("a").Draw()
    gaussian("b").Draw("same")
    assert _classes(c).count("TPaveStats") == 1
    gaussian("c").Draw("sames")
    assert _classes(c).count("TPaveStats") == 2


def test_a_fitted_histogram_describes_its_fit_as_opt_fit_asks_in_a_wider_box():
    c = ROOT.TCanvas("c", "c")
    wrapped = gaussian()
    h = wrapped._xrd
    fitted = Function("f", "pol0", range=(-4.0, 4.0), parameters=[25.0])
    fitted.fit_result = {"chi2": 30.0, "ndf": 39, "npfits": 40}
    h.attach(fitted)
    ROOT.gStyle.SetOptFit(1)
    wrapped.Draw()
    c.Update()
    lines = _stats_lines(c)
    assert "#chi^{2} / ndf = 30 / 39" in lines and any(line.startswith("p0") for line in lines)
    stats = c.GetPrimitive("stats")
    assert stats.GetX2NDC() - stats.GetX1NDC() == pytest.approx(0.36)


def test_a_fitted_graph_has_a_fit_box_and_an_unfitted_one_none():
    c = ROOT.TCanvas("c", "c")
    g = graph()
    g.Draw("AP")
    assert "TPaveStats" not in _classes(c)
    fitted = Function("f", "pol1", range=(0.0, 5.0), parameters=[1.0, 1.0])
    fitted.fit_result = {"chi2": 1.0, "ndf": 2, "npfits": 4}
    g._xrd.functions.append(fitted)
    ROOT.gStyle.SetOptFit(111)
    assert "TPaveStats" in _classes(c)


def test_a_graph_alone_draws_its_axes_and_its_title():
    c = ROOT.TCanvas("c", "c")
    graph().Draw("P")
    made = snapshot.prepare(c)
    assert made[1][1] == "AP" and made[2][0].classname == "TPaveText"


def test_the_stats_box_is_sized_by_its_font_size_when_it_has_one():
    ROOT.gStyle.SetStatFontSize(0.03)
    c = ROOT.TCanvas("c", "c")
    gaussian().Draw()
    c.Update()
    assert c.GetPrimitive("stats").GetY1NDC() == pytest.approx(0.935 - 4 * 0.03)


def test_a_title_is_aligned_as_gstyle_says():
    ROOT.gStyle.SetTitleAlign(13)
    ROOT.gStyle.SetTitleX(0.1)
    ROOT.gStyle.SetTitleFontSize(0.0)
    c = ROOT.TCanvas("c", "c")
    c.Divide(2)
    c.cd(1)
    gaussian().Draw()
    c.cd(1).Update()
    title = c.cd(1).GetPrimitive("title")
    assert title.GetX1NDC() == 0.1 and title.GetY2NDC() == pytest.approx(0.995)
    assert title.GetY2NDC() - title.GetY1NDC() == pytest.approx(0.05)
    assert decorations._aligned(0.5, 0.5, 0.2, 0.2) == (0.5, 0.3)


def test_a_legend_holds_entries_for_objects_names_and_headers():
    c = ROOT.TCanvas("c", "c")
    h = gaussian()
    h.Draw()
    legend = ROOT.TLegend(0.6, 0.7, 0.9, 0.9, "Header")
    first = legend.AddEntry(h, "data", "lep")
    by_name = legend.AddEntry("h")
    missing = legend.AddEntry("nothing", "just a label", "")
    assert legend.GetHeader() == "Header" and legend.GetNRows() == 4
    assert (first.GetLabel(), first.GetOption(), first.GetObject()) == ("data", "lep", h)
    assert (by_name.GetLabel(), by_name.GetObject()) == ("A Gaussian", h)
    assert missing.GetObject() is None
    legend.SetHeader("New", "C")
    assert legend.GetHeader() == "New" and len(legend.GetListOfPrimitives()) == 4
    legend.SetNColumns(2)
    assert legend.GetNRows() == 2 and legend.GetBorderSize() == 1
    legend.Draw()
    model = snapshot.primitive(legend)
    assert model.get("fPrimitives")[1].get("fObject") is h._xrd
    snapshot.model(c).plot()
    legend.DeleteEntry()
    legend.Clear()
    legend.DeleteEntry()
    assert legend.GetListOfPrimitives() == [] and legend.GetHeader() == ""


def test_a_legend_entry_is_restyled_and_relabelled():
    entry = ROOT.TLegendEntry(None, "x", "l")
    entry.SetLabel("y")
    entry.SetOption()
    entry.SetObject("o")
    entry.SetLineColor(2)
    assert (entry.GetLabel(), entry.GetOption(), entry.GetObject()) == ("y", "lpf", "o")


def test_a_legend_made_by_size_or_not_at_all_goes_at_the_top_right():
    assert ROOT.TLegend().GetX2NDC() == pytest.approx(0.88)
    small = ROOT.TLegend(0.2, 0.1, "head")
    assert (small.GetX1NDC(), small.GetY1NDC()) == pytest.approx((0.68, 0.78))
    assert small.GetHeader() == "head"


def test_build_legend_lists_what_the_pad_draws():
    c = ROOT.TCanvas("c", "c")
    gaussian("a").Draw("hist")
    gaussian("b").Draw("same p")
    graph().Draw("l same")
    legend = c.BuildLegend()
    assert [e.GetOption() for e in legend.GetListOfPrimitives()] == ["lf", "lp", "l"]
    assert legend.GetX1NDC() == 0.5
    placed = c.BuildLegend(0.1, 0.1, 0.3, 0.3, "t", "f")
    assert placed.GetHeader() == "t" and placed.GetListOfPrimitives()[1].GetOption() == "f"


def test_draw_frame_frames_a_pad_with_an_empty_histogram():
    c = ROOT.TCanvas("c", "c")
    frame = c.DrawFrame(0, -1, 10, 1, "Frame;x [cm];y")
    assert (c.GetUxmin(), c.GetUymin(), c.GetUxmax(), c.GetUymax()) == (0, -1, 10, 1)
    assert frame.GetName() == "hframe" and frame.GetTitle() == "Frame"
    assert frame.GetXaxis().GetTitle() == "x [cm]"
    frame.GetYaxis().SetTitle("counts")
    assert "TPaveStats" not in _classes(c)


def test_draw_frame_uses_the_core_th1f_when_there_is_one(monkeypatch):
    from xrdroot.pyroot import core

    made = []

    class TH1F(Wrapped):
        def __init__(self, name, title, bins, low, high):
            super().__init__(Histogram.book(name, (bins, low, high), title=title))
            made.append(self)

        def SetMinimum(self, v):
            self._xrd._core["fMinimum"] = v

        def SetMaximum(self, v):
            self._xrd._core["fMaximum"] = v

        def SetStats(self, on):
            self.stats = on

    monkeypatch.setattr(core, "TH1F", TH1F, raising=False)
    c = ROOT.TCanvas("c", "c")
    assert c.DrawFrame(0, 0, 5, 5) is made[0] and made[0].stats == 0
    assert c.GetUymax() == 5


def test_colours_and_a_palette_made_in_the_session_go_with_the_canvas():
    c = ROOT.TCanvas("c", "c")
    h2 = Histogram.book("h2", (4, 0.0, 4.0), (4, 0.0, 4.0))
    h2.fill(np.array([0.5, 1.5]), np.array([0.5, 2.5]))
    Wrapped(h2).Draw("colz")
    assert len(snapshot.model(c).colors) == 0
    ROOT.gStyle.SetPalette(1)
    tables = snapshot.model(c).colors
    assert len(tables) == 2 and tables[1][0].get("fNumber") == 51
    colour = ROOT.TColor.GetColor(12, 34, 56)
    ROOT.gStyle.SetPalette(ROOT.kBird)
    (table,) = snapshot.model(c).colors
    assert table[-1].get("fNumber") == colour
    assert snapshot.model(c).palette().rgb(colour) == pytest.approx((12 / 255, 34 / 255, 56 / 255))
