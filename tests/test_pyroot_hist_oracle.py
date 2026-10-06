"""A histogram workflow - filling, statistics, arithmetic, projections - against ROOT 6.40.

Every value below was printed by ROOT itself running exactly this code, with
the same seed; the generator is ROOT's to the bit, so even the random draws agree.
"""

from __future__ import annotations

import array

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh

#: What ROOT 6.40.04 printed, a line per ``show``.
ROOTS = [
    "entries=5000",
    "mean=7.14488056813e-05",
    "std=0.982280958892",
    "rms=0.982280958892",
    "meanerr=0.0138915505413",
    "skew=-0.014660634331",
    "kurt=-0.0722616063784",
    "integral=5000",
    "integral_w=800",
    "integral_10_20=1028",
    "max=336",
    "maxbin=24",
    "min=0",
    "minbin=1",
    "bin25=333",
    "err25=18.2482875909",
    "center25=-0.08",
    "findbin=28",
    "sumw=5000",
    "eff=5000",
    "r_first=19",
    "r_last=38",
    "r_mean=0.191156716418",
    "r_std=0.751020801374",
    "r_integral=4288",
    "r_max=336",
    "r_maxbin=24",
    "iae=5000",
    "iae_err=70.7106781187",
    "scaled=666",
    "scaled_err=36.4965751818",
    "scaled_entries=5000",
    "added=333",
    "divided=1",
    "divided_err=1",
    "rebinned_n=10",
    "rebinned_5=1479",
    "set_entries=5001",
    "set_mean=0.191156716418",
    "set_err=2",
    "set_err4=0",
    "g_entries=8",
    "g_mean=4.175",
    "g_under=1",
    "g_over=1",
    "g_err5=3",
    "g_err3=1.41421356237",
    "quantiles=2.5,3.5,4.666666667",
    "smooth3=1",
    "cum_name=h_cumulative",
    "cum_last=0",
    "interp=276.625",
    "ks=0.99987089539",
    "chi2=0.970105255039",
    "random=0.183252898079",
    "h2_mean_x=0.470065211835",
    "h2_mean_y=0.326351062488",
    "h2_cov=-0.000857333270171",
    "h2_corr=-0.0100126738746",
    "h2_bin=51",
    "h2_content=1",
    "h2_integral=39",
    "px_name=h2_px",
    "px_entries=100",
    "px_3=14",
    "py_name=myy",
    "py_entries=40",
    "py_5=4",
    "pf_name=h2_pfx",
    "pf_3=0.407142857143",
    "pf_err3=0.0843427686521",
    "h2_maxbin=16",
    "rand2=0.6087149425,0.6185955582",
]


@pytest.fixture(autouse=True)
def _fresh(tmp_path, monkeypatch):
    """A fresh session - and no pad, as ROOT's script had none: ``UnZoom`` then does nothing."""
    from xrdroot.pyroot.graphics import pads

    monkeypatch.setattr(pads, "_CURRENT", [None])
    yield from fresh(tmp_path)


def workflow():
    out = []

    def show(name, value):
        if isinstance(value, float):
            value = f"{value:.12g}"
        out.append(f"{name}={value}")

    ROOT.gRandom.SetSeed(12345)
    h = ROOT.TH1F("h", "gauss;x;n", 50, -4, 4)
    h.FillRandom("gaus", 5000)
    show("entries", h.GetEntries())
    show("mean", h.GetMean())
    show("std", h.GetStdDev())
    show("rms", h.GetRMS())
    show("meanerr", h.GetMeanError())
    show("skew", h.GetSkewness())
    show("kurt", h.GetKurtosis())
    show("integral", h.Integral())
    show("integral_w", h.Integral("width"))
    show("integral_10_20", h.Integral(10, 20))
    show("max", h.GetMaximum())
    show("maxbin", h.GetMaximumBin())
    show("min", h.GetMinimum())
    show("minbin", h.GetMinimumBin())
    show("bin25", h.GetBinContent(25))
    show("err25", h.GetBinError(25))
    show("center25", h.GetBinCenter(25))
    show("findbin", h.FindBin(0.33))
    show("sumw", h.GetSumOfWeights())
    show("eff", h.GetEffectiveEntries())
    h.GetXaxis().SetRangeUser(-1, 2)
    show("r_first", h.GetXaxis().GetFirst())
    show("r_last", h.GetXaxis().GetLast())
    show("r_mean", h.GetMean())
    show("r_std", h.GetStdDev())
    show("r_integral", h.Integral())
    show("r_max", h.GetMaximum())
    show("r_maxbin", h.GetMaximumBin())
    h.GetXaxis().UnZoom()
    err = array.array("d", [0])
    show("iae", h.IntegralAndError(1, 50, err))
    show("iae_err", err[0])
    c = h.Clone("c")
    c.Scale(2.0)
    show("scaled", c.GetBinContent(25))
    show("scaled_err", c.GetBinError(25))
    show("scaled_entries", c.GetEntries())
    c.Add(h, -1.0)
    show("added", c.GetBinContent(25))
    d = h.Clone("d")
    d.Divide(h)
    show("divided", d.GetBinContent(25))
    show("divided_err", d.GetBinError(25))
    r = h.Clone("r")
    r.Rebin(5)
    show("rebinned_n", r.GetNbinsX())
    show("rebinned_5", r.GetBinContent(5))
    h.SetBinContent(3, 7.5)
    show("set_entries", h.GetEntries())
    show("set_mean", h.GetMean())
    h.SetBinError(3, 2.0)
    show("set_err", h.GetBinError(3))
    show("set_err4", h.GetBinError(4))
    g = ROOT.TH1D("g", "g", 10, 0, 10)
    for x in (1.5, 2.5, 2.5, 3.5, 9.9, 11.0, -1.0):
        g.Fill(x)
    g.Fill(4.5, 3.0)
    show("g_entries", g.GetEntries())
    show("g_mean", g.GetMean())
    show("g_under", g.GetBinContent(0))
    show("g_over", g.GetBinContent(11))
    show("g_err5", g.GetBinError(5))
    show("g_err3", g.GetBinError(3))
    q = array.array("d", [0, 0, 0])
    p = array.array("d", [0.25, 0.5, 0.75])
    g.GetQuantiles(3, q, p)
    show("quantiles", ",".join(f"{v:.10g}" for v in q))
    g.Smooth()
    show("smooth3", g.GetBinContent(3))
    cum = h.GetCumulative()
    show("cum_name", cum.GetName())
    show("cum_last", cum.GetBinContent(50))
    show("interp", h.Interpolate(0.1))
    k1 = ROOT.TH1D("k1", "", 20, -3, 3)
    k2 = ROOT.TH1D("k2", "", 20, -3, 3)
    k1.FillRandom("gaus", 1000)
    k2.FillRandom("gaus", 1000)
    show("ks", k1.KolmogorovTest(k2))
    show("chi2", k1.Chi2Test(k2, "UU"))
    show("random", h.GetRandom())
    h2 = ROOT.TH2D("h2", "2d", 10, 0, 1, 10, 0, 1)
    for _ in range(100):
        h2.Fill(ROOT.gRandom.Rndm(), ROOT.gRandom.Rndm() ** 2)
    show("h2_mean_x", h2.GetMean(1))
    show("h2_mean_y", h2.GetMean(2))
    show("h2_cov", h2.GetCovariance())
    show("h2_corr", h2.GetCorrelationFactor())
    show("h2_bin", h2.GetBin(3, 4))
    show("h2_content", h2.GetBinContent(3, 4))
    show("h2_integral", h2.Integral(1, 5, 1, 5))
    px = h2.ProjectionX()
    show("px_name", px.GetName())
    show("px_entries", px.GetEntries())
    show("px_3", px.GetBinContent(3))
    py = h2.ProjectionY("myy", 2, 4)
    show("py_name", py.GetName())
    show("py_entries", py.GetEntries())
    show("py_5", py.GetBinContent(5))
    pf = h2.ProfileX()
    show("pf_name", pf.GetName())
    show("pf_3", pf.GetBinContent(3))
    show("pf_err3", pf.GetBinError(3))
    show("h2_maxbin", h2.GetMaximumBin())
    x2, y2 = array.array("d", [0]), array.array("d", [0])
    h2.GetRandom2(x2, y2)
    show("rand2", f"{x2[0]:.10g},{y2[0]:.10g}")
    return out


def test_the_workflow_gives_every_answer_root_gives(capsys):
    found = workflow()
    for mine, roots in zip(found, ROOTS, strict=False):
        assert mine == roots
    assert len(found) == len(ROOTS)
    assert "Cannot UnZoom if gPad does not exist" in capsys.readouterr().err
