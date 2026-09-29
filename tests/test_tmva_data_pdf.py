"""TMVA's PDFs: a histogram smoothed, splined, sampled, read back and written to XML."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from tmvasupport import session
from xrdroot.tmva import TMVAError, hists
from xrdroot.tmva.options import Options
from xrdroot.tmva.pdf import PDF, PDFSettings, _passes, _runs, pdf_from_xml, settings, spline2
from xrdroot.tmva.xmlfile import Node, child

__all__ = ["session"]


def _histogram(count: int = 2000, nbins: int = 40, seed: int = 1, weights: bool = False) -> object:
    """A TH1F of ``count`` Gaussian values over [-4, 4]."""
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, count)
    weight = rng.uniform(0.5, 1.5, count) if weights else None
    return hists.filled("h", "h", (nbins, -4.0, 4.0), values, weight)


@pytest.mark.parametrize("method", ["Spline0", "Spline1", "Spline2"])
def test_a_density_integrates_to_one_whichever_spline_it_is_read_through(method):
    pdf = PDF("pdf", PDFSettings(nsmooth=2, interpolation=method)).build(_histogram())
    assert pdf.integral() == pytest.approx(1.0, rel=1e-4)
    assert pdf.value([0.0])[0] > pdf.value([3.0])[0] > 0
    assert pdf.integral_between(pdf.xmin, pdf.xmax) == pytest.approx(1.0, rel=1e-3)
    assert 0.0 <= pdf.integral_between(0.0, 0.0) < 0.1


def test_the_integral_between_two_points_is_nothing_when_they_are_the_wrong_way_round():
    pdf = PDF("pdf", PDFSettings()).build(_histogram())
    assert pdf.integral_between(2.0, -2.0) == 0.0
    assert 0.4 < pdf.integral_between(-9.0, 0.0) < 0.6


def test_an_interpolation_xrdroot_does_not_have_is_refused(capsys):
    with pytest.raises(TMVAError, match="PDFInterpol=KDE is a TMVA interpolation"):
        PDF("pdf", PDFSettings(interpolation="KDE")).build(_histogram())


def test_a_least_smoothing_above_the_most_is_refused(capsys):
    with pytest.raises(TMVAError, match="minnsmooth=5, maxnsmooth=2"):
        PDF("pdf", PDFSettings(min_nsmooth=5, max_nsmooth=2)).build(_histogram())


def test_a_histogram_of_negative_weight_has_no_density(capsys):
    histogram = hists.filled("h", "h", (4, 0.0, 4.0), [0.5, 1.5, 2.5], [-1.0, -1.0, -1.0])
    with pytest.raises(TMVAError, match="Integral: -3 <= 0"):
        PDF("pdf", PDFSettings(interpolation="Spline0")).build(histogram)


def test_a_spread_of_smoothing_passes_smooths_the_noisier_bins_more():
    spec = PDFSettings(min_nsmooth=1, max_nsmooth=4)
    assert spec.smoothing() == (1, 4)
    pdf = PDF("pdf", spec).build(_histogram(count=500, weights=True))
    assert pdf.integral() == pytest.approx(1.0, rel=1e-4)
    assert _passes(_histogram(count=500), 1, 4) >= 2


def test_a_smoothing_spread_over_even_errors_smooths_every_bin_the_most():
    booked = hists.book("h", "h", 4, 0, 4)
    histogram = hists.set_bins(booked, [0, 4, 4, 4, 4, 0], [0, 1, 1, 1, 1, 0])
    assert _passes(histogram, 1, 3) == 2
    assert _passes(hists.book("e", "e", 4, 0, 4), 0, 2) == 1


def test_runs_of_marked_bins_are_counted_only_when_they_end_before_the_last():
    assert _runs([True, True, False, True, False, True, True]) == 1
    assert _runs([False, True, True, True, False, True, True, False]) == 2


def test_the_reference_histogram_has_the_bins_asked_for_or_so_many_events_a_bin():
    assert PDFSettings(nbins=30).hist_bins(1000) == 30
    assert PDFSettings(nbins=30, interpolation="KDE").hist_bins(1000) == 150
    assert PDFSettings(avg_per_bin=20).hist_bins(1000) == 50


def test_pdf_options_are_read_by_their_suffix_over_a_base():
    options = Options("NSmoothMVAPdf=3:PDFInterpolMVAPdf=Spline1:CheckHistMVAPdf:NbinsSig[1]=7")
    found = settings(options, "MVAPdf")
    assert (found.nsmooth, found.interpolation, found.check_hist) == (3, "Spline1", True)
    assert settings(options, "Sig[1]", found).nbins == 7


def test_a_spline_through_two_points_is_the_nearest_point():
    assert list(spline2(np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([0.2, 0.9]))) == [
        2.0,
        2.0,
    ]


@pytest.mark.parametrize("edges", [None, [-4.0, -1.0, 0.0, 0.5, 4.0]])
def test_a_density_written_to_xml_is_rebuilt_from_its_original_histogram(edges):
    rng = np.random.default_rng(2)
    if edges is None:
        histogram = _histogram()
    else:
        histogram = hists.book_edges("h", "h", edges)
        histogram.fill(rng.normal(0, 1, 500))
    pdf = PDF("pdf", PDFSettings(nsmooth=1, interpolation="Spline1")).build(histogram)
    parent = Node("Holder")
    pdf.add_xml(parent)
    read = pdf_from_xml(child(ET.fromstring("\n".join(parent.lines())), "PDF"))
    assert read.spec.interpolation == "Spline1" and read.smoothed.name == "h_smoothed"
    assert np.allclose(read.value([-1.0, 0.2, 1.0]), pdf.value([-1.0, 0.2, 1.0]))
