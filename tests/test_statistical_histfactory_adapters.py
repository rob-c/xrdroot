"""HistFactory delegates ROOT storage and ownership while retaining its configuration objects."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from statisticalproviders import install_provider
from xrdroot.histfactory import make, model
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.model import Sample


def test_counting_channels_and_samples_keep_the_histograms_the_provider_returns(monkeypatch):
    received = []

    def counted(name, value):
        received.append((name, value))
        return SimpleNamespace(GetName=lambda: name, value=value)

    install_provider(monkeypatch, "histfactory.count_data", counted)
    install_provider(monkeypatch, "histfactory.count_sample", counted)
    channel, sample = Channel("observed"), Sample("background")
    channel.SetData(7)
    sample.SetValue(2.5)
    assert received == [("observed_data", 7), ("background_hist", 2.5)]
    assert channel.GetData().GetHisto().value == 7
    assert sample.GetHisto().value == 2.5
    assert channel.GetData().GetHistoName() == "observed_data"
    assert sample.GetHistoName() == "background_hist"


def test_histogram_reading_and_workspace_writing_retain_their_explicit_context(monkeypatch):
    received = []
    histogram = object()

    def read(*args):
        received.append(args)
        return histogram

    def write(*args):
        received.append(args)

    install_provider(monkeypatch, "histfactory.read_histogram", read)
    install_provider(monkeypatch, "histfactory.write_model", write)
    files, measurement, workspace, channel = {}, object(), object(), object()
    assert model._histogram(files, "input.root", "region", "data") is histogram
    assert make._written(measurement, "output.root", workspace, channel) is None
    assert received == [
        (files, "input.root", "region", "data"),
        (measurement, "output.root", workspace, channel),
    ]
    assert received[0][0] is files


def test_the_root_counting_sample_keeps_float_narrowing_and_detached_ownership():
    from xrdroot.pyroot.core.hists import TH1F

    sample = Sample("background")
    sample.SetValue(1.23456789)
    histogram = sample.GetHisto()
    assert isinstance(histogram, TH1F)
    assert histogram.GetName() == "background_hist"
    assert histogram.GetBinContent(1) == float(np.float32(1.23456789))
    assert histogram.GetDirectory() is None
