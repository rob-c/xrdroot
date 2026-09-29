"""HistFactory reading its histograms: ``CollectHistograms`` and ``CheckHistograms``.

A channel reads every histogram its configuration names - the data, the additional data,
each sample's nominal, statistical error, histogram systematics, shape systematics and
initial shapes - opening each file once and saying so as ROOT's ``Channel::GetHistogram``
says it; a file that will not open and a histogram that is not there are errors, and
``CheckHistograms`` names the first histogram missing and warns of negative bins.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from histfactoryinputs import write_inputs
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.measurement import Measurement
from xrdroot.histfactory.model import Data, Sample
from xrdroot.histfactory.shapes import ShapeFactor
from xrdroot.histfactory.systematics import HistFactoryError
from xrdroot.roofit.messages import service


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def full_channel(path: str) -> Channel:
    made = Channel("channel1")
    made.SetData("data", path)
    extra = Data("data", path)
    extra.SetName("extra")
    made.AddAdditionalData(extra)
    made.AddAdditionalData(Data())  # no file: nothing to read
    signal = Sample("signal", "signal", path, "dir")
    signal.AddHistoSys("shape", "signal_low", path, "", "signal_high", path, "")
    signal.AddHistoFactor("free", "signal_low", path, "", "signal_high", path, "")
    signal.AddShapeSys("ss", "Gaussian", "shape_err", path, "")
    factor = ShapeFactor("initial")
    factor.SetHistoName("signal")
    factor.SetInputFile(path)
    signal.AddShapeFactor(factor)
    signal.AddShapeFactor("plain")
    made.AddSample(signal)
    background = Sample("background1", "background1", path)
    background.ActivateStatError("background1_statUncert", path)
    made.AddSample(background)
    return made


def test_a_channel_reads_every_histogram_it_names_opening_its_file_once(tmp_path: Any,
                                                                        capsys: Any) -> None:  # fmt: skip
    path = write_inputs(tmp_path)
    made = full_channel(path)
    made.CollectHistograms()
    out = capsys.readouterr().out
    assert out.count("Opened input file") == 1
    assert out.startswith(f"[#2] PROGRESS:HistFactory -- Getting histogram {path}:/data\n"
                          f"[#2] INFO:HistFactory -- Opened input file: {path}: \n")  # fmt: skip
    assert f"Getting histogram {path}:dir/signal\n" in out
    signal, background = made.GetSamples()
    assert signal.GetHisto().GetBinContent(1) == 20.0
    assert signal.GetHistoSysList()[0].GetHistoHigh().GetName() == "signal_high"
    assert signal.GetHistoFactorList()[0].GetHistoLow().GetName() == "signal_low"
    assert signal.GetShapeSysList()[0].GetErrorHist().GetBinContent(2) == 0.25
    assert signal.GetShapeFactorList()[0].GetInitialShape().GetName() == "signal"
    assert signal.GetShapeFactorList()[1].GetInitialShape() is None
    assert background.GetStatError().GetErrorHist().GetName() == "background1_statUncert"
    assert made.GetAdditionalData()[0].GetHisto().GetBinContent(2) == 112.0
    assert made.CheckHistograms() is True


def test_a_measurement_collects_every_channels_histograms(tmp_path: Any) -> None:
    path = write_inputs(tmp_path)
    meas = Measurement("m")
    meas.AddChannel(full_channel(path))
    meas.CollectHistograms()
    assert meas.GetChannel("channel1").GetData().GetHisto().GetBinContent(1) == 122.0


def test_a_file_that_will_not_open_is_an_error_said_as_root_says(tmp_path: Any,
                                                                 capsys: Any) -> None:  # fmt: skip
    made = Channel("c")
    missing = str(tmp_path / "missing.root")
    made.SetData("data", missing)
    with pytest.raises(HistFactoryError, match="cannot be opened"):
        made.CollectHistograms()
    assert f"[#2] ERROR:HistFactory -- Error: Unable to open input file: {missing}\n" in (
        capsys.readouterr().out)  # fmt: skip


def test_a_histogram_the_file_has_not_got_is_an_error_said_as_root_says(tmp_path: Any,
                                                                        capsys: Any) -> None:  # fmt: skip
    path = write_inputs(tmp_path)
    made = Channel("c")
    made.SetData("nothing", path, "dir")
    with pytest.raises(HistFactoryError, match="has no histogram dir/nothing"):
        made.CollectHistograms()
    assert (f"[#2] ERROR:HistFactory -- Histogram 'nothing' wasn't found in file '{path}' in "
            "directory 'dir'.\n") in capsys.readouterr().out  # fmt: skip


def unread(configure: Any) -> Channel:
    """A channel with one sample, configured by ``configure``, whose histograms are not read."""
    made = Channel("c")
    sample = Sample("s", "s", "f.root")
    configure(sample)
    made.AddSample(sample)
    return made


def with_nominal(sample: Sample) -> None:
    from histfactoryinputs import histogram

    sample.SetHisto(histogram([1.0, 2.0], "nominal"))


def stat_hist(sample: Sample) -> None:
    with_nominal(sample)
    sample.ActivateStatError("err", "f.root")


def histo_sys(side: str) -> Any:
    def configure(sample: Sample) -> None:
        from histfactoryinputs import histogram

        with_nominal(sample)
        sample.AddHistoSys("hs", "lo", "f.root", "", "hi", "f.root", "")
        if side == "High":
            sample.GetHistoSysList()[0].SetHistoLow(histogram([1.0, 1.0], "lo"))

    return configure


def shape_sys(sample: Sample) -> None:
    with_nominal(sample)
    sample.AddShapeSys("ss", 0, "err", "f.root")


@pytest.mark.parametrize(("configure", "said"), [
    (lambda s: None, "Error: Nominal Histogram for sample s is nullptr."),
    (stat_hist, "Error: Statistical Error Histogram for sample s is nullptr."),
    (histo_sys("Low"), "Error: HistoSyst Low for Systematic hs in sample s is nullptr."),
    (histo_sys("High"), "Error: HistoSyst High for Systematic hs in sample s is nullptr."),
    (shape_sys, "Error: HistoSyst High for Systematic ss in sample s is nullptr."),
])  # fmt: skip
def test_a_histogram_not_read_is_named_by_check_histograms(configure: Any, said: str,
                                                           capsys: Any) -> None:  # fmt: skip
    assert unread(configure).CheckHistograms() is False
    assert capsys.readouterr().out == f"[#2] ERROR:HistFactory -- {said}\n"


def test_data_not_read_from_its_file_is_named_by_check_histograms(capsys: Any) -> None:
    made = Channel("c")
    made.SetData("data", "f.root")
    assert made.CheckHistograms() is False
    assert capsys.readouterr().out == (
        "[#2] ERROR:HistFactory -- Error: Data Histogram for channel c is nullptr.\n"
    )


def test_negative_bins_of_a_nominal_histogram_are_warned_of_bin_by_bin(capsys: Any) -> None:
    from histfactoryinputs import histogram

    made = unread(lambda s: s.SetHisto(histogram([-1.0, 2.0, -0.5], "neg")))
    assert made.CheckHistograms() is True
    assert capsys.readouterr().out == (
        "[#2] WARNING:HistFactory -- WARNING: Nominal Histogram neg for Sample = s in Channel = c "
        "has negative entries in bin numbers = \n1 : -1 , 3 : -0.5\n"
    )
