"""HistFactory's refusals and corners: what it will not build, and the cases between.

A channel without samples, one whose histograms are missing, a first channel named with a
digit, channels whose datasets differ, a shape factor with an initial shape, an output
directory that cannot be made, a workspace without its ``ModelConfig``, extra data without a
name - each refused as ROOT refuses it, with ROOT's words where ROOT has some. Between them:
channels combined that have no observed data, a combination made from a measurement, the
statistical uncertainty of empty bins and of errors not a number, gammas with no uncertainty,
and a factory saying less when HistFactory's messages are switched off.
"""

from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Iterator
from typing import Any

import pytest

from histfactoryinputs import histogram, write_inputs
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.histfactory import assemble, combine, stat
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.factory import HistoToWorkspaceFactoryFast
from xrdroot.histfactory.make import MakeModelAndMeasurementFast
from xrdroot.histfactory.measurement import Measurement
from xrdroot.histfactory.model import Data, Sample
from xrdroot.histfactory.output import write_measurement
from xrdroot.histfactory.shapes import ShapeFactor
from xrdroot.histfactory.systematics import HistFactoryError, _copied
from xrdroot.roofit.messages import TOPICS, service
from xrdroot.roofit.variables import RooRealVar


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


@pytest.fixture
def path(tmp_path: Any) -> str:
    return write_inputs(tmp_path)


def measurement(prefix: str, *channels: Channel, poi: str = "mu") -> Measurement:
    meas = Measurement("m", "")
    meas.SetOutputFilePrefix(prefix)
    if poi:
        meas.SetPOI(poi)
    for one in channels:
        meas.AddChannel(one)
    return meas


def channel(name: str, path: str, data: bool = True, extra: str = "") -> Channel:
    """A channel of a signal scaled by ``mu`` - with data, and extra data named ``extra``."""
    made = Channel(name)
    if data:
        made.SetData("data", path)
    if extra:
        more = Data("data", path)
        more.SetName(extra)
        made.AddAdditionalData(more)
    signal = Sample("signal", "signal", path)
    signal.AddNormFactor("mu", 1, 0, 5)
    made.AddSample(signal)
    return made


def quietly(call: Any, *args: Any) -> Any:
    """``call(*args)``, what it prints kept out of the test's output."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return call(*args)


def test_channels_whose_combined_asimov_data_cannot_be_made_are_fatal(
        path: str, tmp_path: Any, monkeypatch: Any, capsys: Any) -> None:  # fmt: skip
    """Two channels of the same datasets combine; with no Asimov data made, it is fatal."""
    from xrdroot.roostats import asimov

    meas = measurement(str(tmp_path / "out"), channel("A", path, False), channel("B", path, False))
    meas.CollectHistograms()
    factory = HistoToWorkspaceFactoryFast(meas)
    made = [quietly(factory.MakeSingleChannelModel, meas, one) for one in meas.GetChannels()]
    monkeypatch.setattr(asimov, "GenerateAsimovData", lambda *args: None)
    capsys.readouterr()
    with pytest.raises(HistFactoryError, match="Failed to create combined asimov dataset"):
        factory.MakeCombinedModel(["A", "B"], made)
    assert capsys.readouterr().out.endswith(
        "[#2] FATAL:HistFactory -- Error: Failed to create combined asimov dataset\n")


def test_a_combination_is_made_from_a_measurement_its_channels_made_on_the_way(
        path: str, tmp_path: Any) -> None:  # fmt: skip
    meas = measurement(str(tmp_path / "out"), channel("A", path))
    meas.CollectHistograms()
    ws = quietly(HistoToWorkspaceFactoryFast(meas).MakeCombinedModel, meas)
    assert (ws.GetName(), sorted(d.GetName() for d in ws.allData())) == (
        "combined", ["asimovData", "obsData"])  # fmt: skip
    assert [p.GetName() for p in ws.obj("ModelConfig").GetParametersOfInterest()] == ["mu"]


def test_a_measurements_channel_with_its_histograms_unread_is_fatal(path: str,
                                                                    tmp_path: Any) -> None:  # fmt: skip
    meas = measurement(str(tmp_path / "out"), channel("A", path))
    for call in (MakeModelAndMeasurementFast,
                 HistoToWorkspaceFactoryFast(meas).MakeCombinedModel):  # fmt: skip
        with pytest.raises(HistFactoryError, match="Channel: A has uninitialized histogram"):
            quietly(call, meas)


def test_channels_whose_datasets_differ_are_refused_naming_the_difference(
        path: str, tmp_path: Any, capsys: Any) -> None:  # fmt: skip
    meas = measurement(str(tmp_path / "out"), channel("A", path, extra="one"),
                       channel("B", path, extra="two"))  # fmt: skip
    meas.CollectHistograms()
    capsys.readouterr()
    with pytest.raises(HistFactoryError, match="Inconsistent datasets"):
        MakeModelAndMeasurementFast(meas)
    assert capsys.readouterr().out.endswith(
        "[#2] FATAL:HistFactory -- ERROR: Inconsistent datasets across channel workspaces.\n"
        'Workspace for channel "B" does not match the datasets in channel "A".\n'
        "  Missing datasets:\n    - one\n  Extra datasets:\n    - two\n"
        "All channel workspaces must contain exactly the same datasets.\n\n"
    )


def test_a_first_channel_named_with_a_digit_is_refused(path: str, tmp_path: Any) -> None:
    meas = measurement(str(tmp_path / "out"), channel("1st", path))
    meas.CollectHistograms()
    with pytest.raises(ValueError, match="cannot start with a digit. Got 1st"):
        quietly(MakeModelAndMeasurementFast, meas)


def single(meas: Measurement, name: str = "") -> Any:
    """The measurement's first channel's workspace, made quietly."""
    one = meas.GetChannel(name) if name else meas.GetChannels()[0]
    return quietly(HistoToWorkspaceFactoryFast(meas).MakeSingleChannelWorkspace, meas, one)


def test_a_channel_without_samples_is_refused(tmp_path: Any) -> None:
    with pytest.raises(HistFactoryError, match="the channel empty has no sample"):
        single(measurement(str(tmp_path / "out"), Channel("empty")))


def test_a_channels_histograms_are_read_when_its_workspace_is_made(path: str,
                                                                   tmp_path: Any) -> None:  # fmt: skip
    """The first sample's histogram not read yet: the channel collects its histograms."""
    ws = single(measurement(str(tmp_path / "out"), channel("A", path)))
    assert ws.data("obsData").sumEntries() == 234.0


def test_a_channel_whose_data_is_not_read_is_fatal_when_its_workspace_is_made(
        path: str, tmp_path: Any, capsys: Any) -> None:  # fmt: skip
    made = channel("A", path, False)
    made.GetSamples()[0].SetHisto(histogram([1.0, 2.0], "nominal"))
    made.SetData("data", path)
    with pytest.raises(HistFactoryError):
        HistoToWorkspaceFactoryFast(measurement("out", made)).MakeSingleChannelWorkspace(
            measurement("out"), made)  # fmt: skip
    assert capsys.readouterr().out == (
        "[#2] ERROR:HistFactory -- Error: Data Histogram for channel A is nullptr.\n"
        "[#2] FATAL:HistFactory -- MakeSingleChannelWorkspace: Channel: A has uninitialized "
        "histogram pointers\n"
    )


def test_a_shape_factor_with_an_initial_shape_is_refused(path: str, tmp_path: Any) -> None:
    """``ParamHistFunc::setShape`` is not in the engine yet."""
    made = channel("A", path)
    factor = ShapeFactor("sf")
    factor.SetHistoName("signal")
    factor.SetInputFile(path)
    made.GetSamples()[0].AddShapeFactor(factor)
    meas = measurement(str(tmp_path / "out"), made)
    meas.CollectHistograms()
    with pytest.raises(UnsupportedFeatureError, match="the shape factor sf has an initial shape"):
        single(meas)


def test_a_constant_shape_factor_holds_its_gammas_constant(path: str, tmp_path: Any,
                                                           capsys: Any) -> None:  # fmt: skip
    made = channel("A", path)
    factor = ShapeFactor("sf")
    factor.SetConstant(True)
    made.GetSamples()[0].AddShapeFactor(factor)
    meas = measurement(str(tmp_path / "out"), made)
    meas.CollectHistograms()
    capsys.readouterr()
    ws = HistoToWorkspaceFactoryFast(meas).MakeSingleChannelWorkspace(meas, meas.GetChannel("A"))
    assert "[#2] INFO:HistFactory -- Setting Shape Factor: sf to be constant\n" in (
        capsys.readouterr().out)  # fmt: skip
    assert [ws.var(f"gamma_sf_bin_{i}").isConstant() for i in range(2)] == [True, True]


def test_extra_data_without_a_name_is_fatal(path: str, tmp_path: Any) -> None:
    made = channel("A", path)
    made.AddAdditionalData(Data("data", path))
    meas = measurement(str(tmp_path / "out"), made)
    meas.CollectHistograms()
    with pytest.raises(HistFactoryError, match="Additional Data histogram for channel: A has no"):
        single(meas)


def test_a_constraint_the_workspace_has_not_got_is_fatal(path: str, tmp_path: Any) -> None:
    """``finish`` looks every constraint up in the workspace, and says which it cannot find."""
    from xrdroot.roostats.modelconfig import ModelConfig

    meas = measurement(str(tmp_path / "out"), channel("A", path))
    ws = single(meas)
    with pytest.raises(HistFactoryError, match="Cannot find arg set: ghost in workspace: A"):
        quietly(assemble.finish, ws, ModelConfig("mc", ws), meas.GetChannel("A"),
                ["obs_x_A"], {"constraints": ["ghost"]})  # fmt: skip


def test_a_samples_shapes_are_named_after_its_first_function() -> None:
    """``..._Hist_alpha...`` or ``...nominal...`` becomes ``..._shapes``; else the name is kept."""
    assert [assemble._shape_name(n) for n in ("s_c_Hist_alphanominal", "s_c_nominal", "odd")] == [
        "s_c_shapes", "s_c_shapes", "odd"]  # fmt: skip


def test_a_workspace_without_a_model_config_cannot_be_configured(capsys: Any) -> None:
    from xrdroot.roofit.workspace import RooWorkspace

    with pytest.raises(HistFactoryError):
        combine.configure_for_measurement("model", RooWorkspace("bare"), Measurement("m"))
    assert capsys.readouterr().out == (
        "[#2] FATAL:HistFactory -- Error: Did not find 'ModelConfig' object in file: bare\n"
    )


def test_an_output_directory_that_cannot_be_made_is_fatal(path: str, tmp_path: Any) -> None:
    """The prefix's directory lies under a file: it cannot be made."""
    (tmp_path / "file").write_text("")
    meas = measurement(str(tmp_path / "file" / "sub" / "out"), channel("A", path))
    meas.CollectHistograms()
    with pytest.raises(HistFactoryError, match="Failed to make output directory"):
        quietly(MakeModelAndMeasurementFast, meas)


class Named:
    """A file, as far as its name goes."""

    def GetName(self) -> str:  # noqa: N802 - ROOT's name
        return "out.root"


def test_writing_a_channel_whose_histograms_are_not_read_is_refused(capsys: Any) -> None:
    meas = measurement("out", Channel("c"))
    meas.GetChannel("c").AddSample(Sample("s", "s", "f.root"))
    with pytest.raises(HistFactoryError, match="channel c lacks histograms"):
        write_measurement(meas, Named())
    assert capsys.readouterr().out.endswith(
        "[#2] ERROR:HistFactory -- Measurement.writeToFile(): Channel: c has uninitialized "
        "histogram pointers\n"
    )


def test_with_histfactorys_messages_off_the_factory_prints_no_workspace(path: str,
                                                                        tmp_path: Any,
                                                                        capsys: Any) -> None:  # fmt: skip
    """No ``Print`` of the channel's or the combined workspace, and no Asimov printout."""
    meas = measurement(str(tmp_path / "out"), channel("A", path), poi="")
    meas.CollectHistograms()
    capsys.readouterr()
    for stream in (1, 2):
        service().getStream(stream).removeTopic(TOPICS["HistFactory"])
    ws = MakeModelAndMeasurementFast(meas)
    out = capsys.readouterr().out
    assert "RooWorkspace(" not in out and "HistFactory" not in out
    assert ws.obj("ModelConfig").GetParametersOfInterest() is not None


def test_the_statistical_uncertainty_of_empty_bins_and_of_no_errors(capsys: Any) -> None:
    """A bin of no content has no relative error; a bin of no error a constant gamma."""
    nominal = histogram([4.0, 0.0])
    nominal.SetBinError(1, 2.0)
    error = stat.absolute_uncertainty("abs", nominal)
    relative = stat.relative_uncertainty("rel", [(nominal, error)])
    assert (stat.hist_values(error), stat.hist_values(relative)) == ([2.0, 0.0], [0.5, 0.0])
    assert "Sum of histograms for bin: 2 is <= 0.  Setting error to 0" in capsys.readouterr().out
    gammas = [RooRealVar(f"g{i}", "", 1.0, 0.0, 10.0) for i in range(2)]
    terms, globs = stat.gamma_constraints(gammas, [0.5, 0.0], 0.0, 1)
    assert ([t.GetName() for t in terms], [g.GetName() for g in globs]) == (
        ["g0_constraint"], ["nom_g0"])  # fmt: skip
    assert (gammas[0].isConstant(), gammas[1].isConstant()) == (False, True)
    assert ("[#2] INFO:HistFactory -- Not creating constraint term for g1 because sigma = 0 "
            "(sigma<=0) (bin number = 1)\n") in capsys.readouterr().out  # fmt: skip


def test_gammas_and_their_sigmas_must_be_as_many() -> None:
    with pytest.raises(HistFactoryError, match="1 relative sigmas were given for 2 gammas"):
        stat.gamma_constraints([RooRealVar("a", "", 1.0), RooRealVar("b", "", 1.0)], [0.1], 0, 0)


class Odd:
    """A histogram of one bin whose error is what it is given: negative, or not a number."""

    def __init__(self, error: float) -> None:
        self.error = error

    def GetNbinsX(self) -> int:  # noqa: N802 - ROOT's names
        return 1

    def GetNbinsY(self) -> int:  # noqa: N802
        return 1

    def GetNbinsZ(self) -> int:  # noqa: N802
        return 1

    def IsBinUnderflow(self, number: int) -> bool:  # noqa: N802
        return number == 0

    def IsBinOverflow(self, number: int) -> bool:  # noqa: N802
        return number > 1

    def GetBinError(self, number: int) -> float:  # noqa: N802
        return self.error

    def GetName(self) -> str:  # noqa: N802
        return "odd"

    def Clone(self, name: str = "") -> Any:  # noqa: N802
        return histogram([0.0], name)


def test_a_negative_bin_error_is_taken_as_none_and_one_not_a_number_is_refused(
        capsys: Any) -> None:  # fmt: skip
    assert stat.hist_values(stat.absolute_uncertainty("abs", Odd(-1.0))) == [0.0]
    assert capsys.readouterr().out == (
        "[#2] WARNING:HistFactory -- Warning: In histogram odd bin error for bin 1 is < 0.  "
        "Setting Error to 0\n"
    )
    with pytest.raises(HistFactoryError, match="bin 0 of odd has a NaN error"):
        stat.absolute_uncertainty("abs", Odd(math.nan))


def test_a_copy_of_something_in_no_directory_is_just_its_clone() -> None:
    class Plain:
        def Clone(self) -> str:  # noqa: N802 - ROOT's name
            return "clone"

    assert (_copied(Plain()), _copied(None)) == ("clone", None)


def test_a_channel_without_data_writes_only_its_samples(tmp_path: Any) -> None:
    """Its ``data`` directory is made and left empty; a sample prints without a stat error."""
    from xrdroot import open_root
    from xrdroot.pyroot.core import TFile

    made = Channel("c")
    sample = Sample("s")
    sample.SetHisto(histogram([1.0, 2.0], "nominal"))
    made.AddSample(sample)
    assert made.GetSamples()[0].text().count("\n") == 2
    filename = str(tmp_path / "out.root")
    handle = TFile.Open(filename, "RECREATE")
    quietly(write_measurement, measurement("out", made), handle)
    handle.Close()
    with open_root(filename) as f:
        assert (list(f["c_hists/data"].keys()), list(f["c_hists/s"].keys())) == ([], ["nominal"])
