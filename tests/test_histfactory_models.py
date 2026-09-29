"""HistFactory's models, built by ``MakeModelAndMeasurementFast`` and held to ROOT 6.40's.

Two measurements are built from histograms this suite writes: ``hf001``'s example - overall
systematics, a normalisation factor, statistical errors from a histogram and from the
samples' own errors, Poisson constrained - and a channel of every other kind of piece: a
shape systematic uniformly constrained, Gamma-, uniform- and un-constrained overall
systematics and one of no uncertainty, Poisson and Gaussian shape systematics, a free shape
factor, a sample not normalised by the luminosity, Gaussian statistical errors below the
threshold in a bin, a parameter value set and one held constant. ROOT built the same two
from the same histograms: the parameters, observables, global observables and datasets, the
likelihood before and after the parameters are moved, which parameters are constant and the
gammas' ranges and errors are ROOT's, and so is every message HistFactory gives.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
from collections.abc import Iterator
from typing import Any

import pytest

from histfactoryinputs import write_inputs
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.make import MakeModelAndMeasurementFast
from xrdroot.histfactory.measurement import Measurement
from xrdroot.histfactory.model import Data, Sample
from xrdroot.histfactory.systematics import Constraint
from xrdroot.roofit.messages import service


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def hf001(path: str, prefix: str) -> Measurement:
    meas = Measurement("meas", "meas")
    meas.SetOutputFilePrefix(prefix)
    meas.SetPOI("SigXsecOverSM")
    meas.AddConstantParam("Lumi")
    meas.AddConstantParam("alpha_syst1")
    meas.SetLumi(1.0)
    meas.SetLumiRelErr(0.10)
    chan = Channel("channel1")
    chan.SetData("data", path)
    chan.SetStatErrorConfig(0.05, "Poisson")
    signal = Sample("signal", "signal", path)
    signal.AddOverallSys("syst1", 0.95, 1.05)
    signal.AddNormFactor("SigXsecOverSM", 1, 0, 3)
    chan.AddSample(signal)
    b1 = Sample("background1", "background1", path)
    b1.ActivateStatError("background1_statUncert", path)
    b1.AddOverallSys("syst2", 0.95, 1.05)
    chan.AddSample(b1)
    b2 = Sample("background2", "background2", path)
    b2.ActivateStatError()
    b2.AddOverallSys("syst3", 0.95, 1.05)
    chan.AddSample(b2)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def sink(path: str, prefix: str) -> Measurement:
    meas = Measurement("sink", "sink")
    meas.SetOutputFilePrefix(prefix)
    meas.SetPOI("mu")
    meas.SetLumi(1.0)
    meas.SetLumiRelErr(0.0)
    meas.AddGammaSyst("gsys", 0.1)
    meas.AddGammaSyst("zsys", 0.0)
    meas.AddUniformSyst("shape")
    meas.AddUniformSyst("usys")
    meas.AddNoSyst("nsys")
    meas.SetParamValue("mu", 1.5)
    meas.AddConstantParam("alpha_usys")
    chan = Channel("sink")
    chan.SetData("data", path)
    chan.SetStatErrorConfig(0.2, "Gaussian")
    s = Sample("signal", "signal", path)
    s.AddNormFactor("mu", 1, 0, 5)
    s.AddHistoSys("shape", "signal_low", path, "", "signal_high", path, "")
    for name, low, high in (("gsys", 0.9, 1.1), ("zsys", 0.9, 1.1), ("usys", 0.8, 1.2),
                            ("nsys", 0.8, 1.2)):  # fmt: skip
        s.AddOverallSys(name, low, high)
    s.ActivateStatError()
    chan.AddSample(s)
    b1 = Sample("background1", "background1", path)
    b1.AddShapeSys("ss1", Constraint.Poisson, "shape_err", path, "")
    b1.ActivateStatError()
    chan.AddSample(b1)
    b2 = Sample("background2", "background2", path)
    b2.AddShapeSys("ss2", Constraint.Gaussian, "shape_err", path, "")
    b2.AddShapeFactor("free")
    b2.SetNormalizeByTheory(False)
    b2.AddNormFactor("mu", 1, 0, 5)
    chan.AddSample(b2)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


class Built:
    """A measurement's combined workspace, what was said making it, and where it was written."""

    def __init__(self, directory: pathlib.Path, make: Any, prefix: Any = None) -> None:
        path = write_inputs(directory)
        self.prefix = prefix or str(directory / "results" / "out")
        self.said, self.errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(self.said), contextlib.redirect_stderr(self.errors):
            service().reset()
            self.meas = make(path, self.prefix)
            self.ws = MakeModelAndMeasurementFast(self.meas)
        self.path = path


@pytest.fixture(scope="module")
def built_hf001(tmp_path_factory: Any) -> Built:
    return Built(tmp_path_factory.mktemp("hf001"), hf001)


@pytest.fixture(scope="module")
def built_sink(tmp_path_factory: Any) -> Built:
    return Built(tmp_path_factory.mktemp("sink"), sink)


def names(collection: Any) -> str:
    return ",".join(sorted(one.GetName() for one in collection))


#: ``hf001``'s combined model, as ROOT 6.40 built it from the same histograms.
HF001 = (
    "pdf simPdf\n"
    "vars Lumi,SigXsecOverSM,alpha_syst1,alpha_syst2,alpha_syst3,"
    "gamma_stat_channel1_bin_0,gamma_stat_channel1_bin_1,nom_alpha_syst1,nom_alpha_syst2,"
    "nom_alpha_syst3,nom_gamma_stat_channel1_bin_0,nom_gamma_stat_channel1_bin_1,"
    "nominalLumi,obs_x_channel1\n"
    "poi SigXsecOverSM\n"
    "nuis alpha_syst2,alpha_syst3,gamma_stat_channel1_bin_0,gamma_stat_channel1_bin_1\n"
    "glob nom_alpha_syst1,nom_alpha_syst2,nom_alpha_syst3,nom_gamma_stat_channel1_bin_0,"
    "nom_gamma_stat_channel1_bin_1,nominalLumi\n"
    "obs channelCat,obs_x_channel1\n"
    "data asimovData,obsData\n"
    "sum asimovData 230.0\n"
    "sum obsData 234.0\n"
    "nll 4.549054846130797\n"
    "nll2 5.712875664662276\n"
    "const Lumi,alpha_syst1,nom_alpha_syst1,nom_alpha_syst2,nom_alpha_syst3,"
    "nom_gamma_stat_channel1_bin_0,nom_gamma_stat_channel1_bin_1,nominalLumi\n"
    "gamma gamma_stat_channel1_bin_0 1.0 0.0 1.2500000037252903 0.05000000074505806\n"
    "gamma gamma_stat_channel1_bin_1 1.1 0.0 1.5000000074505806 0.10000000149011612\n"
    "gamma nom_gamma_stat_channel1_bin_0 399.99998807907133 0.0 inf 0.0\n"
    "gamma nom_gamma_stat_channel1_bin_1 99.99999701976783 0.0 inf 0.0\n"
)
#: The channel of every other piece, as ROOT 6.40 built it.
SINK = (
    "pdf simPdf\n"
    "vars Lumi,alpha_shape,alpha_usys,beta_gsys,gamma_free_bin_0,gamma_free_bin_1,"
    "gamma_ss1_bin_0,gamma_ss1_bin_1,gamma_ss2_bin_0,gamma_ss2_bin_1,"
    "gamma_stat_sink_bin_0,gamma_stat_sink_bin_1,mu,nom_alpha_shape,nom_alpha_usys,"
    "nom_beta_gsys,nom_gamma_ss1_bin_0,nom_gamma_ss1_bin_1,nom_gamma_ss2_bin_0,"
    "nom_gamma_ss2_bin_1,nom_gamma_stat_sink_bin_0,nom_gamma_stat_sink_bin_1,obs_x_sink,"
    "theta_gsys\n"
    "poi mu\n"
    "nuis alpha_shape,beta_gsys,gamma_free_bin_0,gamma_free_bin_1,gamma_ss1_bin_0,"
    "gamma_ss1_bin_1,gamma_ss2_bin_0,gamma_ss2_bin_1,gamma_stat_sink_bin_1\n"
    "glob nom_alpha_shape,nom_alpha_usys,nom_beta_gsys,nom_gamma_ss1_bin_0,"
    "nom_gamma_ss1_bin_1,nom_gamma_ss2_bin_0,nom_gamma_ss2_bin_1,"
    "nom_gamma_stat_sink_bin_0,nom_gamma_stat_sink_bin_1\n"
    "obs channelCat,obs_x_sink\n"
    "data asimovData,obsData\n"
    "sum asimovData 295.0\n"
    "sum obsData 234.0\n"
    "nll 80.55243156930756\n"
    "nll2 92.07181991983914\n"
    "const Lumi,alpha_usys,gamma_stat_sink_bin_0,nom_alpha_shape,nom_alpha_usys,"
    "nom_beta_gsys,nom_gamma_ss1_bin_0,nom_gamma_ss1_bin_1,nom_gamma_ss2_bin_0,"
    "nom_gamma_ss2_bin_1,nom_gamma_stat_sink_bin_0,nom_gamma_stat_sink_bin_1,theta_gsys\n"
    "gamma gamma_free_bin_0 1.0 0.0 1000.0 0.0\n"
    "gamma gamma_free_bin_1 1.2 0.0 1000.0 0.0\n"
    "gamma gamma_ss1_bin_0 0.9 0.0 1.625 0.125\n"
    "gamma gamma_ss1_bin_1 1.0 0.0 2.25 0.25\n"
    "gamma gamma_ss2_bin_0 1.0 0.0 1.625 0.125\n"
    "gamma gamma_ss2_bin_1 1.0 0.0 2.25 0.25\n"
    "gamma gamma_stat_sink_bin_0 1.05 0.0 1.4564354568719864 0.09128709137439728\n"
    "gamma gamma_stat_sink_bin_1 1.0 0.0 2.5811388194561005 0.3162277638912201\n"
    "gamma nom_gamma_ss1_bin_0 64.0 0.0 inf 0.0\n"
    "gamma nom_gamma_ss1_bin_1 16.0 0.0 inf 0.0\n"
    "gamma nom_gamma_ss2_bin_0 1.0 0.0 10.0 0.0\n"
    "gamma nom_gamma_ss2_bin_1 1.0 0.0 10.0 0.0\n"
    "gamma nom_gamma_stat_sink_bin_0 1.0 0.0 10.0 0.0\n"
    "gamma nom_gamma_stat_sink_bin_1 1.0 0.0 10.0 0.0\n"
)


def summary(ws: Any, moves: Any, detail: bool = False) -> str:
    """The model's sets, datasets, likelihood - then moved - constants and gammas, a line each;
    in ``detail``, the observables too, with each one's bins and title."""
    config = ws.obj("ModelConfig")
    pdf = config.GetPdf()
    lines = [f"pdf {pdf.GetName()}", f"vars {names(ws.allVars())}",
             f"poi {names(config.GetParametersOfInterest() or [])}",
             f"nuis {names(config.GetNuisanceParameters() or [])}",
             f"glob {names(config.GetGlobalObservables())}",
             f"obs {names(config.GetObservables())}", f"data {names(ws.allData())}"]  # fmt: skip
    lines += [f"sum {d.GetName()} {d.sumEntries()!r}"
              for d in sorted(ws.allData(), key=lambda d: d.GetName())]  # fmt: skip
    nll = pdf.createNLL(ws.data("obsData"))
    lines.append(f"nll {nll.getVal()!r}")
    for name, value in moves:
        ws.var(name).setVal(value)
    lines.append(f"nll2 {nll.getVal()!r}")
    lines.append(f"const {names(v for v in ws.allVars() if v.isConstant())}")
    kinds = ("gamma", "nom_gamma", "obs") if detail else ("gamma", "nom_gamma")
    for v in sorted(ws.allVars(), key=lambda v: v.GetName()):
        if v.GetName().startswith(kinds):
            line = f"{v.GetName()} {v.getVal()!r} {v.getMin()!r} {v.getMax()!r} {v.getError()!r}"
            lines.append(f"var {line} {v.getBins()} {v.GetTitle()}" if detail else f"gamma {line}")
    ws.loadSnapshot("NominalParamValues")
    return "\n".join(lines) + "\n"


def test_hf001s_model_is_roots(built_hf001: Built) -> None:
    """Sets, datasets, likelihood before and after a pull, constants and gammas: ROOT's."""
    moves = (("SigXsecOverSM", 1.3), ("alpha_syst2", 0.4), ("gamma_stat_channel1_bin_1", 1.1))
    assert summary(built_hf001.ws, moves) == HF001


def test_the_channel_of_every_other_piece_is_roots(built_sink: Built) -> None:
    """Gamma, uniform and log-free systematics, shape systematics and factors: ROOT's model."""
    moves = (("mu", 1.3), ("alpha_shape", 0.7), ("beta_gsys", 1.1), ("gamma_ss1_bin_0", 0.9),
             ("gamma_free_bin_1", 1.2), ("gamma_stat_sink_bin_0", 1.05))  # fmt: skip
    assert summary(built_sink.ws, moves) == SINK


def roots_words(measurement: str) -> str:
    """What ROOT 6.40 printed making ``measurement``, from ``tests/data/histfactory-6.40.txt``."""
    text = (pathlib.Path(__file__).parent / "data" / "histfactory-6.40.txt").read_text()
    parts = dict(part.split("\n", 1) for part in text.split("#### ")[1:])
    return parts[measurement]


@pytest.mark.parametrize("which", ["meas", "sink"])
def test_making_the_model_says_everything_root_says(which: str, built_hf001: Built,
                                                    built_sink: Built) -> None:  # fmt: skip
    """Every message, workspace print and ``ModelConfig`` summary, line for line ROOT's."""
    built = built_hf001 if which == "meas" else built_sink
    said = built.said.getvalue()
    said = said[said.index("[#2] INFO:HistFactory -- Making Model and Measurements"):]
    expected = roots_words(which).replace("{input}", built.path).replace("{prefix}", built.prefix)
    assert said == expected


def test_what_is_not_written_to_the_files_is_said_on_the_standard_error(built_hf001: Built) -> None:
    """The workspaces and the measurement object: xrdroot does not write those classes yet."""
    prefix = built_hf001.prefix
    tail = "xrdroot does not write that class yet; the file holds the measurement's histograms\n"
    assert built_hf001.errors.getvalue() == "".join(
        f"xrdroot: the {what} is not written to {prefix}_{kind}_meas_model.root: {tail}"
        for kind, whats in (("channel1", ("RooWorkspace channel1", "Measurement")),
                            ("combined", ("RooWorkspace combined", "Measurement")))
        for what in whats)  # fmt: skip


def test_the_files_hold_the_measurements_histograms_where_its_configuration_says(
        built_hf001: Built) -> None:  # fmt: skip
    """``channel1_hists/<sample>/...`` in each file, and the configuration renamed to point there."""
    from xrdroot import open_root

    filename = f"{built_hf001.prefix}_combined_meas_model.root"
    with open_root(filename) as f:
        folder = f["channel1_hists"]
        assert sorted(folder.keys()) == ["background1", "background2", "data", "signal"]
        assert list(f["channel1_hists/background1"].keys()) == ["background1", "statisticalErrors"]
        assert f["channel1_hists/data/data"].values().tolist() == [122.0, 112.0]
    with open_root(f"{built_hf001.prefix}_channel1_meas_model.root") as f:
        assert "channel1_hists" in f.keys()
    sample = built_hf001.meas.GetChannel("channel1").GetSamples()[1]
    assert (sample.GetInputFile(), sample.GetHistoPath(), sample.GetStatError().GetHistoName()) == (
        filename, "/channel1_hists/background1/", "statisticalErrors")  # fmt: skip
    data = built_hf001.meas.GetChannel("channel1").GetData()
    assert (data.GetInputFile(), data.GetHistoPath()) == (filename, "/channel1_hists/data/")


#: Uneven bins, a preprocess function, shared systematics, extra data, as ROOT 6.40 built it.
VARIETY = (
    "pdf simPdf\n"
    "vars Lumi,alpha_hs,alpha_lsys,gamma_sfv_bin_0,gamma_sfv_bin_1,gamma_ssv_bin_0,"
    "gamma_ssv_bin_1,kappa_lsys,mu,nom_alpha_hs,nom_alpha_lsys,nom_gamma_ssv_bin_0,"
    "nom_gamma_ssv_bin_1,nominalLumi,obs_x_var,tau_lsys\n"
    "poi mu\n"
    "nuis Lumi,alpha_hs,alpha_lsys,gamma_sfv_bin_0,gamma_sfv_bin_1,gamma_ssv_bin_0,"
    "gamma_ssv_bin_1\n"
    "glob nom_alpha_hs,nom_alpha_lsys,nom_gamma_ssv_bin_0,nom_gamma_ssv_bin_1,"
    "nominalLumi\n"
    "obs channelCat,obs_x_var\n"
    "data asimovData,extra,obsData\n"
    "sum asimovData 75.0\n"
    "sum extra 45.0\n"
    "sum obsData 75.0\n"
    "nll 4.279146807924044\n"
    "nll2 26.778926477857727\n"
    "const kappa_lsys,nom_alpha_hs,nom_alpha_lsys,nom_gamma_ssv_bin_0,"
    "nom_gamma_ssv_bin_1,nominalLumi,tau_lsys\n"
    "var gamma_sfv_bin_0 1.0 0.0 1000.0 0.0 100 gamma_sfv_bin_0\n"
    "var gamma_sfv_bin_1 1.2 0.0 1000.0 0.0 100 gamma_sfv_bin_1\n"
    "var gamma_ssv_bin_0 0.9 0.0 1.625 0.125 100 gamma_ssv_bin_0\n"
    "var gamma_ssv_bin_1 1.0 0.0 2.25 0.25 100 gamma_ssv_bin_1\n"
    "var nom_gamma_ssv_bin_0 64.0 0.0 inf 0.0 100 nom_gamma_ssv_bin_0\n"
    "var nom_gamma_ssv_bin_1 16.0 0.0 inf 0.0 100 nom_gamma_ssv_bin_1\n"
    "var obs_x_var 0.5 0.0 3.0 0.0 2 m [GeV]\n"
)
#: The two-dimensional channel, as ROOT 6.40 built it.
SQUARE = (
    "pdf simPdf\n"
    "vars Lumi,gamma_stat_sq_bin_0_0,gamma_stat_sq_bin_0_1,gamma_stat_sq_bin_1_0,"
    "gamma_stat_sq_bin_1_1,mu,nom_gamma_stat_sq_bin_0_0,nom_gamma_stat_sq_bin_0_1,"
    "nom_gamma_stat_sq_bin_1_0,nom_gamma_stat_sq_bin_1_1,nominalLumi,obs_x_sq,obs_y_sq\n"
    "poi mu\n"
    "nuis Lumi,gamma_stat_sq_bin_0_0,gamma_stat_sq_bin_0_1,gamma_stat_sq_bin_1_0,"
    "gamma_stat_sq_bin_1_1\n"
    "glob nom_gamma_stat_sq_bin_0_0,nom_gamma_stat_sq_bin_0_1,nom_gamma_stat_sq_bin_1_0,"
    "nom_gamma_stat_sq_bin_1_1,nominalLumi\n"
    "obs channelCat,obs_x_sq,obs_y_sq\n"
    "data asimovData,obsData\n"
    "sum asimovData 52.0\n"
    "sum obsData 52.0\n"
    "nll 6.639259848918925\n"
    "nll2 6.936507924069916\n"
    "const nom_gamma_stat_sq_bin_0_0,nom_gamma_stat_sq_bin_0_1,nom_gamma_stat_sq_bin_1_0,"
    "nom_gamma_stat_sq_bin_1_1,nominalLumi\n"
    ""
    "var gamma_stat_sq_bin_0_0 1.0 0.0 2.767766922712326 0.3535533845424652 100 gamma_stat_sq_bin_0_0\n"
    ""
    "var gamma_stat_sq_bin_0_1 1.0 0.0 2.5811388194561005 0.3162277638912201 100 gamma_stat_sq_bin_0_1\n"
    ""
    "var gamma_stat_sq_bin_1_0 1.1 0.0 2.666666716337204 0.3333333432674408 100 gamma_stat_sq_bin_1_0\n"
    ""
    "var gamma_stat_sq_bin_1_1 1.0 0.0 2.507556736469269 0.30151134729385376 100 gamma_stat_sq_bin_1_1\n"
    ""
    "var nom_gamma_stat_sq_bin_0_0 8.000000273828343 0.0 inf 0.0 100 nom_gamma_stat_sq_bin_0_0\n"
    ""
    "var nom_gamma_stat_sq_bin_0_1 10.000000134435878 0.0 inf 0.0 100 nom_gamma_stat_sq_bin_0_1\n"
    ""
    "var nom_gamma_stat_sq_bin_1_0 8.99999946355822 0.0 inf 0.0 100 nom_gamma_stat_sq_bin_1_0\n"
    ""
    "var nom_gamma_stat_sq_bin_1_1 10.999999801818461 0.0 inf 0.0 100 nom_gamma_stat_sq_bin_1_1\n"
    "var obs_x_sq 0.5 0.0 2.0 0.0 2 obs_x_sq\n"
    "var obs_y_sq 0.5 0.0 2.0 0.0 2 obs_y_sq\n"
)
#: The three-dimensional channel, as ROOT 6.40 built it.
CUBE = (
    "pdf simPdf\n"
    "vars Lumi,gamma_stat_cube_bin_0_0_0,gamma_stat_cube_bin_0_0_1,"
    "gamma_stat_cube_bin_1_0_0,gamma_stat_cube_bin_1_0_1,mu,"
    "nom_gamma_stat_cube_bin_0_0_0,nom_gamma_stat_cube_bin_0_0_1,"
    "nom_gamma_stat_cube_bin_1_0_0,nom_gamma_stat_cube_bin_1_0_1,nominalLumi,obs_x_cube,"
    "obs_y_cube,obs_z_cube\n"
    "poi mu\n"
    "nuis Lumi,gamma_stat_cube_bin_0_0_0,gamma_stat_cube_bin_0_0_1,"
    "gamma_stat_cube_bin_1_0_0,gamma_stat_cube_bin_1_0_1\n"
    "glob nom_gamma_stat_cube_bin_0_0_0,nom_gamma_stat_cube_bin_0_0_1,"
    "nom_gamma_stat_cube_bin_1_0_0,nom_gamma_stat_cube_bin_1_0_1,nominalLumi\n"
    "obs channelCat,obs_x_cube,obs_y_cube,obs_z_cube\n"
    "data asimovData,obsData\n"
    "sum asimovData 10.0\n"
    "sum obsData 30.0\n"
    "nll 21.32388311079403\n"
    "nll2 15.847992299900097\n"
    "const nom_gamma_stat_cube_bin_0_0_0,nom_gamma_stat_cube_bin_0_0_1,"
    "nom_gamma_stat_cube_bin_1_0_0,nom_gamma_stat_cube_bin_1_0_1,nominalLumi\n"
    "var gamma_stat_cube_bin_0_0_0 1.0 0.0 6.0 1.0 100 gamma_stat_cube_bin_0_0_0\n"
    ""
    "var gamma_stat_cube_bin_0_0_1 1.0 0.0 3.8867512941360474 0.5773502588272095 100 gamma_stat_cube_bin_0_0_1\n"
    ""
    "var gamma_stat_cube_bin_1_0_0 1.0 0.0 4.535533845424652 0.7071067690849304 100 gamma_stat_cube_bin_1_0_0\n"
    "var gamma_stat_cube_bin_1_0_1 1.1 0.0 3.5 0.5 100 gamma_stat_cube_bin_1_0_1\n"
    "var nom_gamma_stat_cube_bin_0_0_0 1.0 0.0 inf 0.0 100 nom_gamma_stat_cube_bin_0_0_0\n"
    ""
    "var nom_gamma_stat_cube_bin_0_0_1 3.000000107689392 0.0 inf 0.0 100 nom_gamma_stat_cube_bin_0_0_1\n"
    ""
    "var nom_gamma_stat_cube_bin_1_0_0 2.0000000684570858 0.0 inf 0.0 100 nom_gamma_stat_cube_bin_1_0_0\n"
    "var nom_gamma_stat_cube_bin_1_0_1 4.0 0.0 inf 0.0 100 nom_gamma_stat_cube_bin_1_0_1\n"
    "var obs_x_cube 0.5 0.0 2.0 0.0 2 obs_x_cube\n"
    "var obs_y_cube 0.5 0.0 1.0 0.0 1 obs_y_cube\n"
    "var obs_z_cube 0.5 0.0 2.0 0.0 2 obs_z_cube\n"
)
#: A channel of one sample and no parameter of interest, as ROOT 6.40 built it.
PLAIN = (
    "pdf simPdf\n"
    "vars Lumi,mu,nominalLumi,obs_x_plain\n"
    "poi \n"
    "nuis \n"
    "glob nominalLumi\n"
    "obs channelCat,obs_x_plain\n"
    "data asimovData,obsData\n"
    "sum asimovData 30.0\n"
    "sum obsData 234.0\n"
    "nll 292.4104908224435\n"
    "nll2 240.01725293705056\n"
    "const nominalLumi\n"
    "var obs_x_plain 1.25 1.0 2.0 0.0 2 obs_x_plain\n"
)
#: A single channel's model with an Asimov dataset of the measurement's, as ROOT 6.40 built it.
ASIM = (
    "pdf model_asim\n"
    "vars Lumi,mu,nominalLumi,obs_x_asim\n"
    "poi mu\n"
    "nuis Lumi\n"
    "glob nominalLumi\n"
    "obs obs_x_asim\n"
    "data asimovData,asimov_mu0,obsData\n"
    "sum asimovData 130.0\n"
    "sum asimov_mu0 100.0\n"
    "sum obsData 234.0\n"
    "nll 173.8158355766207\n"
    "nll2 147.47863792759102\n"
    "const nominalLumi\n"
    "var obs_x_asim 1.75 1.0 2.0 0.0 2 obs_x_asim\n"
)


def variety(path: str, prefix: str) -> Measurement:
    meas = Measurement("variety", "")
    meas.SetOutputFilePrefix(prefix)
    meas.SetPOI("mu")
    meas.AddPOI("nope")
    meas.AddPreprocessFunction("sq", "mu*mu", "mu[1,0,5]")
    meas.AddLogNormSyst("lsys", 0.1)
    meas.AddConstantParam("notthere")
    meas.SetParamValue("absent", 2.0)
    chan = Channel("var")
    chan.SetData("vdata", path)
    extra = Data("vbkg", path)
    extra.SetName("extra")
    chan.AddAdditionalData(extra)
    for name, hist, norms in (("sigv", "vsig", ("mu", "mu", "sq")), ("bkgv", "vbkg", ())):
        sample = Sample(name, hist, path)
        for norm in norms:
            sample.AddNormFactor(norm, 1, 0, 5)
        if norms:
            sample.AddOverallSys("lsys", 0.9, 1.1)
        sample.AddHistoSys("hs", f"{hist}_lo", path, "", f"{hist}_hi", path, "")
        sample.AddShapeFactor("sfv")
        sample.AddShapeSys("ssv", Constraint.Poisson, "verr", path, "")
        chan.AddSample(sample)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def square(path: str, prefix: str) -> Measurement:
    meas = Measurement("square", "")
    meas.SetOutputFilePrefix(prefix)
    meas.SetPOI("mu")
    chan = Channel("sq")
    chan.SetData("data2d", path)
    signal = Sample("sig", "sig2d", path)
    signal.AddNormFactor("mu", 1, 0, 5)
    chan.AddSample(signal)
    background = Sample("bkg", "bkg2d", path)
    background.ActivateStatError()
    chan.AddSample(background)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def cube(path: str, prefix: str) -> Measurement:
    meas = Measurement("cube", "")
    meas.SetOutputFilePrefix(prefix)
    meas.SetPOI("mu")
    chan = Channel("cube")
    chan.SetData("data3d", path)
    signal = Sample("sig", "sig3d", path)
    signal.AddNormFactor("mu", 1, 0, 5)
    signal.ActivateStatError()
    chan.AddSample(signal)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def plain(path: str, prefix: str) -> Measurement:
    meas = Measurement("plain", "")
    meas.SetOutputFilePrefix(prefix)
    chan = Channel("plain")
    chan.SetData("data", path)
    signal = Sample("signal", "signal", path)
    signal.AddNormFactor("mu", 1, 0, 5)
    chan.AddSample(signal)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def detailed(ws: Any, moves: Any) -> str:
    """The summary of :func:`summary`, with the observables' and gammas' bins and titles."""
    return summary(ws, moves, detail=True)


@pytest.mark.parametrize(("make", "expected", "moves"), [
    (variety, VARIETY, (("mu", 1.3), ("alpha_hs", 0.5), ("alpha_lsys", -0.7),
                        ("gamma_sfv_bin_1", 1.2), ("gamma_ssv_bin_0", 0.9))),
    (square, SQUARE, (("mu", 1.3), ("gamma_stat_sq_bin_1_0", 1.1))),
    (cube, CUBE, (("mu", 1.3), ("gamma_stat_cube_bin_1_0_1", 1.1))),
    (plain, PLAIN, (("mu", 1.3),)),
])  # fmt: skip
def test_more_kinds_of_channel_build_roots_models_saying_what_root_says(
        make: Any, expected: str, moves: Any, tmp_path: Any, monkeypatch: Any) -> None:  # fmt: skip
    """Uneven bins, two and three dimensions, a model of no parameter of interest: ROOT's."""
    monkeypatch.chdir(tmp_path)  # the plain model's prefix names no directory: it is written here
    built = Built(tmp_path, make, "results_h" if make is plain else None)
    assert detailed(built.ws, moves) == expected
    said = built.said.getvalue()
    said = said[said.index("[#2] INFO:HistFactory -- Making Model and Measurements"):]
    words = roots_words(built.meas.GetName())
    assert said == words.replace("{input}", built.path).replace("{prefix}", built.prefix)


def test_a_single_channels_model_makes_the_measurements_asimov_datasets(tmp_path: Any) -> None:
    """``asimov_mu0``: the model's expectation with ``mu`` set to zero - then set back."""
    from xrdroot.histfactory.factory import HistoToWorkspaceFactoryFast
    from xrdroot.histfactory.shapes import Asimov

    path = write_inputs(tmp_path)
    meas = Measurement("asim", "")
    meas.SetPOI("mu")
    asimov = Asimov("asimov_mu0")
    asimov.SetParamValue("mu", 0.0)
    asimov.SetFixedParam("mu")
    meas.AddAsimovDataset(asimov)
    chan = Channel("asim")
    chan.SetData("data", path)
    signal = Sample("signal", "signal", path)
    signal.AddNormFactor("mu", 1, 0, 5)
    chan.AddSample(signal)
    chan.AddSample(Sample("background1", "background1", path))
    meas.AddChannel(chan)
    with contextlib.redirect_stdout(io.StringIO()) as said:
        meas.CollectHistograms()
        ws = HistoToWorkspaceFactoryFast(meas).MakeSingleChannelModel(meas, meas.GetChannels()[0])
    assert detailed(ws, (("mu", 1.3),)) == ASIM
    assert ("[#2] PROGRESS:HistFactory -- Generating additional Asimov Dataset: asimov_mu0\n"
            "Configuring Asimov Dataset: Setting mu = 0\n") in said.getvalue()
