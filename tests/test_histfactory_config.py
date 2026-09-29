"""HistFactory's configuration: the measurement, its channels, samples and systematics.

What each piece holds, how it is copied - a channel added to a measurement and a sample added
to a channel are copies, as C++ copies them - and how it prints: every ``Print`` and the
measurement's ``PrintTree`` below are what ROOT 6.40 printed for the same calls, tab by tab,
with its messages; so are the refusals of a constraint type ROOT does not know and of a
channel the measurement has not got.
"""

from __future__ import annotations

import copy
import io
from collections.abc import Iterator
from typing import Any

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.measurement import Measurement
from xrdroot.histfactory.model import Data, Sample
from xrdroot.histfactory.shapes import (
    Asimov,
    PreprocessFunction,
    ShapeFactor,
    ShapeSys,
    StatError,
    StatErrorConfig,
)
from xrdroot.histfactory.systematics import (
    Constraint,
    HistFactoryError,
    HistoFactor,
    HistoSys,
    NormFactor,
    OverallSys,
)
from xrdroot.pyroot.core import TH1F
from xrdroot.roofit.cmdargs import RooCmdArg
from xrdroot.roofit.messages import service


@pytest.fixture(autouse=True)
def _messages() -> Iterator[None]:
    """Every test starts from the message streams RooFit starts with."""
    service().reset()
    yield
    service().reset()


def printed(obj: Any) -> str:
    stream = io.StringIO()
    obj.Print(stream)
    return stream.getvalue()


def test_the_normalisation_systematics_print_as_root_prints_them(capsys: Any) -> None:
    """An ``OverallSys`` and a ``NormFactor``, to ``cout`` and to a stream given."""
    sys = OverallSys()
    sys.SetName("os")
    sys.SetLow(0.9)
    sys.SetHigh(1.1)
    norm = NormFactor()
    norm.SetName("nf")
    norm.SetVal(1.5)
    norm.SetLow(0)
    norm.SetHigh(3)
    sys.Print()
    assert capsys.readouterr().out == "\t \t Name: os\t Low: 0.9\t High: 1.1\n"
    assert printed(norm) == "\t \t Name: nf\t Val: 1.5\t Low: 0\t High: 3\n"
    assert (sys.GetName(), sys.GetLow(), sys.GetHigh(), norm.GetVal()) == ("os", 0.9, 1.1, 1.5)


def histo_sys(kind: Any = HistoSys) -> Any:
    made = kind()
    made.SetName("hs")
    made.SetHistoNameLow("lo")
    made.SetInputFileLow("f.root")
    made.SetHistoPathLow("p/")
    made.SetHistoNameHigh("hi")
    made.SetInputFileHigh("g.root")
    made.SetHistoPathHigh("q/")
    return made


def test_a_histogram_systematic_prints_where_its_histograms_are() -> None:
    made = histo_sys()
    assert printed(made) == (
        "\t \t Name: hs\t HistoFileLow: f.root\t HistoNameLow: lo\t HistoPathLow: p/\t "
        "HistoFileHigh: g.root\t HistoNameHigh: hi\t HistoPathHigh: q/\n"
    )
    assert [made.GetInputFileLow(), made.GetHistoNameLow(), made.GetHistoPathLow(),
            made.GetInputFileHigh(), made.GetHistoNameHigh(), made.GetHistoPathHigh()] == [
        "f.root", "lo", "p/", "g.root", "hi", "q/"]  # fmt: skip


def test_a_histogram_systematic_keeps_detached_copies_of_its_histograms() -> None:
    """Its histograms are taken out of any directory, and a copy of it has clones of them."""
    made = histo_sys(HistoFactor)
    low, high = TH1F("lo", "", 2, 0, 2), TH1F("hi", "", 2, 0, 2)
    made.SetHistoLow(low)
    made.SetHistoHigh(high)
    made.SetHistoLow(None)  # nothing to detach
    made.SetHistoLow(low)
    copied = copy.deepcopy(made)
    assert (made.GetHistoLow() is low, made.GetHistoHigh() is high) == (True, True)
    assert copied.GetHistoLow() is not low and copied.GetHistoLow().GetName() == "lo"
    assert copied.GetHistoNameHigh() == "hi"
    copied.SetHistoNameHigh("other")
    assert made.GetHistoNameHigh() == "hi"


def test_a_shape_systematic_prints_its_error_histograms_place_and_takes_its_constraint() -> None:
    shape = ShapeSys()
    shape.SetName("ss")
    shape.SetHistoName("err")
    shape.SetInputFile("f.root")
    shape.SetHistoPath("d/")
    assert printed(shape) == "\t \t Name: ss\t InputFile: f.root\t HistoName: err\t HistoPath: d/\n"
    assert shape.GetConstraintType() == Constraint.Gaussian
    shape.SetConstraintType("Poisson")
    assert shape.GetConstraintType() == Constraint.Poisson
    shape.SetConstraintType(0)
    assert (shape.GetConstraintType(), shape.GetErrorHist()) == (0, None)


def test_a_shape_factor_prints_its_initial_shape_range_and_constness_as_root_does() -> None:
    factor = ShapeFactor()
    factor.SetName("sf")
    assert printed(factor) == "\t \t Name: sf\n\t \t Value: 1  L(0 - 1000)\n"
    assert factor.HasInitialShape() is False
    factor.SetHistoName("shape")
    factor.SetInputFile("f.root")
    factor.SetConstant(True)
    factor.SetVal(2)
    factor.SetLow(0.5)
    factor.SetHigh(9)
    assert printed(factor) == (
        "\t \t Name: sf\n\t \t  Shape Hist Name: shape Shape Hist Path Name:  Shape Hist "
        "FileName: f.root\n\t \t Value: 2  L(0.5 - 9)\n\t \t ( Constant ): \n"
    )
    assert (factor.HasInitialShape(), factor.IsConstant(), factor.GetVal(), factor.GetLow(),
            factor.GetHigh(), factor.GetInitialShape()) == (True, True, 2.0, 0.5, 9.0, None)
    hist = TH1F("s", "", 1, 0, 1)
    factor.SetInitialShape(hist)
    assert factor.GetInitialShape() is hist


def test_a_stat_error_and_its_channel_configuration_print_as_root_prints_them() -> None:
    stat = StatError()
    stat.Activate(True)
    stat.SetHistoName("se")
    assert printed(stat) == "\t \t Activate: 1\t InputFile: \t HistoName: se\t histoPath: \n"
    stat.SetUseHisto()
    assert (stat.GetActivate(), stat.GetUseHisto()) == (True, True)
    config = StatErrorConfig()
    assert printed(config) == "\t \t RelErrorThreshold: 0.05\t ConstraintType: Poisson\n"
    config.SetConstraintType(Constraint.Gaussian)
    config.SetRelErrorThreshold(0.1)
    assert printed(config) == "\t \t RelErrorThreshold: 0.1\t ConstraintType: Gaussian\n"
    assert (config.GetRelErrorThreshold(), config.GetConstraintType()) == (0.1, 0)


def test_a_preprocess_function_is_the_factory_command_that_makes_it() -> None:
    made = PreprocessFunction("fn", "a*b", "a[1,0,2],b[2]")
    assert printed(made) == "\t \t Name: fn\t \t Expression: a*b\t \t Dependents: a[1,0,2],b[2]\n"
    assert made.GetCommand() == "expr::fn('a*b',{a[1,0,2],b[2]})"
    made.SetName("g")
    made.SetExpression("a")
    made.SetDependents("a[1]")
    assert (made.GetName(), made.GetExpression(), made.GetDependents()) == ("g", "a", "a[1]")


def test_a_constraint_type_is_named_and_found_by_name_or_refused() -> None:
    """``Gauss`` and ``Pois`` are names too; an empty name and an unknown one are errors."""
    assert (Constraint.Name(0), Constraint.Name(1), Constraint.Name(7)) == (
        "Gaussian", "Poisson", "")  # fmt: skip
    assert (Constraint.GetType("Gauss"), Constraint.GetType("Pois")) == (0, 1)


@pytest.mark.parametrize(("name", "said"), [
    ("", "Error: Given empty name for ConstraintType\n"),
    ("Wrong", "Error: Unknown name given for Constraint Type: Wrong\n"),
])  # fmt: skip
def test_a_constraint_type_root_does_not_know_is_refused_as_root_says(
        name: str, said: str, capsys: Any) -> None:  # fmt: skip
    with pytest.raises(HistFactoryError):
        Constraint.GetType(name)
    assert capsys.readouterr().out == said


def test_data_prints_where_its_histogram_is_read_from() -> None:
    data = Data("dh", "f.root", "p")
    assert printed(data) == (
        "\t \t InputFile: f.root\t HistoName: dh\t HistoPath: p\t HistoAddress: 0x0\n"
    )
    data.SetName("extra")
    hist = TH1F("h", "", 1, 0, 1)
    data.SetHisto(hist)
    copied = copy.deepcopy(data)
    assert (data.GetName(), data.GetHistoName(), copied.GetHisto() is not hist) == (
        "extra", "h", True)  # fmt: skip
    assert "HistoAddress: 0x" in printed(data) and "0x0" not in printed(data)


def signal() -> Sample:
    made = Sample("sig", "hsig", "f.root", "dir")
    made.ActivateStatError("err", "e.root", "ed")
    made.SetNormalizeByTheory(False)
    return made


SAMPLE = (
    "\t \t Name: sig\t \t Channel: {channel}\t NormalizeByTheory: False\t StatErrorActivate: "
    "False\n\t \t \t \t \t InputFile: f.root\t HistName: hsig\t HistoPath: dir\t HistoAddress: "
    "0x0\n\t \t \t StatError Activate: 1\t InputFile: f.root\t HistName: err\t HistoPath: ed\t "
    "HistoAddress: 0x0\n"
)


def test_a_sample_prints_its_histogram_and_stat_errors_place_as_root_does(capsys: Any) -> None:
    """ROOT prints the sample's own file for its statistical error histogram's, and so here."""
    made = signal()
    made.Print()
    assert capsys.readouterr().out == SAMPLE.format(channel="")
    stat = made.GetStatError()
    assert (stat.GetInputFile(), stat.GetHistoName(), stat.GetHistoPath(), stat.GetUseHisto()) == (
        "e.root", "err", "ed", True)  # fmt: skip
    assert (made.GetNormalizeByTheory(), made.HasStatError()) == (False, False)
    made.SetName("s2")
    made.SetInputFile("g.root")
    made.SetHistoName("h2")
    made.SetHistoPath("")
    made.SetChannelName("c")
    assert (made.GetName(), made.GetInputFile(), made.GetHistoName(), made.GetHistoPath(),
            made.GetChannelName()) == ("s2", "g.root", "h2", "", "c")  # fmt: skip


def test_a_sample_takes_its_systematics_by_their_numbers_or_as_objects_copied() -> None:
    made = Sample("s")
    made.AddOverallSys("a", 0.9, 1.1)
    given = OverallSys()
    given.SetName("b")
    made.AddOverallSys(given)
    made.AddNormFactor("mu", 1, 0, 5)
    norm = NormFactor()
    norm.SetName("nu")
    made.AddNormFactor(norm)
    made.AddHistoSys("h", "lo", "f.root", "", "hi", "f.root", "")
    made.AddHistoSys(histo_sys())
    made.AddHistoFactor("hf", "lo", "f.root", "p", "hi", "f.root", "p")
    made.AddShapeFactor("sf")
    made.AddShapeFactor("sg", 2.0, 1.0, 3.0)
    made.AddShapeFactor(ShapeFactor("sh"))
    made.AddShapeSys("ss", "Poisson", "err", "f.root", "d")
    made.AddShapeSys(ShapeSys("st"))
    assert [s.GetName() for s in made.GetOverallSysList()] == ["a", "b"]
    assert made.GetOverallSysList()[1] is not given
    assert [(n.GetName(), n.GetHigh()) for n in made.GetNormFactorList()] == [("mu", 5.0),
                                                                              ("nu", 1.0)]
    assert [s.GetHistoNameHigh() for s in made.GetHistoSysList()] == ["hi", "hi"]
    assert made.GetHistoFactorList()[0].GetHistoPathLow() == "p"
    assert [(f.GetName(), f.GetVal(), f.GetHigh()) for f in made.GetShapeFactorList()] == [
        ("sf", 1.0, 1000.0), ("sg", 2.0, 3.0), ("sh", 1.0, 1000.0)]  # fmt: skip
    assert [(s.GetName(), s.GetConstraintType()) for s in made.GetShapeSysList()] == [
        ("ss", 1), ("st", 0)]  # fmt: skip
    stat = StatError()
    stat.Activate()
    made.SetStatError(stat)
    assert made.GetStatError().GetActivate() and made.GetStatError() is not stat


def test_a_counting_sample_is_a_one_bin_histogram_of_its_value() -> None:
    made = Sample("bkg")
    made.SetValue(7.5)
    assert (made.GetHisto().GetName(), made.GetHistoName(), made.GetHisto().GetBinContent(1)) == (
        "bkg_hist", "bkg_hist", 7.5)  # fmt: skip
    assert copy.deepcopy(made).GetHisto() is not made.GetHisto()
    assert Sample().GetNormalizeByTheory() is False


CHANNEL = (
    "\t Channel Name: chan\t InputFile: c.root\n\t Data:\n\t \t InputFile: f.root\t HistoName: "
    "dh\t HistoPath: p\t HistoAddress: 0x0\n\t statErrorConfig:\n\t \t RelErrorThreshold: "
    "0.05\t ConstraintType: Poisson\n\t Samples: \n" + SAMPLE.format(channel="chan")
    + "\t End of Channel chan\n"
)  # fmt: skip


def channel() -> Channel:
    made = Channel("chan", "c.root")
    made.SetData("dh", "f.root", "p")
    made.AddSample(signal())
    return made


def test_a_channel_prints_its_data_configuration_and_samples_as_root_does() -> None:
    """The sample added is a copy, told its channel's name."""
    made = channel()
    assert printed(made) == CHANNEL
    assert printed(Channel("empty")).endswith("ConstraintType: Poisson\n\t End of Channel empty\n")
    assert made.GetSamples()[0].GetChannelName() == "chan"
    made.SetName("c2")
    made.SetInputFile("d.root")
    made.SetHistoPath("hp")
    assert (made.GetName(), made.GetInputFile(), made.GetHistoPath()) == ("c2", "d.root", "hp")


def test_a_channels_data_is_a_data_a_histogram_or_one_count() -> None:
    made = Channel("count")
    made.SetData(Data("x", "f.root"))
    assert made.GetData().GetInputFile() == "f.root"
    made.SetData(7)
    assert (made.GetData().GetHisto().GetName(), made.GetData().GetHisto().GetBinContent(1)) == (
        "count_data", 7.0)  # fmt: skip
    hist = TH1F("obs", "", 1, 0, 1)
    made.SetData(hist)
    assert made.GetData().GetHisto() is hist
    extra = Data("more")
    extra.SetName("extra")
    made.AddAdditionalData(extra)
    assert [d.GetName() for d in made.GetAdditionalData()] == ["extra"]


def test_a_channels_stat_error_configuration_is_a_threshold_and_a_type_or_a_copy() -> None:
    made = Channel("c")
    made.SetStatErrorConfig(0.1, "Gaussian")
    assert (made.GetStatErrorConfig().GetRelErrorThreshold(),
            made.GetStatErrorConfig().GetConstraintType()) == (0.1, 0)  # fmt: skip
    made.SetStatErrorConfig(0.2, 1)
    assert made.GetStatErrorConfig().GetConstraintType() == 1
    config = StatErrorConfig()
    made.SetStatErrorConfig(config)
    assert made.GetStatErrorConfig() is not config
    assert made.GetStatErrorConfig().GetRelErrorThreshold() == 0.05
    copied = copy.deepcopy(channel())
    assert printed(copied) == CHANNEL


TREE = (
    "Measurement Name: meas\t OutputFilePrefix: out/x\t POI: munu\t Lumi: 2\t LumiRelErr: 0.05\t "
    "BinLow: 1\t BinHigh: 4\t ExportOnly: 1\nConstant Params:  Lumi\nPreprocess Functions:  "
    "expr::fn('a*b',{a[1,0,2],b[2]})\nChannels:\n" + CHANNEL
    + "[#2] INFO:HistFactory -- End Measurement: meas\n"
)  # fmt: skip


def measurement() -> Measurement:
    made = Measurement("meas", "t")
    made.SetOutputFilePrefix("out/x")
    made.SetPOI("nu")
    made.SetPOI("mu")
    made.AddConstantParam("Lumi")
    made.AddConstantParam("Lumi")
    made.SetParamValue("mu", 2)
    made.SetParamValue("mu", 3)
    made.AddPreprocessFunction("fn", "a*b", "a[1,0,2],b[2]")
    made.SetLumi(2)
    made.SetLumiRelErr(0.05)
    made.SetBinLow(1)
    made.SetBinHigh(4)
    made.AddChannel(channel())
    return made


def test_a_measurement_says_what_it_is_set_and_prints_its_tree_as_root_does(capsys: Any) -> None:
    """A constant listed twice and a value changed are warned of; the tree is ROOT's."""
    made = measurement()
    assert capsys.readouterr().out == (
        "[#2] WARNING:HistFactory -- Warning: Setting parameter: Lumi to constant, but it is "
        "already listed as constant.  You may ignore this warning.\n"
        "[#2] INFO:HistFactory -- Setting parameter: mu value to 2\n"
        "[#2] WARNING:HistFactory -- Warning: Chainging parameter: mu value from: 2 to: 3\n"
        "[#2] INFO:HistFactory -- Setting parameter: mu value to 3\n"
    )
    made.PrintTree()
    assert capsys.readouterr().out == TREE
    stream = io.StringIO()
    Measurement("bare").PrintTree(stream)
    assert stream.getvalue() == (
        "Measurement Name: bare\t OutputFilePrefix: \t POI: \t Lumi: 1\t LumiRelErr: 0.1\t "
        "BinLow: 0\t BinHigh: 1\t ExportOnly: 1\n"
    )


def test_a_measurement_holds_its_parameters_lumi_bins_and_functions() -> None:
    made = measurement()
    assert (made.GetName(), made.GetTitle(), made.ClassName(), made.GetOutputFilePrefix()) == (
        "meas", "t", "RooStats::HistFactory::Measurement", "out/x")  # fmt: skip
    made.AddPOI("xi")
    assert (made.GetPOI(), made.GetPOI(2), made.GetPOIList()) == ("mu", "xi", ["mu", "nu", "xi"])
    assert (made.GetConstantParams(), made.GetParamValues()) == (["Lumi"], {"mu": 3.0})
    made.ClearConstantParams()
    made.ClearParamValues()
    assert (made.GetConstantParams(), made.GetParamValues()) == ([], {})
    assert (made.GetLumi(), made.GetLumiRelErr(), made.GetBinLow(), made.GetBinHigh()) == (
        2.0, 0.05, 1, 4)  # fmt: skip
    assert made.GetPreprocessFunctions() == ["expr::fn('a*b',{a[1,0,2],b[2]})"]
    made.AddFunctionObject(PreprocessFunction("g", "x", "x[1]"))
    assert [f.GetName() for f in made.GetFunctionObjects()] == ["fn", "g"]
    made.SetFunctionObjects([])
    made.SetName("m2")
    assert (made.GetFunctionObjects(), made.GetName(), made.GetInterpolationScheme()) == (
        [], "m2", "")  # fmt: skip
    asimov = Asimov("asimov_mu0")
    made.AddAsimovDataset(asimov)
    assert made.GetAsimovDatasets() == [asimov]


def test_a_measurement_says_how_each_systematic_is_constrained() -> None:
    made = Measurement("m")
    made.AddGammaSyst("a", 0.1)
    made.AddLogNormSyst("b", 0.2)
    made.AddUniformSyst("c")
    made.AddNoSyst("d")
    assert (made.GetGammaSyst(), made.GetLogNormSyst(), made.GetUniformSyst(),
            made.GetNoSyst()) == ({"a": 0.1}, {"b": 0.2}, {"c": 1.0}, {"d": 1.0})  # fmt: skip


def test_a_measurements_channels_are_copies_found_by_name(capsys: Any) -> None:
    """A channel it has not got is an error said as ROOT says it; a copy copies its channels."""
    made = measurement()
    capsys.readouterr()
    assert (made.HasChannel("chan"), made.HasChannel("nope")) == (True, False)
    assert made.GetChannel("chan") is made.GetChannels()[0]
    with pytest.raises(HistFactoryError, match="has no channel nope"):
        made.GetChannel("nope")
    assert capsys.readouterr().out == (
        "[#2] ERROR:HistFactory -- Error: Did not find channel: nope in measurement: meas\n"
    )
    copied = copy.deepcopy(made)
    assert copied.GetChannels()[0] is not made.GetChannels()[0]
    copied.AddPOI("other")
    assert made.GetPOIList() == ["mu", "nu"]


def asimov_workspace() -> Any:
    from xrdroot.roofit.variables import RooRealVar
    from xrdroot.roofit.workspace import RooWorkspace

    ws = RooWorkspace("w")
    ws.Import(RooRealVar("mu", "mu", 1, 0, 5), RooCmdArg("Silence"))
    ws.Import(RooRealVar("alpha", "alpha", 0, -5, 5), RooCmdArg("Silence"))
    return ws


def test_an_asimov_sets_its_values_then_fixes_its_parameters_saying_so(capsys: Any) -> None:
    asimov = Asimov("asimov_mu0")
    asimov.SetParamValue("mu", 0.0)
    asimov.SetParamValue("alpha", 1.0)
    asimov.SetFixedParam("mu")
    asimov.SetFixedParam("alpha", False)
    ws = asimov_workspace()
    asimov.ConfigureWorkspace(ws)
    assert capsys.readouterr().out == (
        "Configuring Asimov Dataset: Setting alpha = 1\n"
        "Configuring Asimov Dataset: Setting mu = 0\n"
        "Configuring Asimov Dataset: Setting alpha to constant \n"
        "Configuring Asimov Dataset: Setting mu to constant \n"
    )
    assert (ws.var("mu").getVal(), ws.var("mu").isConstant(), ws.var("alpha").isConstant()) == (
        0.0, True, False)  # fmt: skip
    assert (asimov.GetParamsToSet(), asimov.GetParamsToFix()) == (
        {"mu": 0.0, "alpha": 1.0}, {"mu": True, "alpha": False})  # fmt: skip
    asimov.SetName("a2")
    assert asimov.GetName() == "a2"


@pytest.mark.parametrize(("setup", "said"), [
    (lambda a: a.SetParamValue("nu", 1.0), "Error: Trying to set variable: 0x0 to a specific "
     "value in creation of asimov dataset: a but this variable doesn't appear to exist in the "
     "workspace\n"),
    (lambda a: a.SetFixedParam("nu"), "Error: Trying to set variable: 0x0 constant in creation "
     "of asimov dataset: a but this variable doesn't appear to exist in the workspace\n"),
    (lambda a: a.SetParamValue("mu", 9.0), "Error: Attempting to set variable: 0x"),
])  # fmt: skip
def test_an_asimov_of_a_parameter_missing_or_out_of_range_is_refused_as_root_says(
        setup: Any, said: str, capsys: Any) -> None:  # fmt: skip
    asimov = Asimov("a")
    setup(asimov)
    with pytest.raises(HistFactoryError):
        asimov.ConfigureWorkspace(asimov_workspace())
    out = capsys.readouterr().out
    assert out.startswith(said)
    if "range" in out:
        assert out.endswith(" to value: 9, however it appears that this is not withn the "
                            "variable's range: [0, 5]\n")  # fmt: skip


def test_a_measurement_does_not_write_its_xml_configuration() -> None:
    with pytest.raises(UnsupportedFeatureError, match="PrintXML"):
        measurement().PrintXML("xml", "prefix")

