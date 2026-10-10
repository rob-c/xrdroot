"""A ``Measurement`` written as ROOT 6.40 streams one, byte for byte, and read back.

The donors are ROOT's own: ``hf-measurement-6.40.root`` holds the measurement
``hf001_example.C`` builds from ``hf-example-6.40.root``, and
``hf-allkinds-6.40.root`` one of every kind of piece, written whole. Outside
the histograms inside it the writer's bytes are ROOT's to the last one; inside
them they differ only where this writer differs from ROOT 6.40 for any
histogram - the ``TObject`` bits ROOT 6.40 no longer writes and
``TAttMarker``'s version. ROOT 6.40 reads the written files back as its own
(checked by hand against the oracle, which the tests do not need), and this
reader reads ROOT's files and its own alike.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

import pytest

from support import plain
from xrdroot import create, open_root
from xrdroot.hist import Histogram
from xrdroot.histfactory.channel import Channel
from xrdroot.histfactory.layouts import LAYOUTS, described, full
from xrdroot.histfactory.measurement import Measurement
from xrdroot.histfactory.model import Data, Sample
from xrdroot.histfactory.output import write_measurement
from xrdroot.histfactory.shapes import Asimov, ShapeFactor, ShapeSys
from xrdroot.histfactory.streamed import stream
from xrdroot.histfactory.systematics import Constraint, HistoFactor, HistoSys
from xrdroot.pyroot.core import TFile
from xrdroot.pyroot.core.hists import TH1F

DATA = pathlib.Path(__file__).parent / "data"

#: What a byte of ours and of ROOT's are wherever a histogram's record differs: TAttMarker's
#: version, 2 here and 3 in ROOT 6.40, and the TObject bits ROOT 6.40 masks on writing -
#: ``kIsOnHeap | kNotDeleted`` here; there none, a list's ``kIsOwner``, or ``kMustCleanup``
#: for a histogram a macro made in a directory.
ALLOWED = {(0x02, 0x03), (0x03, 0x00), (0x00, 0x40), (0x00, 0x08)}


def hf001() -> Measurement:
    """``hf001_example.C``'s measurement, its input where the macro has it."""
    path = "./data/example.root"
    meas = Measurement("meas", "meas")
    meas.SetOutputFilePrefix("./results/example_UsingC")
    meas.SetPOI("SigXsecOverSM")
    meas.AddConstantParam("alpha_syst1")
    meas.AddConstantParam("Lumi")
    meas.SetLumi(1.0)
    meas.SetLumiRelErr(0.10)
    meas.SetBinHigh(2)
    chan = Channel("channel1")
    chan.SetData("data", path)
    chan.SetStatErrorConfig(0.05, "Poisson")
    signal = Sample("signal", "signal", path)
    signal.AddOverallSys("syst1", 0.95, 1.05)
    signal.AddNormFactor("SigXsecOverSM", 1, 0, 3)
    chan.AddSample(signal)
    background1 = Sample("background1", "background1", path)
    background1.ActivateStatError("background1_statUncert", path)
    background1.AddOverallSys("syst2", 0.95, 1.05)
    chan.AddSample(background1)
    background2 = Sample("background2", "background2", path)
    background2.ActivateStatError()
    background2.AddOverallSys("syst3", 0.95, 1.05)
    chan.AddSample(background2)
    meas.AddChannel(chan)
    meas.CollectHistograms()
    return meas


def h(name: str, a: float, b: float) -> Any:
    made = TH1F(name, name, 2, 1, 3)
    made.SetBinContent(1, a)
    made.SetBinContent(2, b)
    return made


def _two_sided(kind: Any, name: str, low: Any, high: Any) -> Any:
    made = kind()
    made.SetName(name)
    made.SetHistoLow(low)
    made.SetHistoHigh(high)
    return made


def _sample_a1() -> Sample:
    """Every kind of systematic, two where the order of two matters."""
    sA1 = Sample("sA1")
    sA1.SetHisto(h("hA1", 5, 6))
    sA1.AddOverallSys("os1", 0.9, 1.1)
    sA1.AddOverallSys("os2", 0.8, 1.2)
    sA1.AddNormFactor("mu", 1, 0, 3)
    sA1.AddNormFactor("nu", 2, 1, 5)
    sA1.AddHistoSys(_two_sided(HistoSys, "hs1", h("hs1lo", 4, 5), h("hs1hi", 6, 7)))
    sA1.AddHistoSys(_two_sided(HistoSys, "hs2", h("hs2lo", 4.5, 5.5), h("hs2hi", 5.5, 6.5)))
    sA1.AddHistoFactor(_two_sided(HistoFactor, "hf1", h("hf1lo", 1, 2), h("hf1hi", 2, 3)))
    ss1 = ShapeSys()
    ss1.SetName("ss1")
    ss1.SetConstraintType(Constraint.Poisson)
    ss1.SetErrorHist(h("ss1err", 0.1, 0.2))
    sA1.AddShapeSys(ss1)
    sf1 = ShapeFactor()
    sf1.SetName("sf1")
    sf1.SetConstant(True)
    sf1.SetInitialShape(h("sf1init", 1, 2))
    sf1.SetVal(2.5)
    sf1.SetLow(0.5)
    sf1.SetHigh(7.0)
    sA1.AddShapeFactor(sf1)
    sA1.GetStatError().SetErrorHist(h("errA1", 0.05, 0.06))
    sA1.ActivateStatError("errA1", "")
    return sA1


def allkinds() -> Measurement:
    """The measurement ``hf-allkinds-6.40.root`` holds, as its macro built it."""
    meas = Measurement("all", "all kinds")
    meas.SetOutputFilePrefix("./results/all")
    meas.AddPOI("mu")
    meas.AddPOI("nu")
    meas.AddConstantParam("alpha_a")
    meas.AddConstantParam("Lumi")
    meas.SetParamValue("mu", 1.5)
    meas.SetParamValue("alpha_a", 0.25)
    meas.AddPreprocessFunction("f1", "mu*2", "mu")
    meas.AddPreprocessFunction("f2", "nu+1", "nu")
    asimov = Asimov("asi")
    asimov.SetFixedParam("mu")
    asimov.SetFixedParam("nu", False)
    asimov.SetParamValue("alpha_a", 0.5)
    meas.AddAsimovDataset(asimov)
    meas.AddAsimovDataset(Asimov("other"))
    meas.SetLumi(2.0)
    meas.SetLumiRelErr(0.05)
    meas.SetBinLow(1)
    meas.SetBinHigh(2)
    meas.AddGammaSyst("g", 0.1)
    meas.AddLogNormSyst("l", 0.2)
    meas.AddUniformSyst("u")
    meas.AddNoSyst("n")
    chA = Channel("chA")
    dA = Data()
    dA.SetName("dataA")
    dA.SetHisto(h("dA", 12, 15))
    chA.SetData(dA)
    extra = Data()
    extra.SetName("extra")
    extra.SetHisto(h("dX", 3, 4))
    chA.AddAdditionalData(extra)
    chA.SetStatErrorConfig(0.03, "Gaussian")
    chA.AddSample(_sample_a1())
    sA2 = Sample("sA2")
    sA2.SetHisto(h("hA2", 7, 8))
    sA2.SetNormalizeByTheory(False)
    sA2.AddShapeFactor("sf2")
    sA2.AddShapeSys("ss2", Constraint.Gaussian, "hss2", "f.root", "p/")
    chA.AddSample(sA2)
    meas.AddChannel(chA)
    chB = Channel("chB")
    dB = Data()
    dB.SetName("dataB")
    dB.SetHisto(h("dB", 20, 25))
    chB.SetData(dB)
    sB1 = Sample("sB1")
    sB1.SetHisto(h("hB1", 9, 10))
    sB1.ActivateStatError()
    chB.AddSample(sB1)
    sB2 = Sample("sB2", "hB2", "in.root", "d/")
    sB2.SetHisto(h("hB2", 11, 12))
    sB2.AddHistoSys("hsB", "lo", "a.root", "p/", "hi", "b.root", "q/")
    chB.AddSample(sB2)
    meas.AddChannel(chB)
    return meas


def assert_roots_bytes(measurement: Measurement, donor: str, name: str) -> None:
    """The measurement streamed for the donor's key is ROOT's, histograms' records aside."""
    with open_root(str(DATA / donor)) as f:
        key = f._key(name)
        theirs = key.payload(f._source)
    made = stream(measurement, key.keylen)
    ours = bytes(made.buf.data)
    assert len(ours) == len(theirs)
    differing = {i for i, (a, b) in enumerate(zip(ours, theirs, strict=True)) if a != b}
    inside = {i for lo, hi in made.histograms for i in range(lo, hi)}
    assert differing <= inside
    assert {(ours[i], theirs[i]) for i in differing} <= ALLOWED


def test_hf001s_measurement_is_streamed_as_root_streamed_it(tmp_path: Any, monkeypatch: Any,
                                                             capsys: Any) -> None:  # fmt: skip
    """Its histograms read from ROOT's example input, by the paths the macro gives."""
    (tmp_path / "data").mkdir()
    shutil.copy(DATA / "hf-example-6.40.root", tmp_path / "data" / "example.root")
    monkeypatch.chdir(tmp_path)
    assert_roots_bytes(hf001(), "hf-measurement-6.40.root", "meas")
    assert "Getting histogram ./data/example.root:/signal" in capsys.readouterr().out


def test_a_measurement_of_every_kind_of_piece_is_streamed_as_root_streamed_it() -> None:
    """Strings, maps and a base class inside vectors written member-wise, null and repeated
    histogram pointers, empty vectors and maps, bools and an enum."""
    assert_roots_bytes(allkinds(), "hf-allkinds-6.40.root", "all")


def flat(value: Any) -> Any:
    """A read measurement with each histogram as its name and contents, comparable by ``==``."""
    if isinstance(value, Histogram):
        return (value.name, value.values().tolist())
    if isinstance(value, dict):
        return {key: flat(item) for key, item in value.items()}
    if isinstance(value, list):
        return [flat(item) for item in value]
    return plain(value)


def test_a_written_measurement_reads_back_as_roots_does(tmp_path: Any) -> None:
    """Both by this reader, member by member, ROOT's file lacking the descriptions of the
    classes it streamed member-wise - this reader knows those."""
    filename = str(tmp_path / "all.root")
    handle = TFile.Open(filename, "RECREATE")
    handle.WriteTObject(allkinds(), "all")
    handle.Close()
    with open_root(filename) as ours, open_root(str(DATA / "hf-allkinds-6.40.root")) as theirs:
        mine, roots = ours["all"], theirs["all"]
        assert flat(mine) == flat(roots)
        assert set(ours._source.streamers()) >= {full(name) for name in LAYOUTS}
    base = full("HistogramUncertaintyBase")
    sample = mine["fChannels"][0]["fSamples"][0]
    assert [one[base]["fName"] for one in sample["fHistoSysList"]] == ["hs1", "hs2"]
    assert sample["fHistoSysList"][0][base]["fhLow"].values().tolist() == [4.0, 5.0]
    assert plain(sample["fShapeFactorList"][0]) == {
        base: {"fName": "sf1", "fInputFileLow": "", "fHistoNameLow": "", "fHistoPathLow": "",
               "fInputFileHigh": "", "fHistoNameHigh": "", "fHistoPathHigh": "", "fhLow": None,
               "fhHigh": sample["fShapeFactorList"][0][base]["fhHigh"]},
        "fConstant": True, "fHasInitialShape": False, "fVal": 2.5, "fLow": 0.5, "fHigh": 7.0,
    }  # fmt: skip
    assert plain(mine["fAsimovDatasets"]) == [
        {"fName": "asi", "fParamsToFix": {"mu": True, "nu": False},
         "fParamValsToSet": {"alpha_a": 0.5}},
        {"fName": "other", "fParamsToFix": {}, "fParamValsToSet": {}},
    ]  # fmt: skip
    assert plain(mine["fParamValues"]) == {"alpha_a": 0.25, "mu": 1.5}
    assert (mine["TNamed"]["fBits"], sample["fhCountingHist"]) == (0, None)


def test_a_channel_without_samples_reads_back_with_none(tmp_path: Any) -> None:
    """Its samples' columns have no rows, so none of them - strings, vectors, maps - is read."""
    meas = Measurement("bare", "a channel of data alone")
    chan = Channel("chC")
    chan.SetData(h("dC", 1, 2))
    meas.AddChannel(chan)
    with create(str(tmp_path / "bare.root")) as out:
        out["bare"] = meas
    with open_root(str(tmp_path / "bare.root")) as f:
        back = f["bare"]
        assert f._key("bare").title == "a channel of data alone"
    channel = back["fChannels"][0]
    assert (channel["fSamples"], channel["fAdditionalData"]) == ([], [])
    assert channel["fData"]["fhData"]["fHist"].values().tolist() == [1.0, 2.0]


def test_the_descriptions_carried_are_the_layouts_with_the_measurements_tnamed() -> None:
    found = described()
    assert set(found) == {full(name) for name in LAYOUTS}
    assert list(found[full("Measurement")])[:2] == ["TNamed", "fOutputFilePrefix"]
    base = found[full("ShapeSys")][full("HistogramUncertaintyBase")]
    assert (base.stype, base.typename) == (0, "BASE")
    enum = found[full("StatErrorConfig")]["fConstraintType"]
    assert (enum.stype, enum.typename) == (3, full("Constraint::Type"))


def test_writing_a_measurement_writes_it_as_given_and_renames_the_given_one(
        tmp_path: Any, capsys: Any) -> None:  # fmt: skip
    """``Measurement::writeToFile``: the histograms into their directories, the measurement
    as it was before them, and the one in hand told where they went."""
    meas = allkinds()
    chA, chB = meas.GetChannels()
    chA.GetSamples()[1].GetShapeSysList()[0].SetErrorHist(h("ss2err", 0.1, 0.1))
    hsB = chB.GetSamples()[1].GetHistoSysList()[0]
    hsB.SetHistoLow(h("lo", 1, 1))
    hsB.SetHistoHigh(h("hi", 2, 2))
    filename = str(tmp_path / "out.root")
    handle = TFile.Open(filename, "RECREATE")
    write_measurement(meas, handle)
    handle.Close()
    with open_root(filename) as f:
        assert sorted(f.keys()) == ["all", "chA_hists", "chB_hists"]
        data = f["all"]["fChannels"][0]["fData"]
        assert (data["fInputFile"], data["fHistoName"], data["fHistoPath"]) == ("", "dA", "")
    given = meas.GetChannels()[0].GetData()
    assert (given.GetInputFile(), given.GetHistoPath()) == (filename, "/chA_hists/data/")
    assert capsys.readouterr().out.endswith("[#2] PROGRESS:HistFactory -- Saved Measurement\n")


@pytest.mark.parametrize("name", ["hf-measurement-6.40.root"])
def test_roots_own_output_file_reads_without_describing_every_class(name: str) -> None:
    """ROOT describes only the classes it streamed whole; the rest are known here."""
    with open_root(str(DATA / name)) as f:
        meas = f["meas"]
    assert [s["fName"] for s in meas["fChannels"][0]["fSamples"]] == [
        "signal", "background1", "background2"]  # fmt: skip
    assert meas["fChannels"][0]["fData"]["fInputFile"] == "./data/example.root"
