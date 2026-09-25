"""Whole analyses, the way ROOT's tutorials write them, checked against NumPy by hand.

``df102_NanoAODDimuonAnalysis`` is the model: events with exactly two muons
of opposite charge, the invariant mass of the pair, a histogram of it. It
runs here over NanoAOD-like muons written to a tree and to RNTuples, and over
the thousand CMS open-data events ROOT itself wrote as an RNTuple, and the
masses are compared with the same arithmetic written out in NumPy.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import xrdroot
from frames import DATA, muon_columns, write_muons
from xrdroot import Jagged, RDataFrame
from xrdroot.rdf import SampleInfo, vecops

NANOAOD = str(DATA / "rntuple" / "Run2012BC_DoubleMuParked_Muons_1000evts_rntuple_v1-0-0-0.root")

#: df102's histogram of the dimuon mass: 30 000 bins, log-spaced there, even here.
MODEL = ("Dimuon_mass", "Dimuon mass;m_{#mu#mu} (GeV);N_{Events}", 300, 0.25, 300)


def _dimuon(df: RDataFrame) -> RDataFrame:
    two = df.Filter("nMuon == 2", "Events with exactly two muons")
    opposite = two.Filter("Muon_charge[0] != Muon_charge[1]", "Muons with opposite charge")
    return opposite.Define("Dimuon_mass", "InvariantMass(Muon_pt, Muon_eta, Muon_phi, Muon_mass)")


def _by_hand(columns: dict) -> np.ndarray:
    """The dimuon masses, in NumPy, one event at a time as the C++ loop takes them."""
    masses = []
    for event in range(len(columns["nMuon"])):
        rows = [columns[f"Muon_{name}"][event] for name in ("pt", "eta", "phi", "mass", "charge")]
        if len(rows[0]) != 2 or rows[4][0] == rows[4][1]:
            continue
        pt, eta, phi, mass = rows[:4]
        x, y, z = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        e = np.sqrt(x * x + y * y + z * z + mass * mass)
        total = [part.sum(dtype=np.float32) for part in (x, y, z, e)]
        masses.append(np.sqrt(total[3] ** 2 - total[0] ** 2 - total[1] ** 2 - total[2] ** 2))
    return np.asarray(masses, np.float32)


def _counts(masses: np.ndarray) -> np.ndarray:
    return np.histogram(masses, bins=np.linspace(0.25, 300, 301))[0]


def test_the_dimuon_spectrum_of_a_tree_is_numpy_by_hand(tmp_path):
    with RDataFrame("Events", write_muons(tmp_path / "muons.root"), step=1000) as df:
        spectrum = _dimuon(df)
        h, masses, report = (
            spectrum.Histo1D(MODEL, "Dimuon_mass"),
            spectrum.Take("Dimuon_mass"),
            df.Report(),
        )
        wanted = _by_hand(muon_columns(5000))
        assert np.allclose(masses.GetValue(), wanted, rtol=1e-6)
        assert h.GetValue().values().tolist() == _counts(masses.GetValue()).tolist()
        assert h.GetValue().entries == len(wanted)
        cuts = report.GetValue()
        assert [cut.name for cut in cuts] == [
            "Events with exactly two muons",
            "Muons with opposite charge",
        ]
        assert cuts[0].all == 5000 and cuts[1].passed == len(wanted)


def _mass_of_pairs(pt, eta, phi, mass):
    return vecops.InvariantMass(pt, eta, phi, mass)


def _two(nMuon):
    return nMuon == 2


def _opposite(Muon_charge):
    return vecops.Take(Muon_charge, 1, default=0).content * 0 + (vecops.Product(Muon_charge) < 0)


def test_the_same_analysis_in_python_callables_gives_the_same_masses(tmp_path):
    with RDataFrame("Events", write_muons(tmp_path / "muons.root"), step=700) as df:
        strings = _dimuon(df).Take("Dimuon_mass")
        columns = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"]
        called = df.Filter(_two).Filter(lambda Muon_charge: vecops.Product(Muon_charge) < 0)
        masses = called.Define("m", _mass_of_pairs, columns).Take("m")
        assert masses.GetValue().tolist() == strings.GetValue().tolist()


def test_the_dimuon_spectrum_of_an_rntuple_split_across_files_and_workers(tmp_path):
    files = [write_muons(tmp_path / f"part{i}.root", 3000, rntuple=True) for i in range(2)]
    wanted = _by_hand(muon_columns(3000))
    with RDataFrame("Events", files, step=800, workers=2) as parallel:
        two = _dimuon(parallel).Histo1D(MODEL, "Dimuon_mass").GetValue()
    with RDataFrame("Events", files, step=800) as serial:
        one = _dimuon(serial).Histo1D(MODEL, "Dimuon_mass").GetValue()
        assert one.entries == two.entries == 2 * len(wanted)
        assert one.values(flow=True).tolist() == two.values(flow=True).tolist()
        assert one.mean() == two.mean()
        assert serial.Count().GetValue() == 6000
        assert serial.GetColumnType("Muon_pt") == "std::vector<float>"  # as this writer declares it
        assert "RNTuple 'Events' of " in serial.Describe().splitlines()[0]


def test_roots_own_open_data_dimuons(tmp_path):
    with xrdroot.open_root(NANOAOD) as f:
        events = f["Events"]
        columns = events.arrays(
            ["nMuon", "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass", "Muon_charge"]
        )
        df = RDataFrame(events)
        masses = _dimuon(df).Take("Dimuon_mass").GetValue()
        assert np.allclose(masses, _by_hand(columns), rtol=1e-5)
        assert len(masses) > 100
        assert df.GetColumnType("nMuon") == "ROOT::RNTupleCardinality<std::uint32_t>"
        assert df.GetColumnType("Muon_charge") == "ROOT::VecOps::RVec<std::int32_t>"
        assert df.Describe().splitlines()[0] == f"Dataframe from RNTuple 'Events' in {NANOAOD}"


def test_an_rntuple_by_name_and_its_samples(tmp_path):
    path = write_muons(tmp_path / "muons.root", 100, rntuple=True)
    with RDataFrame("Events", path) as df:
        info = (
            df.DefinePerSample("where", lambda sample: sample.AsString()).Take("where").GetValue()
        )
        assert set(info.tolist()) == {f"{path}/Events"}
    with xrdroot.open_root(path) as f:
        per = RDataFrame(f["Events"]).DefinePerSample(
            "first", lambda sample: sample.EntryRange()[1]
        )
        assert set(per.Take("first").GetValue().tolist()) == {100}


def test_a_tree_handed_over_describes_itself_and_its_samples(tmp_path):
    path = write_muons(tmp_path / "muons.root", 50)
    with xrdroot.open_root(path) as f:
        df = RDataFrame(f["Events"])
        assert repr(df).startswith(f"<RDataFrame over TTree 'Events' in {path}")
        tagged = df.DefinePerSample("muons", lambda sample: sample.Contains("muons")).Sum("muons")
        assert tagged.GetValue() == 50


def test_sample_information_says_where_entries_are_from():
    info = SampleInfo("a.root/Events", (10, 20))
    assert (info.as_string(), info.contains("a.root"), info.entry_range()) == (
        "a.root/Events",
        True,
        (10, 20),
    )
    assert repr(info) == "<SampleInfo 'a.root/Events' entries 10 to 20>"
    assert (
        RDataFrame(4)
        .DefinePerSample("s", lambda sample: sample.AsString() == "")
        .Sum("s")
        .GetValue()
        == 4
    )
    assert repr(RDataFrame(4)) == "<RDataFrame over 4 empty entries, 0 columns>"


def test_files_are_found_by_glob_and_a_glob_of_nothing_is_refused(tmp_path):
    for i in range(3):
        write_muons(tmp_path / f"run{i}.root", 10)
    with RDataFrame("Events", str(tmp_path / "run*.root")) as df:
        assert df.Count().GetValue() == 30
    with pytest.raises(FileNotFoundError, match="matches no file"):
        RDataFrame("Events", str(tmp_path / "nothing*.root"))
    with pytest.raises(ValueError, match="at least one file"):
        RDataFrame("Events", [])


def test_a_collection_can_be_given_back_by_a_callable_and_used_in_strings(tmp_path):
    with RDataFrame("Events", write_muons(tmp_path / "muons.root", 200)) as df:
        leading = df.Define("sorted", lambda Muon_pt: vecops.Reverse(vecops.Sort(Muon_pt)))
        first = leading.Filter("nMuon > 0").Define("lead", "sorted[0]").Take("lead").GetValue()
        wanted = [row.max() for row in muon_columns(200)["Muon_pt"] if len(row)]
        assert first.tolist() == wanted
        good = df.Define("good", "Muon_pt[Muon_pt > 30 && abs(Muon_eta) < 2.4]")
        counts = good.Define("n_good", "good.size()").Take("n_good").GetValue()
        pts, etas = muon_columns(200)["Muon_pt"], muon_columns(200)["Muon_eta"]
        assert counts.tolist() == [
            int(((p > 30) & (np.abs(e) < 2.4)).sum()) for p, e in zip(pts, etas)
        ]
        assert isinstance(good.Take("good").GetValue(), Jagged)


def test_rntuples_in_several_files_travel_to_workers_by_where_they_are(tmp_path):
    files = [write_muons(tmp_path / f"part{i}.root", 20, rntuple=True) for i in range(2)]
    with RDataFrame("Events", files) as df:
        source = df._graph.source
        again = pickle.loads(pickle.dumps(source))
        assert (len(again), again.boundaries()) == (40, [0, 20, 40])
        again.close()
        everything = df.AsNumpy().GetValue()
        assert list(everything) == [
            "nMuon",
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_mass",
            "Muon_charge",
        ]
