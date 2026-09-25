"""Writing every histogram class, profiles and efficiencies, as ROOT lays them out.

The layouts come from files ROOT 6.22 and 6.24 wrote - ``tprofile.root``,
``tgme.root``, ``tconfidence-level.root`` and ``uproot-issue-227b.root`` -
and a written file must describe each class exactly as those donors do. The
classes no donor here holds are two bases and a checksum, and the checksum
is ROOT's ``TClass::GetCheckSum``, which is worked out here from ROOT's
source and held to every checksum the donors carry, and to the ones ROOT's
own streamer dump in go-hep has for the classes this suite has no file of.

A profile ROOT wrote is written back byte for byte. Everything else is
booked, filled, written and read back to the same bins, errors, running
sums and entries.
"""

from __future__ import annotations

import copy
import io
import pathlib

import numpy as np
import pytest

from support import plain
from xrdroot import Efficiency, Histogram, Profile, create, open_root
from xrdroot.efficiency import USE_WEIGHTS
from xrdroot.winfo import INFOS
from xrdroot.writer import _payload

DATA = pathlib.Path(__file__).parent / "data"

#: The histogram family, harvested or derived: every class the checksum covers.
FAMILY = [
    *(f"TH{dimensions}{kind}" for dimensions in (1, 2, 3) for kind in "CSIFD"),
    *("TH1", "TH2", "TH3", "TAtt3D", "TProfile", "TProfile2D", "TProfile3D", "TEfficiency"),
]

#: ROOT's checksums for the classes derived rather than harvested, as ROOT's
#: streamer dump in go-hep (``groot/rdict/cxx_root_streamers_gen.go``) has them.
ROOT_DUMP = {
    "TH1C": 0x36F6E4AD,
    "TH1S": 0x8C4D9DCB,
    "TH1I": 0x627564F6,
    "TH2C": 0xBD0010FE,
    "TH2S": 0x1256CA1C,
    "TH2I": 0xE87E9147,
    "TH2F": 0x689CC295,
}

#: The integer types ``TClass::GetCheckSum`` does not count as an enum.
INTEGERS = ("int", "Int_t", "unsigned int", "UInt_t")


def checksum(name, elements):
    """``TClass::GetCheckSum`` over a class's streamer elements.

    The class name, then for each base its name and its own checksum, and
    for each member its name, its type, its array dimensions and the counter
    a ``[n]`` comment names - an enum being counted once more before it.
    """
    found = 0

    def mix(text):
        nonlocal found
        for byte in text.encode():
            found = (found * 3 + byte) & 0xFFFFFFFF

    mix(name)
    for kind, member, title, stype, _size, _length, dims, maxima, typename, _extra in elements:
        if kind == "TStreamerBase":
            mix(member)
            found = (found * 3 + (maxima[1] & 0xFFFFFFFF)) & 0xFFFFFFFF
            continue
        if stype == 3 and typename not in INTEGERS:
            found = (found * 3 + 1) & 0xFFFFFFFF
        mix(member)
        mix(typename)
        for dimension in maxima[:dims]:
            found = (found * 3 + dimension) & 0xFFFFFFFF
        if title.startswith("["):
            mix(title[1 : title.index("]")])
    return found


def written(**objects) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        for name, obj in objects.items():
            out[name] = obj
    return buf.getvalue()


def read_back(data: bytes):
    return open_root(io.BytesIO(data))


def moments(histogram):
    """Every running sum a histogram keeps, by name."""
    return {name: home[name] for name, home in histogram._moment_homes().items()}


def test_every_checksum_of_the_histogram_family_is_roots_calculation_of_it():
    for name in FAMILY:
        stored, _version, elements = INFOS[name]
        assert checksum(name, elements) == stored, name
    for name, root in ROOT_DUMP.items():
        assert INFOS[name][0] == root, name


def test_a_written_file_describes_the_histogram_family_as_its_donors_do():
    theirs = {}
    for donor in ("tprofile", "tgme", "uproot-issue-227b", "tconfidence-level"):
        with open_root(str(DATA / f"{donor}.root")) as handle:
            theirs.update(handle._source.streamers())
    objects = {
        "h1": Histogram.book("h1", (1, 0, 1), kind="F"),
        "p1": Profile.book("p1", (1, 0, 1)),
        "p2": Profile.book("p2", (1, 0, 1), (1, 0, 1)),
        "p3": Profile.book("p3", (1, 0, 1), (1, 0, 1), (1, 0, 1)),
        "e": Efficiency.book("e", (1, 0, 1)),
    }
    with read_back(written(**objects)) as back:
        ours = back._source.streamers()
    for classname in ("TH1", "TH1D", "TH1F", "TH2", "TH2D", "TH3", "TH3D", "TProfile"):
        assert classname in ours
    for classname in ("TProfile2D", "TProfile3D", "TAtt3D", "TEfficiency"):
        members = ours[classname]
        for name, member in members.items():
            other = theirs[classname][name]
            assert (member.title, member.stype, member.typename, member.count) == (
                other.title,
                other.stype,
                other.typename,
                other.count,
            )


def test_a_profile_root_wrote_is_written_back_byte_for_byte():
    for donor, name in (
        ("tprofile", "p1d"),
        ("tprofile", "p2d"),
        ("uproot-issue-227b", "hprof3d"),
    ):
        with open_root(str(DATA / f"{donor}.root")) as handle:
            profile = handle[name]
            theirs = handle._key(name).payload(handle._source)
        assert _payload(profile)[1] == theirs, name


@pytest.mark.parametrize("kind", "CSIFD")
@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_every_histogram_class_writes_and_reads_back_whole(kind, dimensions):
    axes = [(3, 0.0, 3.0), [0.0, 1.0, 5.0], (2, -1.0, 1.0)][:dimensions]
    h = Histogram.book("h", *axes, title="a title;x;y;z", kind=kind)
    rng = np.random.default_rng(dimensions)
    points = [rng.uniform(-1.5, 5.5, 200) for _ in axes]
    h.fill(*points, weight=rng.integers(1, 4, 200).astype(float))
    with read_back(written(h=h)) as back:
        again = back["h"]
    assert again.classname == f"TH{dimensions}{kind}" and again.title == "a title"
    assert again.values(flow=True).tolist() == h.values(flow=True).tolist()
    assert again.variances(flow=True).tolist() == h.variances(flow=True).tolist()
    assert again.entries == h.entries and moments(again) == moments(h)
    assert [axis.title for axis in again.axes] == ["x", "y", "z"][:dimensions]


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_every_profile_class_writes_and_reads_back_whole(dimensions):
    axes = [(3, 0.0, 3.0), [0.0, 1.0, 5.0], (2, -1.0, 1.0)][:dimensions]
    p = Profile.book("p", *axes, error_option="s", value_range=(0, 10))
    rng = np.random.default_rng(dimensions)
    points = [rng.uniform(-1.0, 5.0, 100) for _ in range(dimensions + 1)]
    p.fill(*points, weight=rng.uniform(0.5, 2.0, 100))
    with read_back(written(p=p)) as back:
        again = back["p"]
    assert again.classname == p.classname and again.error_mode == "s"
    assert again.values(flow=True).tolist() == p.values(flow=True).tolist()
    assert again.errors(flow=True).tolist() == p.errors(flow=True).tolist()
    assert plain(again.members["fBinSumw2"]) == plain(p.members["fBinSumw2"])
    assert again.entries == p.entries and moments(again) == moments(p)


def test_an_efficiency_writes_and_reads_back_whole():
    eff = Efficiency.book("trigger", (4, 0, 4), (2, 0, 2), title="Trigger")
    eff.fill([True, False, True], [0.5, 0.5, 3.5], [0.5, 1.5, 1.5])
    with read_back(written(trigger=eff)) as back:
        again = back["trigger"]
    assert (again.name, again.title, again.method) == ("trigger", "Trigger", "clopper-pearson")
    assert again.values().tolist() == eff.values().tolist()
    assert [part.tolist() for part in again.intervals()] == [
        part.tolist() for part in eff.intervals()
    ]
    assert again.total.name == "trigger_total" and again.total.entries == 3
    heavy = Efficiency.book("heavy", (1, 0, 1))
    heavy.fill([True], [0.5], weight=[2.0])
    with read_back(written(heavy=heavy)) as back:
        assert back["heavy"].members["TNamed"]["fBits"] & USE_WEIGHTS
        assert back["heavy"].passed.variances().tolist() == [4.0]


def test_an_efficiency_keeps_the_priors_it_was_given_bin_by_bin():
    with open_root(str(DATA / "tconfidence-level.root")) as handle:
        eff = handle["eff"]
    members = copy.copy(eff.members)
    members["fFunctions"] = []  # the fit ROOT attached is not a thing this writes
    bare = Efficiency("TEfficiency", members)
    with read_back(written(eff=bare)) as back:
        again = back["eff"]
    assert plain(again.members["fBeta_bin_params"]) == plain(eff.members["fBeta_bin_params"])
    assert again.members["TNamed"]["fBits"] == eff.members["TNamed"]["fBits"]
    assert again.values(method="bayesian").tolist() == eff.values(method="bayesian").tolist()


def test_a_histogram_from_an_older_root_gains_the_member_the_newer_layout_has():
    with open_root(str(DATA / "gauss-h1.root")) as handle:
        old = handle["h1d"]
    assert "fStatOverflows" not in old.members["TH1"]
    with read_back(written(h=old)) as back:
        assert back["h"].members["TH1"]["fStatOverflows"] == 2
        assert back["h"].values(flow=True).tolist() == old.values(flow=True).tolist()
