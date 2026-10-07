"""Branches of objects, written as ROOT 6.40 wrote ``object-branches-6.40.root``.

The donor holds one of every kind of object branch this writer writes, made
by ``object_branches.C``; the same five entries are written here, and every
basket's bytes - the numbers, packed or not, the vectors, the collections,
the four-vectors streamed whole and the clones array member by member - and
every field of every branch's record is held to ROOT's.
"""

from __future__ import annotations

import io
import pathlib

import numpy as np
import pytest

from xrdroot import create, open_root
from xrdroot.wbranch import Collection, Split, Vector, Whole
from xrdroot.wclasses import Layout, Member, checksum, harvested
from xrdroot.wobjects import IGNORED, named, stream, stream_clones
from xrdroot.writer import WBuffer

DATA = pathlib.Path(__file__).parent / "data"
DONOR = DATA / "object-branches-6.40.root"
#: GenVector's Cartesian four-vector, by the name the donor writes it under.
LORENTZ = "ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> >"
#: The donor's macro class, declared as ``object_branches.C`` declares it.
PACKED = (
    Member("fD", 8, "double", "a double", 8),
    Member("fF", 5, "float", "a float", 4),
    Member("fI", 3, "int", "an int", 4),
    Member("fP", 9, "Double32_t", "[-10,10,12] a packed double", 8),
    Member("fH", 19, "Float16_t", "[0,0,8] a packed float", 4),
)
ENTRIES = np.arange(5)


def packed_layout() -> Layout:
    made = Layout("Packed", 1, 0, PACKED)
    return made._replace(checksum=checksum("Packed", made.elements()))


def _lorentz(px: float, py: float, pz: float, e: float) -> bytes:
    coordinates = {"fX": px, "fY": py, "fZ": pz}
    members = {"fP": coordinates.__getitem__, "fE": e}
    buf = WBuffer()
    stream(buf, "TLorentzVector", members.__getitem__)
    return bytes(buf.data)


def _lines(i: int) -> bytes:
    lines = [
        {"fX1": k, "fY1": i, "fX2": k + 1, "fY2": i + 0.5, "fLineColor": 1, "fLineStyle": 1,
         "fLineWidth": 1}.__getitem__
        for k in range(i)
    ]  # fmt: skip
    return named("TClonesArray", stream_clones("TLine", lines, bypass=True))


def _columns() -> dict[str, object]:
    """The donor's five entries, as ``object_branches.C`` fills them."""
    packed = (1.25 * ENTRIES - 2).astype(np.float32).astype(np.float64)
    truncated = packed.astype(np.int64).astype(np.float64)  # C++ to int: no negative zero
    lv = [ENTRIES * 1.0, ENTRIES * 2.0, ENTRIES * 3.0, 10.0 + ENTRIES]
    counts = ENTRIES % 3
    columns: dict[str, object] = {
        "numbers": [np.full(i, 0.5 * i, dtype=np.float32) for i in ENTRIES],
        "lvs": counts,
        "tlv": [_lorentz(i, -i, 2 * i, 20 + i) for i in ENTRIES],
        "lines": [_lines(int(i)) for i in ENTRIES],
    }
    for name, values in zip("DFIPH", (truncated,) * 3 + (packed,) * 2, strict=True):
        columns[f"packed\0f{name}"] = values
    for name, values in zip("XYZT", lv, strict=True):
        columns[f"lv\0fCoordinates.f{name}"] = values
        rows = [np.full(c, x) for c, x in zip(counts, values, strict=True)]
        columns[f"lvs\0lvs.fCoordinates.f{name}"] = rows
    return columns


@pytest.fixture(scope="module")
def written():
    lorentz = harvested(LORENTZ)
    specs = {
        "numbers": Vector("f"),
        "packed": Split(packed_layout(), split=1, own=True),
        "lv": Split(lorentz),
        "lvs": Collection(lorentz),
        "tlv": Whole("TLorentzVector", custom=True, split=0),
        "lines": Whole("TClonesArray", object=True, holds=("TLine",), split=0),
    }
    IGNORED.update({"TLorentzVector", "TVector3"})
    buf = io.BytesIO()
    try:
        with create(buf, level=1) as out:
            tree = out.tree("t", specs, title="objects")
            tree.extend(_columns())
    finally:
        IGNORED.difference_update({"TLorentzVector", "TVector3"})
    with open_root(io.BytesIO(buf.getvalue())) as back, open_root(DONOR) as donor:
        yield back["t"], donor["t"], tree


def _branches(tree):
    """Every branch that holds baskets, by name, as the reader lists them."""
    return {name: branch for name, branch in tree.branches.items() if branch.record.basket_seek}


def test_every_basket_holds_the_bytes_root_wrote(written):
    ours, roots, _tree = written
    mine, theirs = _branches(ours), _branches(roots)
    assert set(theirs) - {"h1", "h2"} == set(mine)
    for name, branch in mine.items():
        record = branch.record
        assert len(record.basket_seek) == len(theirs[name].record.basket_seek), name
        for at in range(len(record.basket_seek)):
            assert branch.basket(at).data == theirs[name].basket(at).data, (name, at)


def _fields(record):
    element = list(record.element or ())
    if record.classname.startswith("vector<"):
        element[2] = None  # a std::vector's checksum: its name's here, ROOT 6.40's own there
    return (record.title, record.classname, record.split, record.entries,
            record.entry_offset_len, record.basket_size, record.tot_bytes, element)  # fmt: skip


def test_every_branch_says_what_roots_says_of_itself(written):
    ours, roots, _tree = written
    mine = {record.name: record for top in ours.records for record in top.walk()}
    theirs = {record.name: record for top in roots.records for record in top.walk()}
    assert set(theirs) - {"h1", "h2"} == set(mine)
    for name, record in mine.items():
        assert _fields(record) == _fields(theirs[name]), name


def test_what_is_written_reads_back_as_roots_reads(written):
    ours, roots, _tree = written
    for name in _branches(ours):
        assert repr(ours[name].array()) == repr(roots[name].array()), name


#: What ROOT 6.40's ``TBranch::Streamer`` streams each of the donor's branches to, on its
#: own as ``GetTotalSize`` streams it, measured with ROOT from the donor read back.
STREAMED = {"numbers": 497, "packed": 3123, "fD": 477, "lv": 3553, "fCoordinates": 2928,
            "fCoordinates.fX": 529, "lvs": 3656, "lvs.fCoordinates.fX": 591, "tlv": 481,
            "lines": 488}  # fmt: skip


def test_a_branch_streams_on_its_own_to_the_length_roots_does(written):
    from xrdroot.wbranch import streamed_size

    *_, tree = written
    every = {b.name: b for top in tree._tops for b in _walk(top)}
    for name, length in STREAMED.items():
        assert streamed_size(every[name], 5, held=False) == length, name


def _walk(branch):
    return [branch, *(below for child in branch.children for below in _walk(child))]
