"""A split ``TClonesArray`` of a class the file describes, and the writer does not.

``roottest``'s ``root/io/tree/create-rc`` clones a tree whose ``TClonesArray``
holds a class a macro declared: the class is not one of ROOT's, so no donor
here describes it, and the clone is refused unless the writer takes the
description the program published for it - which is what
:func:`xrdroot.interp.declare` exists for. The member's rows of a whole
``TRef`` also come back out of the source's baskets as the value a file
describes rather than as the bytes a streamer made, so the column writes
that value back as the record again.
"""

from __future__ import annotations

import struct
from typing import Any

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.interp import DECLARED, declare
from xrdroot.wbranch import (
    REFERENCE,
    CloneMember,
    SplitClones,
    WholeRows,
    clones_members,
    tref_record,
)

#: ``tref_test_pid``'s streamer elements as ROOT 6.40 records them for the class the
#: ``roottest`` macro declares, read out of a file the oracle wrote: its ``TNamed`` base,
#: then the reference. The checksum is what :class:`~xrdroot.wclasses.checksum` makes of
#: them, and what the branch ``r1.fRef`` of ROOT's own file names.
ELEMENTS = (
    ("TStreamerBase", "TNamed", "The TNamed base class", 67, 0, 0, 0, (0,) * 5, "BASE", (1,)),
    ("TStreamerObject", "fRef", "a reference", 61, 0, 0, 0, (0,) * 5, "TRef", ()),
)
CHECKSUM = 2532022293


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> tuple[int, int, tuple[Any, ...]]:
    """``tref_test_pid`` as a program that declared it has published its layout."""
    layout = (CHECKSUM, 1, ELEMENTS)
    monkeypatch.setitem(DECLARED, "tref_test_pid", layout)
    return layout


def test_a_split_array_of_a_class_the_program_published_is_split_by_that_layout(published):
    """A class a macro declared is no class of ROOT's, so the cloned file's name for it
    answers from the layout the layer that declared it published - bases' members first,
    each numbered by its place in the class's own elements, as ROOT numbers them."""
    found = clones_members("tref_test_pid")
    assert [(member.owner, member.name) for member in found] == [
        ("TObject", "fUniqueID"),
        ("TObject", "fBits"),
        ("TNamed", "fName"),
        ("TNamed", "fTitle"),
        ("tref_test_pid", "fRef"),
    ]
    reference = found[-1]
    assert (reference.fid, reference.stype, reference.typename) == (1, 61, REFERENCE)
    assert (reference.checksum, reference.version) == (CHECKSUM, 1)


def test_a_split_array_of_a_published_class_carries_its_description_and_names_none(published):
    """The file is told what the class is: ROOT writes ``tref_test_pid``'s
    ``TStreamerInfo`` into it and names no such class among the ones it looked up."""
    split = SplitClones("tref_test_pid", split=99)
    (carried,) = split.declared()
    assert (carried.name, carried.checksum, carried.version) == ("tref_test_pid", CHECKSUM, 1)
    assert carried.elements() == ELEMENTS
    assert "tref_test_pid" not in split.classes()
    assert {"TNamed", "TObject", "TClonesArray"} <= set(split.classes())
    top = split.build("r1", 32000)
    assert top.info.clones == "tref_test_pid"
    assert [child.name for child in top.children] == [
        "r1.fUniqueID",
        "r1.fBits",
        "r1.fName",
        "r1.fTitle",
        "r1.fRef",
    ]
    assert [child.info.fid for child in top.children][-1] == 1


def test_a_class_nothing_describes_is_still_refused(monkeypatch):
    """Neither ROOT nor this program has the class, so the writer says what to do instead."""
    monkeypatch.delitem(DECLARED, "NobodyDeclaredThis", raising=False)
    with pytest.raises(UnsupportedFeatureError, match="carries a description of"):
        clones_members("NobodyDeclaredThis")


def test_a_macro_class_keeps_its_own_numbering_which_starts_after_its_bases():
    """``roottest``'s ``create-r`` writes the same class off the declaration it was made
    from, and ROOT numbers a member's branch by its place in the elements: the base is
    element 0, so the first member of the class's own is 1 and not 0."""
    from xrdroot.wclasses import declared

    class _TrefPid:
        _cxx_layout_ = ("tref_test_pid", ("TNamed",), (("fRef", "TRef", "a reference", ()),))

    layout = declared(_TrefPid)
    assert clones_members(layout)[-1].fid == len(layout.bases)
    assert clones_members("TNamed")[-1].fid == 2  # fTitle, after the TObject base and fName


def test_a_reference_that_came_back_out_of_a_file_is_written_as_its_record_again():
    """A clone reads a whole ``TRef`` member back as the value the file describes -
    ``TObject``'s version, the object it names, its bits and its ``fPID`` - which is the
    record ``TRef::Streamer`` writes, so the column lays it out as one."""
    rows = [
        {"TObject": {"fUniqueID": 2, "fBits": 0x03000000}, "fPID": 0},
        {"TObject": {"fUniqueID": 7, "fBits": 0x03000000}, "fPID": 3},
    ]
    column = WholeRows("r1.fRef", REFERENCE, 32000)
    raw, sizes = column.pack_many([rows])
    assert raw == struct.pack(">HIIH", 1, 2, 0x03000000, 0) + struct.pack(
        ">HIIH", 1, 7, 0x03000000, 3
    )
    assert list(sizes) == [len(raw)]


def test_an_entry_of_no_objects_is_no_bytes_of_that_member():
    """A split array's entry may hold no objects at all, and the column writes nothing."""
    column = WholeRows("r1.fRef", REFERENCE, 32000)
    assert column.pack_many([[]]) == (b"", [0])


def test_bytes_a_streamer_made_are_written_as_they_are():
    """An object in memory was already streamed by its own class, and its entry's rows
    arrive as the bytes a streamer made of them - the ones the column passes on."""
    column = WholeRows("r1.fRef", REFERENCE, 32000)
    assert column.pack_many([b"\x00\x01\x02"]) == (b"\x00\x01\x02", [3])
    assert column.pack(tref_record(_Ref(2, 0x03000000, None))) == struct.pack(
        ">HIIH", 1, 2, 0x03000000, 0
    )


def test_a_described_object_this_writer_cannot_stream_is_refused():
    """Only a reference's described value is one this writer knows how to write back."""
    column = WholeRows("r1.fV", "TLorentzVector", 32000)
    with pytest.raises(ValueError, match="already streamed"):
        column.pack_many([[{"fX": 1.0}]])


def test_the_description_a_split_array_carries_is_the_one_published_for_it(monkeypatch):
    """A layout published later is the one a later write is split by, as a dictionary
    re-declaring a class replaces what it said of the class before."""
    split = SplitClones("tref_test_pid", split=2)
    changed = (987654321, 3, ELEMENTS)
    declare("tref_test_pid", changed)
    try:
        (carried,) = split.declared()
        assert (carried.checksum, carried.version) == (987654321, 3)
        assert clones_members("tref_test_pid")[-1].checksum == 987654321
    finally:
        DECLARED.pop("tref_test_pid", None)


class _Ref:
    """A reference as the writer is handed one, which is only asked for its record."""

    def __init__(self, number: int, bits: int, pid: Any) -> None:
        self.number, self.bits, self._pid = number, bits, pid

    def GetUniqueID(self) -> int:
        return self.number

    def _bitword(self) -> int:
        return self.bits


def test_a_member_that_is_a_number_is_a_member_column_and_not_a_whole_one(published):
    """Every member of a published class is laid out by what it is, not all the same way."""
    found = clones_members("tref_test_pid")
    assert isinstance(found[0], CloneMember) and found[0].typename == ""
