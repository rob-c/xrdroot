"""A member of a class that is one of ROOT's own classes, read back.

``roottest/root/io/bigevent``'s ``Event`` holds a ``TLorentzVector fLorentz`` (line 137) and its
``BigTrack`` a ``TLorentzVector fKine`` (line 247), and a ``TLorentzVector`` is a ``TVector3 fP``
and a ``Double_t fE``. ROOT writes the ``TVector3`` where the member stands and puts nothing in
the bytes saying which class it is - only the description the file carries does - so the reader
hands back a dictionary of members and no class, and a macro's ``fLorentz.Px()`` has nothing to
be read off. What the reader gives back for such a member is therefore those members *and* the
name of their class, which is what the namespace's ``TVector3`` is made from, as the ``TArray``
members of the same class are given back as the two members ``ROOT.TArrayF`` is made from.

The bytes here are ROOT 6.40.04's own, written into a ``TBufferFile`` with ``WriteObjectAny`` by

    TLorentzVector v(1, 2, 3, 4); b.WriteObjectAny(&v, TLorentzVector::Class());

and the same for a ``TVector3(1, 2, 3)``.
"""

from __future__ import annotations

import datetime

from xrdroot.buffer import Buffer
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_for, catalogue_of, find_class
from xrdroot.pyroot.core.vectors import TVector3
from xrdroot.rdobjects import DescribedSource, Members, _class_of, reader_for

#: ``WriteObjectAny``'s byte count, its new-class tag and the name ``TLorentzVector``, which
#: stand in front of the record and are read by :meth:`Buffer.any` rather than by its reader.
WRITTEN_TLV = 23
#: The same for ``TVector3``: its name is nine bytes, so three fewer.
WRITTEN_VECTOR = 21

#: ROOT 6.40.04's record of a ``TLorentzVector``: its byte count and version, the ``TObject``
#: base, the ``TVector3`` ``fP`` - a record of its own, with a base and the three coordinates -
#: and ``fE``.
LORENTZ = bytes.fromhex(
    "40000053ffffffff544c6f72656e747a566563746f72004000003c0004000100000000000000"
    "00400000240003000100000000000000003ff00000000000004000000000000000"
    "4008000000000000"
    "4010000000000000"
)

#: ROOT 6.40.04's record of a ``TVector3`` of (1, 2, 3), which is the same record the ``fP``
#: above holds, with the class's name in front of it.
VECTOR = bytes.fromhex(
    "40000035ffffffff54566563746f7233004000002400030001"
    "00000000000000003ff000000000000040000000000000004008000000000000"
)

#: A macro's own class: a ``TDatime``, which streams itself as one packed word and no record, a
#: pointer to a ``TVector3``, which names its class where it points, and a number.
CLOCK = r"""
class Clock {
public:
   TDatime fWhen;
   TVector3 *fV;
   Int_t fN;
   ClassDef(Clock, 1)
};
"""

#: ``fWhen`` the 2nd of January 2024 at 03:04:05 - ROOT's packed date word, which counts its
#: years from 1995 - ``fV`` a null pointer, and ``fN`` 7.
CLOCK_FILLED = bytes.fromhex("4000000e0001" + "74443105" + "00000000" + "00000007")


def _read(classname: str, payload: bytes) -> dict[str, object]:
    """A record read by the reader a file or a buffer describes ``classname`` with."""
    return reader_for(classname, catalogue_for(classname))(Buffer(payload))


def _declared(classname: str) -> tuple[object, DescribedSource]:
    """The reader a macro's class reads by, and the layouts it asks its catalogue for."""
    run_source(CLOCK, "clock.C", call=False)
    catalogue = catalogue_of(find_class(classname))
    return reader_for(classname, catalogue), DescribedSource(catalogue)


def test_a_tvector3_member_of_a_tlorentzvector_is_read_as_that_class_of_members():
    """``fP`` is a ``TVector3`` written where it stands, and what the bytes give is its members:
    the three coordinates, and the base every ROOT object has."""
    row = _read("TLorentzVector", LORENTZ[WRITTEN_TLV:])
    assert row["fE"] == 4.0
    found = row["fP"]
    assert isinstance(found, Members)
    assert found.classname == "TVector3"
    assert (found["fX"], found["fY"], found["fZ"]) == (1.0, 2.0, 3.0)


def test_the_members_read_are_what_the_namespaces_tvector3_is_made_from():
    """The row names its class and holds the members that class is described with, which is
    what ``ROOT.TVector3`` is made from: nothing the record holds is thrown away, and what a
    macro asks the member for is the number that was written."""
    found = _read("TLorentzVector", LORENTZ[WRITTEN_TLV:])["fP"]
    assert isinstance(found, dict)
    vector = TVector3(found["fX"], found["fY"], found["fZ"])
    assert (vector.Px(), vector.Py(), vector.Pz()) == (1.0, 2.0, 3.0)
    assert vector.Mag() == 3.7416573867739413  # what ROOT's own GetP() gives


def test_a_tvector3_read_as_a_class_of_its_own_holds_no_class_of_its_own():
    """A ``TVector3`` read where a class stands - a key of a file - is the record itself, and
    none of its members is one of ROOT's classes, so nothing in it is named."""
    row = _read("TVector3", VECTOR[WRITTEN_VECTOR:])
    assert row["fX"] == 1.0
    assert not any(isinstance(value, Members) for value in row.values())


def test_a_member_of_a_class_that_streams_itself_is_left_as_what_it_reads_as():
    """A ``TDatime`` is written as one packed word rather than as a record of its members, so
    what comes back for it is the moment it stands for - which is no dictionary to name a class
    of, so it is left alone rather than made into a ``Members``."""
    read, _source = _declared("Clock")
    row = read(Buffer(CLOCK_FILLED))
    assert row["fWhen"] == datetime.datetime(2024, 1, 2, 3, 4, 5)
    assert row["fN"] == 7
    assert not isinstance(row["fWhen"], Members)


def test_a_pointer_to_one_of_root_classes_is_left_as_the_class_it_names_itself():
    """A pointer names the class in front of what it points at, so the reader already hands
    back the class it was written as; a member held by value is the one that needs naming."""
    _read_clock, source = _declared("Clock")
    assert _class_of(source, "Clock") == (("fWhen", "TDatime"),)


def test_a_class_holding_none_of_root_objects_keeps_the_members_it_declares():
    """A class of numbers is read by the reader the members describe, with nothing named: the
    ``TVector3`` of a member is in the layout nowhere but behind a pointer."""
    read, _source = _declared("Clock")
    assert read(Buffer(CLOCK_FILLED))["fN"] == 7
    run_source("class Plain { public: Int_t fN; ClassDef(Plain, 1) };", "plain.C", call=False)
    source = DescribedSource(catalogue_of(find_class("Plain")))
    assert _class_of(source, "Plain") == ()
