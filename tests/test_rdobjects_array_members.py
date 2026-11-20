"""A member of a declared class that is one of ROOT's array classes, read back.

``roottest/root/io/bigevent``'s ``Event`` holds a ``TArrayF fArrayF`` by value and a
``TArrayI *fArrayI`` by pointer, and its ``Track`` a ``TArrayI fInts``. ROOT describes each of
those classes as two members - the counter ``fN`` of the base ``TArray``, and the values
``fArray`` counted by it - and gives the class no streamer of its own, so a member of one is
written as those two members alone: no record round them, and no class name in front of the one
held by value. The reader reads that as the row of numbers it is, and what a macro holds must be
the class the namespace has, whose ``fArray`` and ``GetArray()`` it reaches - so the reader
gives back the two members the description names, which is what makes that class from them.

The bytes here are ROOT 6.40.04's own, written by a macro that held an ``ArrHolder`` of

    class ArrHolder { public: Int_t fN; TArrayF fArrayF; TArrayI *fArrayI; ClassDef(ArrHolder,1) };

with ``fN = 7``, a ``TArrayF`` of three values and a ``TArrayI`` of two, into a
``TBufferFile`` with ``WriteObjectAny``.
"""

from __future__ import annotations

import numpy as np

from xrdroot.buffer import Buffer
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_of, find_class
from xrdroot.pyroot.vectors import TArrayF, TArrayI
from xrdroot.rdobjects import DescribedSource, _array_members, reader_for

#: ``WriteObjectAny``'s byte count, its new-class tag and the name ``ArrHolder``, which stand in
#: front of the record and are read by :meth:`Buffer.any` rather than by a class's reader.
WRITTEN_ANY = 18

#: ROOT 6.40.04's record: ``40 00 00 32`` is its byte count and ``00 01`` its version, then
#: ``fN`` 7, the ``TArrayF``'s counter 3 and its three values (1.5, 3, 4.5), and the ``TArrayI``
#: pointed at - a byte count of 24, its class's name, its counter 2 and its two values.
FILLED = bytes.fromhex(
    "40000044ffffffff417272486f6c646572004000003200010000000700000003"
    "3fc00000404000004090000040000018ffffffff544172726179490000000002"
    "0000000b00000016"
)

#: The same class with nothing in it: ``fN`` 0, an empty ``TArrayF``, and a null ``fArrayI``,
#: which ROOT writes as a byte count of zero where the pointer stands.
EMPTY = bytes.fromhex("40000020ffffffff417272486f6c646572004000000e000100000000" + "00" * 12)

HOLDER = r"""
class ArrHolder {
public:
   Int_t fN;
   TArrayF fArrayF;
   TArrayI *fArrayI;
   ClassDef(ArrHolder, 1)
};
"""


def _reader(declared: str, classname: str) -> tuple[object, DescribedSource]:
    """The reader a macro's class reads by, and the layouts it asks its catalogue for."""
    run_source(declared, "arrays.C", call=False)
    catalogue = catalogue_of(find_class(classname))
    return reader_for(classname, catalogue), DescribedSource(catalogue)


def test_a_tarrayf_held_by_value_is_read_as_the_counter_and_the_values_of_the_class():
    """``TArrayF fArrayF`` is written as its ``fN`` and its ``fArray`` with no record of its
    own, so what comes back is those two members: the base's counter, and the values."""
    read, _source = _reader(HOLDER, "ArrHolder")
    row = read(Buffer(FILLED[WRITTEN_ANY:]))
    assert row["fN"] == 7
    assert row["fArrayF"]["TArray"] == {"fN": 3}
    assert list(row["fArrayF"]["fArray"]) == [1.5, 3.0, 4.5]
    assert row["fArrayF"]["fArray"].dtype == np.float32  # a Float_t, not a double


def test_a_tarrayi_reached_through_a_pointer_is_read_as_the_class_it_points_at():
    """``TArrayI *fArrayI`` names its class once, in front of the object, and the object is the
    same two members - which the reader gives back under the pointer's member."""
    read, _source = _reader(HOLDER, "ArrHolder")
    row = read(Buffer(FILLED[WRITTEN_ANY:]))
    assert row["fArrayI"]["TArray"] == {"fN": 2}
    assert list(row["fArrayI"]["fArray"]) == [11, 22]


def test_what_is_read_back_is_what_the_namespaces_own_array_is_made_from():
    """The two members are what the class the namespace has is made from: ``ROOT.TArrayF`` and
    ``TArrayI`` take the values and give back what a macro asks them for, so nothing the record
    holds is thrown away on the way."""
    read, _source = _reader(HOLDER, "ArrHolder")
    row = read(Buffer(FILLED[WRITTEN_ANY:]))
    floats, ints = TArrayF(row["fArrayF"]["fArray"]), TArrayI(row["fArrayI"]["fArray"])
    assert floats.GetArray().tolist() == [1.5, 3.0, 4.5]
    assert ints.fArray.tolist() == [11, 22]
    assert ints.GetSize() == row["fArrayI"]["TArray"]["fN"]


def test_a_pointer_that_was_never_made_is_still_a_null_pointer():
    """A member that is a pointer to one of the arrays is null when nothing was put there, and
    a member that is one held by value is empty rather than missing - the rule for every other
    member a record holds."""
    read, _source = _reader(HOLDER, "ArrHolder")
    row = read(Buffer(EMPTY[WRITTEN_ANY:]))
    assert row["fN"] == 0
    assert row["fArrayF"]["TArray"] == {"fN": 0}
    assert len(row["fArrayF"]["fArray"]) == 0
    assert row["fArrayI"] is None


def test_a_class_holding_none_of_the_arrays_keeps_the_members_it_declares():
    """A class of plain numbers is read by the reader the members describe, with no reshaping
    of any of them: the name of an array class appears nowhere in its layout."""
    declared = "class Plain { public: Int_t fN; double fX; ClassDef(Plain, 1) };"
    read, source = _reader(declared, "Plain")
    assert _array_members(source, "Plain") == ()
    assert read(Buffer(bytes.fromhex("4000000e000100000003" + "4008000000000000"))) == {
        "fN": 3,
        "fX": 3.0,
    }


def test_a_base_that_is_one_of_the_arrays_is_read_as_the_base_it_is():
    """``TH1F`` derives from ``TArrayF``, and that base is written as the same two members - but
    a base is not a member of the class holding it, so it is left to the reader of bases, which
    keeps the histogram's bins a plain row of numbers for the rest of the kit. Only a member is
    reshaped, which is what the name of the class an element is *typed* with says."""
    from xrdroot.rdobjects import _source

    run_source("class Bare { public: Int_t fN; ClassDef(Bare, 1) };", "bare.C", call=False)
    source = _source(catalogue_of(find_class("Bare")))
    assert _array_members(source, "TH1") == ("fContour", "fSumw2")
    assert source.streamers()["TH1F"]["TArrayF"].typename == "BASE"  # a class, but no member
