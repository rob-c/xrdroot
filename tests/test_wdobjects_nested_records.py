"""A member of a class a program declares that is itself one of ROOT's objects, written whole.

``roottest/root/io/bigevent``'s ``Event`` holds two ``TLorentzVector``s by value - ``fLorentz``
and ``fKine`` - and each of those holds a ``TVector3`` whole, in ``fP``. So a member of a member
is a record inside a record: the ``TVector3``'s own count, version and members, standing where
the ``TLorentzVector``'s ``fP`` goes, and no pointer in front of it. What the writer is handed
for such a member is the members a layer made of the object, which is what lets one record go
inside another; the object itself cannot be, since nothing below the layer that knows the
program's objects may ask a ``TLorentzVector`` for the ``TVector3`` it keeps - and ``GetP()``,
which is the momentum, is not that vector.

The bytes below are ROOT 6.40.04's own, dumped from the oracle:

  root -l -b -q -e 'TLorentzVector a(1,2,3,4); TBufferFile t(TBuffer::kWrite, 1000);
    t.WriteObjectAny(&a, TLorentzVector::Class()); ...print the buffer...'

which is the record a member of a declared class holds of one. The ``TVector3`` inside it is
``40000024`` (36 bytes), its version 3, the ``TObject`` it derives from at version 1, and then
``fX``, ``fY`` and ``fZ``.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.buffer import Buffer
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_of, find_class, root_row
from xrdroot.pyroot.core.vectors import TLorentzVector, TVector3
from xrdroot.rdobjects import reader_for
from xrdroot.wdobjects import Instance, write_instance
from xrdroot.writer import WBuffer

#: ``TLorentzVector(1,2,3,4)`` as ROOT wrote it: the record's length, version 4, the
#: ``TObject`` base, the ``TVector3`` in ``fP`` - a record of its own - and ``fE``.
LORENTZ = (
    "4000 003c 0004"  # the record: 60 bytes, version 4
    " 0001 00000000 00000000"  # TObject: its version, fUniqueID, fBits
    " 4000 0024 0003 0001 00000000 00000000"
    " 3ff0000000000000 4000000000000000 4008000000000000"  # fP: 1, 2, 3
    " 4010000000000000"  # fE
)

#: The same for ``TLorentzVector(5,6,7,8)``, which is the second element of the array below.
LORENTZ_SECOND = (
    "4000 003c 0004 0001 00000000 00000000"
    " 4000 0024 0003 0001 00000000 00000000"
    " 4014000000000000 4018000000000000 401c000000000000"
    " 4020000000000000"
)

#: A class of a program that holds one of them by value and two in an array - the shape
#: ``Event.h`` has, an object as a member and an object as an element of one.
HOLDERS = r"""
class LorentzHolder {
public:
   TLorentzVector fLorentz;
   TLorentzVector fPair[2];
   ClassDef(LorentzHolder, 1)
};
"""


def _hex(text: str) -> str:
    """Spaces out of a hex string, so that a record can be laid out beside ROOT's own dumps."""
    return text.replace(" ", "")


def _catalogue() -> Any:
    """The catalogue the holder's declaration reaches, with ROOT's layouts for what it holds."""
    run_source(HOLDERS, "holder.C", call=False)
    return catalogue_of(find_class("LorentzHolder"))


def _row(x: float, y: float, z: float, energy: float) -> dict[str, Any]:
    """A ``TLorentzVector``'s members as its layout names them: ``fP`` is the members of the
    ``TVector3`` it keeps whole, not the momentum ``GetP()`` answers with."""
    row = root_row(TLorentzVector(x, y, z, energy), "TLorentzVector")
    row["fP"] = root_row(TVector3(x, y, z), "TVector3")
    return row


def _written(members: dict[str, Any]) -> tuple[str, Any]:
    """What the holder's record is written as, and the catalogue that wrote it."""
    catalogue = _catalogue()
    buf = WBuffer()
    write_instance(buf, Instance("LorentzHolder", members), catalogue)
    return bytes(buf.data).hex(), catalogue


def test_a_vector_inside_a_four_vector_is_a_record_of_its_own():
    """The ``TVector3`` a ``TLorentzVector`` holds by value is written where ``fP`` goes, as
    the record the layout describes - its own length and version, then its members."""
    buf = WBuffer()
    write_instance(buf, Instance("TLorentzVector", _row(1, 2, 3, 4)), _catalogue())
    assert bytes(buf.data).hex() == _hex(LORENTZ)


def test_a_member_held_by_value_is_one_record_inside_another():
    written, _catalogue_used = _written({"fLorentz": _row(1, 2, 3, 4), "fPair": []})
    assert written == "40000042" + "0001" + _hex(LORENTZ)


def test_an_array_of_them_is_the_same_record_once_for_each_element():
    """``TLorentzVector fPair[2]`` is two records in turn, with no pointer in front of either
    and no count of its own: the array's length is the declaration's, not the file's."""
    written, _catalogue_used = _written(
        {"fLorentz": _row(1, 2, 3, 4), "fPair": [_row(1, 2, 3, 4), _row(5, 6, 7, 8)]}
    )
    assert written == "400000c2" + "0001" + _hex(LORENTZ) + _hex(LORENTZ) + _hex(
        LORENTZ_SECOND
    )


def test_an_array_of_them_held_empty_is_no_record_at_all():
    """An array the row leaves empty writes nothing where it goes: the declaration already
    said how long it is, so there is no count and no placeholder to write."""
    empty, _catalogue_used = _written({"fLorentz": _row(1, 2, 3, 4), "fPair": []})
    unset, _catalogue_used = _written({"fLorentz": _row(1, 2, 3, 4), "fPair": None})
    assert empty == unset == "40000042" + "0001" + _hex(LORENTZ)


def test_one_object_where_an_array_of_them_goes_is_refused():
    """A mapping in an array member's place would be walked as its keys and written as a
    record each - nonsense bytes - so it is refused, naming the member and what to hand."""
    with pytest.raises(UnsupportedFeatureError, match="fPair is an array of TLorentzVector"):
        _written({"fLorentz": _row(1, 2, 3, 4), "fPair": _row(1, 2, 3, 4)})


def test_the_classes_a_nested_record_reaches_are_the_ones_the_file_must_name():
    """A reader meets ``TVector3`` inside the record and nowhere else, so the write must say
    it reached ROOT's layout for it - and for the ``TLorentzVector`` holding it."""
    _bytes, catalogue = _written({"fLorentz": _row(1, 2, 3, 4), "fPair": []})
    assert sorted(catalogue.used) == ["TLorentzVector", "TVector3"]


def test_a_nested_record_reads_back_as_the_members_it_was_written_from():
    """Writing and reading agree on the shape: the ``TVector3`` comes back as the dict of its
    members, which is what one level of nesting means on both sides."""
    members = {"fLorentz": _row(1, 2, 3, 4), "fPair": [_row(1, 2, 3, 4), _row(5, 6, 7, 8)]}
    written, catalogue = _written(members)
    back = reader_for("LorentzHolder", catalogue)(Buffer(bytes.fromhex(written)))
    assert back["fLorentz"]["fE"] == 4.0
    assert back["fLorentz"]["fP"] == {
        "TObject": {"fUniqueID": 0, "fBits": 0},
        "fX": 1.0,
        "fY": 2.0,
        "fZ": 3.0,
    }
    assert back["fPair"][1]["fP"]["fZ"] == 7.0
