"""ROOT's arrays as a member of a class a program declares, written by ``wdobjects``.

``roottest/root/io/bigevent``'s ``Event`` holds a ``TArrayF`` by value and a ``TArrayI *``
behind a pointer, and its ``Track`` a ``TArrayI`` by value - the members that stopped the file
from being written at all. ROOT streams an array through a ``Streamer`` of its own, so a member
of one is its count word and then its values: no version in front of them, since a version
word is what ROOT writes for a class past its first and every ``TArray`` is at 1, and no
marker byte before them either, which is how the ``TStreamerBasicPointer`` a file *describes*
``fArray`` with would be read. The array bytes below are the ones ROOT 6.40.04 itself wrote
for the same arrays, taken from the oracle:

  root -l -b -q -e 'TFile f("t.root","RECREATE"); TArrayF a(3); a[0]=0.5; a[1]=1.5;
    a[2]=2.5; f.WriteObject(&a,"a"); f.Close();'

Every expectation here is the whole record of the class rather than a member's slice of it, so
a member's position is checked as much as its bytes: the record is a byte count and a version,
then ``fArrayF``, ``fArrayI`` and ``fArrayD`` in the order the declaration gives them, and a
member left unset is the empty ``TArray``'s count of nothing or the zero of a null pointer.
"""

from __future__ import annotations

from typing import Any

import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import catalogue_of, find_class
from xrdroot.pyroot.vectors import TArrayD, TArrayF, TArrayI
from xrdroot.wdobjects import DeclaredObject, Instance, write_value
from xrdroot.writer import WBuffer

#: A class of the three arrays ``Event`` and ``Track`` hold: two by value and one behind a
#: pointer, which is how ROOT describes them - an object any and an object-any pointer.
ARRAYS = r"""
class ArrayHolder {
public:
   TArrayF fArrayF;
   TArrayI *fArrayI;
   TArrayD fArrayD;
   ClassDef(ArrayHolder, 1)
};
"""


def _hex(text: str) -> str:
    """Spaces out of a hex string, so that a payload can be laid out beside ROOT's own dumps."""
    return text.replace(" ", "")


def _catalogue() -> Any:
    """The catalogue a macro's class reaches: ``TArrayF`` and its siblings described as a
    file ROOT wrote describes them, which is what a reader of ours needs in the file."""
    run_source(ARRAYS, "arrays.C", call=False)
    return catalogue_of(find_class("ArrayHolder"))


def _payload(members: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    """What the class is written as, and the classes its streamer record describes."""
    written, _used, carried = DeclaredObject(
        Instance("ArrayHolder", members), _catalogue()
    ).root_payload(0)
    return written.hex(), tuple(sorted(carried))


def test_a_float_array_is_its_count_and_then_five_bytes_a_value():
    written, _described = _payload({"fArrayF": TArrayF([0.5, 1.5, 2.5])})
    assert written == _hex(
        "40 00 00 1a 00 01"  # the record: its length, its version
        " 00 00 00 03 3f 00 00 00 3f c0 00 00 40 20 00 00"  # fArrayF: three, then the floats
        " 00 00 00 00"  # fArrayI, null
        " 00 00 00 00"  # fArrayD, empty
    )


def test_the_file_says_what_each_array_it_holds_looks_like():
    """A reader meets ``TArrayF`` in the streamer record and nowhere else, so the file must
    carry ROOT's own description of it, checksum and all."""
    _written, described = _payload(
        {"fArrayF": TArrayF([0.5]), "fArrayI": TArrayI([7]), "fArrayD": TArrayD([1.25])}
    )
    assert described == (("ArrayHolder", 1), ("TArrayD", 1), ("TArrayF", 1), ("TArrayI", 1))


def test_an_array_behind_a_pointer_is_the_tag_that_names_it_and_then_its_values():
    """``TArrayI *fArrayI``: the pointer's counted tag names the class, which is what ROOT's
    ``WriteObjectAny`` writes, and the array itself follows as ``fN`` and its values."""
    written, _described = _payload({"fArrayI": TArrayI([7, 9])})
    assert written == _hex(
        "40 00 00 26 00 01"
        " 00 00 00 00"  # fArrayF, empty
        " 40 00 00 18 ff ff ff ff 54 41 72 72 61 79 49 00"  # the pointer: 24 bytes, TArrayI
        " 00 00 00 02 00 00 00 07 00 00 00 09"  # fN and the two values
        " 00 00 00 00"  # fArrayD, empty
    )


def test_a_pointer_that_is_null_is_the_zero_a_null_pointer_always_is():
    written, _described = _payload({"fArrayI": None})
    assert written == _hex("40 00 00 0e 00 01 00 00 00 00 00 00 00 00 00 00 00 00")


def test_an_array_nobody_set_is_the_count_of_nothing():
    """A default-made ``TArrayF`` has no values, and ROOT writes it as the count word alone."""
    written, _described = _payload({"fArrayF": None})
    assert written == _hex("40 00 00 0e 00 01 00 00 00 00 00 00 00 00 00 00 00 00")


def test_a_double_array_is_its_count_and_eight_bytes_a_value():
    written, _described = _payload({"fArrayD": TArrayD([1.25])})
    assert written == _hex(
        "40 00 00 16 00 01"
        " 00 00 00 00 00 00 00 00"  # fArrayF, empty, and fArrayI, null
        " 00 00 00 01 3f f4 00 00 00 00 00 00"  # fArrayD: one, then the double
    )


def test_a_run_of_numbers_stands_where_the_array_itself_may():
    """The layer that builds an instance may hand over the values rather than ROOT's array,
    and they are the same bytes - which is what the count word then counts."""
    made = WBuffer()
    write_value(made, "TArrayF", [0.5, 1.5, 2.5], _catalogue())
    assert bytes(made.data).hex() == _hex("00 00 00 03 3f 00 00 00 3f c0 00 00 40 20 00 00")


def test_a_value_that_keeps_no_array_is_refused_and_says_what_to_hand_over():
    with pytest.raises(UnsupportedFeatureError, match="stands where a ROOT array should be"):
        _payload({"fArrayF": 3.5})
