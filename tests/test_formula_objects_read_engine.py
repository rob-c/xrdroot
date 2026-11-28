"""Reading a whole-object branch's members: the column, the ``TBits``, and the refusals.

``xrdroot.formula.objects`` learns *which* names a whole-object branch offers from a few
entries; this file is the other half - what each name's **value** is when an expression is
evaluated over a batch of entries. The shapes ROOT's own trees hold are all here as plain
dictionaries, which is what keeps the module inside the ``hist`` component's own isolation
scope: a member is a number, a string, a fixed array, a list of objects or a ``TBits``, and
one entry's object is one dictionary.

Two rules run through the file. A member is a **column** - one value per entry - and a
member of a collection is a row per entry, because a collection was crossed. And a
``TBits`` is not read as one value at all: ROOT's ``TSelectorDraw::ProcessFillObject`` fills
the number of each bit a ``TBits`` holds *set*, one fill per bit, so a name that is one reads
the numbers of its set bits where ``TestBitNumber`` reads a value per bit position.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.formula.objects import Whole, learned

#: One entry of a class holding a number, a ``char[]``, a fixed array, a histogram's two
#: sums, a collection, a two-dimensional list and a ``TBits``.
EVENT = {
    "TObject": {"fUniqueID": np.int32(7)},
    "fNtrack": np.int32(12),
    "fType": np.array([ord(char) for char in "type1"] + [0, 0], np.int8),
    "fClosestDistance": np.array([1.5, 2.5, 3.5], np.float32),
    "fH": {"fTsumw": np.float64(4.0), "fTsumwx": np.float64(10.0)},
    "fTracks": [{"fPx": np.float32(1.0)}, {"fPx": np.float32(3.0)}],
    "fMatrix": [[1.0, 2.0], [3.0, 4.0]],
    "fBits": {
        "fNbits": np.int32(10),
        "fNbytes": np.int32(2),
        "fAllBits": np.array([0b00000101, 0b00000010], np.uint8),
    },
}


def _learned(**extra: object) -> Whole:
    """What one whole-object branch called ``event`` offers, learned from two of its entries."""
    return learned({"event": [EVENT, EVENT]})


def test_a_member_is_read_as_a_column_of_one_value_per_entry():
    """An expression over a batch of entries reads a member once per entry, and the column it
    gets is NumPy's array of them - which is what a histogram of it is filled from."""
    found = _learned()
    column = found.read("fNtrack", [EVENT, EVENT])
    assert column.tolist() == [12, 12]
    assert column.dtype == np.int32
    assert found.kinds["fNtrack"] == "i"


def test_a_char_array_member_is_the_c_string_it_holds():
    """ROOT streams a fixed array of ``char`` as the characters it holds, so a formula sees a
    string and not numbers - ``fType == "type1"``, which is how ROOT's own tree tests select on
    a ``char fType[20]`` member. The characters stop at the first NUL, as ``strlen`` does."""
    found = _learned()
    assert found.read("fType", [EVENT, EVENT]) == ["type1", "type1"]
    assert found.kinds["fType"] == "U"  # a string, which is what the characters became


def test_a_fixed_array_member_is_a_row_per_entry():
    """A member that is an array is a value with dimensions, so its column is one row per entry
    and not one number - the width a bin counts over."""
    found = _learned()
    rows = found.read("fClosestDistance", [EVENT, EVENT])
    assert [row.tolist() for row in rows] == [[1.5, 2.5, 3.5], [1.5, 2.5, 3.5]]
    assert found.dims["fClosestDistance"] == 1


def test_a_member_of_a_collection_is_a_row_per_entry_through_it():
    """``fTracks.fPx`` loops the objects of a collection the way a branch of a collection is
    looped over: one row of the tracks' momenta for each entry."""
    found = _learned()
    assert found.read("fTracks.fPx", [EVENT, EVENT]) == [[1.0, 3.0], [1.0, 3.0]]
    assert found.dims["fTracks.fPx"] == 1  # one value per track, and no width of its own


def test_a_list_of_lists_is_a_row_and_a_list_that_is_empty_is_a_column_of_nothing():
    """A member whose value is a list is a row like an array's, and one whose lists are empty is
    a value of no kind rather than a crossing: ``_objects`` offers a level only when the
    elements really are objects."""
    found = _learned()
    assert found.read("fMatrix", [EVENT, EVENT]) == [
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    ]  # a row an entry, and the row is the two lists
    assert found.dims["fMatrix"] == 1
    assert found.kinds["fMatrix"] == "f"  # the kind of the numbers inside the row
    empty = {"fEmpty": []}
    nothing = learned({"e": [empty]})
    assert nothing.read("fEmpty", [empty, empty]) == [[], []]
    assert nothing.kinds["fEmpty"] == "O"  # no value to name a kind after


def test_a_derived_method_is_arithmetic_over_the_members_it_reads():
    """``GetMean`` is not a member: a histogram keeps the sums of weights and of ``x * w`` it
    accumulated as it was filled, so the mean is their quotient - and zero, not a NaN, for a
    histogram nothing was filled into."""
    found = _learned()
    assert found.read("fH.GetMean", [EVENT, EVENT]).tolist() == [2.5, 2.5]
    assert found.branch("fH.GetMean") == "event"  # read from the same branch as its members
    empty = {"fTsumw": np.float64(0.0), "fTsumwx": np.float64(0.0)}
    assert learned({"h": [empty]}).read("GetMean", [empty]).tolist() == [0.0]


def test_a_derived_method_of_a_collection_s_member_is_offered_bare_too():
    """A collection was crossed to reach the sums, so the method is a name of its own beside the
    loops over it - ``GetMean`` as well as ``fList.GetMean`` - exactly as a member of a
    collection is."""
    entry = {"fList": [{"fTsumw": np.float64(2.0), "fTsumwx": np.float64(5.0)}]}
    found = learned({"e": [entry]})
    assert "GetMean" in found and "fList.GetMean" in found
    assert found.read("GetMean", [entry]) == [[2.5]]


def test_a_derived_method_a_branch_already_offers_is_not_derived_over_it():
    """A name the tree already reads - a branch, or a member an earlier whole branch offered -
    stays that name: ROOT resolves a branch before it looks inside an object."""
    entry = {"fTsumw": np.float64(1.0), "fTsumwx": np.float64(1.0)}
    found = learned({"h": [entry]}, taken=["GetMean"])
    assert "GetMean" not in found
    assert "h.GetMean" in found  # a name no branch of the tree has


def test_a_bits_member_reads_the_numbers_of_its_set_bits():
    """ROOT draws a ``TBits`` rather than reading it as one value: the numbers of the bits it
    holds set, one fill a bit, in the order ``FirstSetBit`` walks them - ``0``, ``2`` and ``9``
    for the bytes ``0b00000101`` and ``0b00000010`` over ten bits."""
    found = _learned()
    assert [numbers.tolist() for numbers in found.read("fBits", [EVENT, EVENT])] == [
        [0, 2, 9],
        [0, 2, 9],
    ]
    assert found.whole_number("fBits")  # ROOT bins a TBits a whole number wide


def test_the_method_asking_one_bit_reads_a_value_per_bit_position():
    """``TestBitNumber(n)`` indexes one bit, so what it is read from is a value per *position*
    rather than per object: ten of them here, ``1`` where the bit is set - the name ROOT's own
    macro selects on."""
    found = _learned()
    assert found.tested(["fBits", "TestBitNumber"]) == "fBits.TestBitNumber"
    rows = found.read("fBits.TestBitNumber", [EVENT, EVENT])
    assert [row.tolist() for row in rows] == [[1, 0, 1, 0, 0, 0, 0, 0, 0, 1]] * 2
    assert found.tested(["fNtrack"]) is None  # a name that asks no bit


def test_a_bits_member_of_a_collection_is_a_row_of_rows():
    """A ``TBits`` reached through a collection is drawn for each object of it, so the column is
    a row per entry and a row per element inside it."""
    entry = {"fList": [{"fBits": EVENT["fBits"]}, {"fBits": EVENT["fBits"]}]}
    found = learned({"c": [entry]})
    rows = found.read("fList.fBits", [entry])
    assert [[numbers.tolist() for numbers in row] for row in rows] == [[[0, 2, 9], [0, 2, 9]]]


def test_entries_that_disagree_about_a_member_are_refused():
    """An expression is read entry for entry, so a batch whose objects do not agree - one holds
    the member and another does not, or one holds a row where another holds a number - is
    refused with what to do instead rather than filled with a guess."""
    found = _learned()
    without = {"fNtrack": np.int32(1)}
    with pytest.raises(UnsupportedFeatureError, match="the entries differ in holding it"):
        found.read("fH.fTsumw", [EVENT, without])
    ragged = {"fH": {"fTsumw": np.array([1.0, 2.0], np.float64)}}
    with pytest.raises(UnsupportedFeatureError, match="holds one value where another holds a row"):
        Whole({"fH.fTsumw": ("event", ("fH", "fTsumw"))}).read("fH.fTsumw", [ragged, EVENT])


def test_a_path_through_a_member_a_value_is_not_is_refused_by_name():
    """A path is walked one member at a time, and a value that is no object ends it: the refusal
    names the member asked for, so a formula's author knows which name the file does not
    answer."""
    found = Whole({"a.b": ("event", ("a", "b"))}, kinds={"a.b": "i"})
    with pytest.raises(UnsupportedFeatureError, match=r"'a\.b' is a member of the objects"):
        found.read("a.b", [{"a": np.int32(5)}])


def test_a_bits_member_one_entry_does_not_hold_is_refused_with_its_own_sentence():
    """A ``TBits`` is drawn from an object, so a value that is no object at all cannot be drawn:
    the sentence says so and says to select the entries that hold it, which is a different
    refusal from an inconsistent batch's."""
    found = Whole({"fBits": ("event", ())}, bits={"fBits": "set"})
    with pytest.raises(UnsupportedFeatureError, match="is a ``TBits`` of the objects"):
        found.read("fBits", [np.int32(3)])


def test_an_entry_that_is_no_object_teaches_nothing_and_offers_no_name():
    """What a branch offers is learned from the objects of a few entries; an entry that holds a
    number where an object was expected offers nothing rather than failing the whole tree."""
    assert list(learned({"e": [3]})) == []
    assert list(learned({"e": []})) == []  # and no entry at all teaches nothing either


def test_a_member_of_a_base_is_found_after_the_object_s_own_members():
    """A name is looked for among the object's own members first and in its base classes after
    them, so a plain member met before a base's dictionary does not end the search - and a base
    that does not hold the name passes it on to the next."""
    entry = {"fNtrack": np.int32(1), "TObject": {"fUniqueID": np.int32(7)}}
    found = learned({"e": [entry]})
    assert "fUniqueID" in found
    assert found.read("fUniqueID", [entry]).tolist() == [7]
    two = {"TAttLine": {"fLineColor": np.int16(1)}, "TObject": {"fUniqueID": np.int32(9)}}
    assert learned({"e": [two]}).read("fUniqueID", [two]).tolist() == [9]


def test_the_object_a_name_leads_into_is_named_even_without_the_last_name():
    """``owner`` answers the deepest object here that a name walks into, whether or not it holds
    the last member - which is what tells a member this reader cannot read from a name that is
    nothing at all, the object a formula walked into being here."""
    found = _learned()
    assert found.owner(["fH", "GetEntries"]) == "fH"  # the object, not the method it lacks
    assert found.owner(["fH", "fTsumw"]) == "fH"  # a leaf is no object: fH.fTsumw is a name
    assert found.owner(["fH"]) == "fH"  # the object itself
    assert found.owner(["nope"]) is None  # nothing leads anywhere


def test_a_name_that_ends_at_the_value_itself_is_read_as_a_column():
    """A path with no steps left is the value: that is how a member whose own value an
    expression names - a whole branch read as it is - becomes a column."""
    found = Whole({"fNtrack": ("event", ())}, kinds={"fNtrack": "i"})
    assert found.read("fNtrack", [np.int32(12), np.int32(13)]).tolist() == [12, 13]


def test_a_getter_naming_no_member_is_answered_by_the_rest_of_the_name():
    """``GetHistogram().GetXaxis()`` is the axis of the member that method returns, and a file
    has members without method bodies - so which member a getter of that kind answers with is
    whatever member the *rest* of the name reads, tried as each object the expression reached.
    An object of this tree's own, one step shorter than the name, answers nothing."""
    entry = {"fH": {"fXaxis": {"fXmax": np.float64(6.0)}}}
    found = learned({"e": [entry]})
    assert found.member(["GetHistogram", "GetXaxis", "GetXmax"]) == "fH.fXaxis.fXmax"
    assert "fH.GetXaxis.GetXmax" in set(found.answered(["GetHistogram", "GetXaxis", "GetXmax"]))
    assert found.member(["GetNothing", "GetXmax"]) is None  # no member reads the rest of it
