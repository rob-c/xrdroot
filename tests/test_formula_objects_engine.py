"""The names a tree's whole-object branch offers: members, base classes, getters and ``TBits``.

ROOT writes an object branch two ways. Split, every member is a branch of its own; **whole**
(``split=0``) one branch streams the object and no member is a branch, so ``TTreeFormula`` looks a
name up *inside* the object each entry holds - ``fNtrack``, ``event.fNtrack``,
``fEvtHdr.fEvtNum`` - and loops over a collection's objects as a branch of a collection is looped
over. This kit learns those names from the values of a few entries, because a file's description of
the class lives above this layer.

Everything here is a plain dictionary and a few free functions, so the engine is testable without a
file: one entry's object is a dict, a base class is a nested dict named by the class, and a
collection is a list of dicts. That is what keeps ``xrdroot.formula.objects`` inside the ``hist``
component's own isolation scope.
"""

from __future__ import annotations

import numpy as np

from xrdroot.formula.objects import (
    SAMPLE,
    Whole,
    learned,
    spelled,
)

#: One entry of a class with a member, a base class's member, an array and a histogram-like pair.
EVENT = {
    "TObject": {"fUniqueID": 7},
    "fNtrack": np.int32(12),
    "fEvtHdr": {"fEvtNum": np.int32(3), "fRun": np.int32(1)},
    "fClosestDistance": np.array([1.0, 2.5, 3.5], np.float32),
    "fTracks": [{"fPx": 1.0, "fPy": 2.0}, {"fPx": 3.0, "fPy": 4.0}],
}


def _one() -> Whole:
    """What one whole-object branch called ``event`` offers, learned from two entries."""
    return learned({"event": [EVENT, EVENT]})


def test_a_getter_and_the_member_it_reads_are_two_spellings_of_one_name():
    """ROOT's formulas resolve a method the way they resolve a member, so ``GetNtrack`` and
    ``fNtrack`` are one name - with ``Is`` spelling ``fIs``, the other convention its classes use.
    """
    assert spelled("GetNtrack") == ("GetNtrack", "fNtrack")
    assert spelled("IsValid") == ("IsValid", "fIsValid")
    assert spelled("fNtrack") == ("fNtrack",)  # a member written as it is has no other spelling


def test_a_member_is_offered_bare_and_with_its_object_in_front():
    """ROOT reads ``fNtrack`` and ``event.fNtrack`` as the same member, and a member of a *member*
    object is reached through it, not offered at the object's own level."""
    found = _one()
    assert "fNtrack" in found and "event.fNtrack" in found
    assert found.branch("fNtrack") == "event"
    assert "fEvtHdr.fEvtNum" in found  # a member object is a level of its own
    assert "fEvtNum" not in found
    assert found.member(["GetNtrack"]) == "fNtrack"  # a getter resolves to the member


def test_a_base_class_s_members_are_the_derived_object_s_own():
    """A base's members are written at the derived object's level, so ``fUniqueID`` is a name of the
    object even though the dictionary nests it under the base's name."""
    found = _one()
    assert "fUniqueID" in found
    assert found.member(["fUniqueID"]) == "fUniqueID"


def test_a_member_of_a_collection_loops_and_is_also_a_name_of_its_own():
    """``fTracks.fPx`` is the tracks' momentum, looped over; and because a collection was crossed,
    ``fPx`` alone is that member too, which is how ROOT reads a tree that holds the collection
    whole."""
    found = _one()
    assert "fTracks.fPx" in found and "fPx" in found
    assert found.dims["fTracks.fPx"] == 1  # one value per track, and the member has no width
    assert found.kinds["fTracks.fPx"] == "f"  # a Float_t, as the value read gave it


def test_a_fixed_array_is_a_member_with_dimensions_and_a_whole_number_is_not():
    """``fClosestDistance`` is three numbers in a row, so it has a dimension; ``fNtrack`` is a whole
    number, which ROOT bins a whole number wide."""
    found = _one()
    assert found.dims["fClosestDistance"] == 1
    assert found.whole_number("fNtrack") is True
    assert found.whole_number("fClosestDistance") is False
    assert found.whole_number("fNothingAtAll") is False  # a name it has no kind for


def test_a_method_that_is_arithmetic_over_members_is_offered_only_when_all_of_them_are_here():
    """ROOT's ``TH1::GetMean`` keeps no member for the answer: it is ``fTsumwx / fTsumw``, so the
    name is offered only where the object holds both - an object that is not quite a histogram
    would otherwise answer for a mean it never computed."""
    histogram = {"fTsumw": 4.0, "fTsumwx": 10.0}
    found = learned({"h": [histogram]})
    assert "GetMean" in found and "h.GetMean" in found
    assert found.derived["GetMean"][0] == ("fTsumwx", "fTsumw")
    assert found.branch("GetMean") == "h"  # read from the branch that holds its members
    partial = learned({"h": [{"fTsumw": 4.0}]})  # one member of the two
    assert "GetMean" not in partial


def test_the_mean_of_nothing_filled_is_zero_rather_than_a_division_by_zero():
    """``GetMean``'s arithmetic divides the two sums, and a histogram nothing was filled into has
    no weight to divide by - ROOT answers 0 there, not NaN."""
    from xrdroot.formula.objects import _mean

    assert _mean(np.array([10.0]), np.array([4.0])).tolist() == [2.5]
    assert _mean(np.array([0.0]), np.array([0.0])).tolist() == [0.0]


def test_a_bits_object_is_offered_under_two_names_and_asks_after_one_bit():
    """A ``TBits`` the tree holds whole is *drawn* a set bit at a time, and ROOT's macro selects on
    ``event.fTriggerBits.TestBitNumber(28)``: the member itself is the number of bits set, and the
    method's argument indexes a value per bit position."""
    bits = {"fNbits": np.int32(32), "fNbytes": np.int32(4), "fAllBits": np.zeros(4, np.uint8)}
    found = learned({"event": [{"fTriggerBits": bits}]})
    assert "fTriggerBits" in found and "event.fTriggerBits" in found
    assert "fTriggerBits.TestBitNumber" in found
    assert found.whole_number("fTriggerBits") is True  # ROOT bins a TBits a whole number wide
    assert found.dims["fTriggerBits"] == 1  # one value per object, not per bit


def test_a_name_a_branch_already_has_is_the_branch_s():
    """ROOT reads the branch first: a tree that has both a branch ``fNtrack`` and a whole object
    holding one reads the branch, so the member is not offered over it."""
    found = learned({"event": [EVENT]}, taken=["fNtrack"])
    assert "fNtrack" not in found
    assert "event.fNtrack" in found  # named through its object, it is unambiguous


def test_an_object_of_another_class_offers_its_own_members_too():
    """A collection of pointers may hold objects of several classes, and what ROOT resolves is every
    member any of them holds - so a name only a later object has is still a name."""
    found = learned({"c": [{"fA": 1}, {"fB": 2}]})  # one object of each class, in one branch
    assert "fA" in found and "fB" in found


def test_a_whole_object_branch_prints_what_it_offers_and_iterates_it():
    """The value ROOT's formula sees is a mapping from a name to where it is read; the repr counts
    the members and the objects, and iteration covers the derived methods too."""
    found = _one()
    assert repr(found) == f"<Whole {len(found.paths) + len(found.derived)} members of 1 objects>"
    assert set(found) == set(found.paths) | set(found.derived)
    # every name that holds members is an object: the branch itself and the base class in it
    assert "event" in found.objects() and "event.TObject" in found.objects()


def test_a_name_written_either_way_resolves_to_the_member_that_has_it():
    """A formula mixes the two spellings - ``fH.GetXaxis().GetXmax()`` is ``fH.fXaxis.fXmax`` - so
    every combination of the parts is tried rather than each part mapped on its own."""
    histogram = {"fH": {"fXaxis": {"fXmax": 10.0}}}
    found = learned({"event": [histogram]})
    assert found.member(["fH", "fXaxis", "fXmax"]) == "fH.fXaxis.fXmax"
    assert found.member(["fH", "GetXaxis", "GetXmax"]) == "fH.fXaxis.fXmax"
    assert found.member(["fH", "GetXaxis", "fXmax"]) == "fH.fXaxis.fXmax"
    assert found.member(["fH", "fNothing"]) is None


def test_a_getter_that_names_no_member_is_answered_by_what_the_rest_of_the_name_reads():
    """ROOT calls the method its class declares and reads the rest off what comes back, so
    ``GetHistogram().GetXaxis()`` is the axis of whatever member that method returns. A file has
    members and no method bodies, so the member is the one that reads the rest of the name."""
    histogram = {"total": {"fXaxis": {"fXmax": 10.0}}}
    found = learned({"event": [histogram]})
    assert found.unnamed("GetTotal") is True  # no member spelt ``fTotal`` here
    assert found.unnamed("fXaxis") is False  # not a getter's shape at all
    assert found.member(["GetTotal", "fXaxis", "fXmax"]) == "total.fXaxis.fXmax"


def test_the_sample_size_is_what_a_tree_reads_to_learn_a_class():
    """The names are learned from a few entries of a branch, not all of them: the constant is ROOT's
    answer to how many, and a collection empty in the first entries is learned from a later one."""
    assert SAMPLE == 8
    empty_first = learned({"event": [{"fTracks": []}, {"fTracks": [{"fPx": 1.0}]}]})
    assert "fTracks.fPx" in empty_first
