"""Names no branch answers: the members of an object a tree holds whole.

ROOT's ``TTreeFormula`` reads a name that is no leaf off the objects a tree
keeps: ``tree->Draw("GetNtrack()")`` calls the method on the ``Event`` object
the branch ``event`` holds, ``fH.GetXaxis().GetXmax()`` walks from a member to
a member of that, and ``fTracks.fPx`` loops over the objects of a collection.
A file read here has no compiled class to ask, so a whole object comes back as
a dictionary of its members - ``tests/data/object-branches-6.40.root`` is ROOT's
own, written by ``object_branches.C`` beside it, and what each member should
give is what that macro filled: entry ``i`` has ``tlv`` at
``SetPxPyPzE(i, -i, 2i, 20+i)``, ``h1`` a ``TH1F`` of ``4`` bins filled ``i``
times, and ``lines`` the ``i`` lines the macro made.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from xrdroot import FormulaError, Jagged, UnsupportedFeatureError, compile_formula, open_root
from xrdroot.errors import UnsupportedFeatureError as Refused
from xrdroot.formula import select
from xrdroot.formula.objects import SAMPLE, Whole, learned, spelled
from xrdroot.formula.select import TreeNames
from xrdroot.treedraw import draw, is_integer

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def whole_tree():
    with open_root(str(DATA / "object-branches-6.40.root")) as f:
        yield f["t"]


# -- what a whole branch offers -------------------------------------------------


def test_a_member_of_a_whole_object_is_read_by_the_name_its_class_gave_it(whole_tree):
    got = whole_tree.arrays(["tlv.fE", "tlv.fP.fX", "fE"])
    assert got["tlv.fE"].tolist() == [20, 21, 22, 23, 24]
    assert got["tlv.fP.fX"].tolist() == [0, 1, 2, 3, 4]
    assert got["fE"].tolist() == got["tlv.fE"].tolist()  # bare, as ROOT reads a member


def test_a_member_a_base_class_declares_is_read_at_its_objects_own_level(whole_tree):
    # TH1 keeps fEntries and fNcells, and h1 is a TH1F: ROOT writes a base class's
    # members beside the object's own, and reads them by their bare name.
    got = whole_tree.arrays(["h1.fEntries", "fNcells", "h1.fXaxis.fNbins"])
    assert got["h1.fEntries"].tolist() == [1, 2, 3, 4, 5]
    assert got["fNcells"].tolist() == [6, 6, 6, 6, 6]
    assert got["h1.fXaxis.fNbins"].tolist() == [4] * 5


def test_a_member_that_is_an_array_is_a_row_of_values_an_entry(whole_tree):
    values = whole_tree.arrays(["TArrayF"])["TArrayF"]
    assert isinstance(values, Jagged) and len(values) == 5


def test_a_getter_names_the_member_it_reads(whole_tree):
    got = whole_tree.arrays(["tlv.GetE()", "h1.GetEntries()", "GetEntries()"])
    assert got["tlv.GetE()"].tolist() == [20, 21, 22, 23, 24]
    assert got["h1.GetEntries()"].tolist() == [1, 2, 3, 4, 5]
    assert got["GetEntries()"].tolist() == got["h1.GetEntries()"].tolist()


def test_a_chain_of_calls_names_a_member_of_a_member(whole_tree):
    got = whole_tree.arrays(["h1.GetXaxis().GetXmax()", "h1.GetXaxis().fXmax", "h2.fYaxis.fXmax"])
    assert got["h1.GetXaxis().GetXmax()"].tolist() == [4, 4, 4, 4, 4]
    assert got["h1.GetXaxis().fXmax"].tolist() == [4, 4, 4, 4, 4]
    assert got["h2.fYaxis.fXmax"].tolist() == [2, 2, 2, 2, 2]


def test_a_call_is_a_refusal_when_the_object_has_no_member_behind_it(whole_tree):
    # TH1::GetRMS() is arithmetic over the bins too, and no relation between the
    # members a file holds spells it, so there is no class in a file to call it.
    refused = r"is a method of 'h1'.*name the member the method works from"
    with pytest.raises(Refused, match=refused):
        whole_tree.arrays(["h1.GetRMS()"])


def test_a_name_leading_into_an_object_no_member_was_learnt_of_is_unsupported(whole_tree):
    # ``lines`` is a TClonesArray, which is no whole object: nothing of what its
    # lines hold was learnt, so the name is refused for the object it walks into
    # and never as a name nothing has.
    with pytest.raises(Refused, match="asks for 'fX1' inside 'lines'"):
        whole_tree.arrays(["lines.fX1"])


def test_a_member_that_holds_an_object_is_not_a_value(whole_tree):
    with pytest.raises(Refused, match="names 'fXaxis' of 'h1'"):
        whole_tree.arrays(["h1.fXaxis"])


def test_a_call_nothing_of_the_tree_answers_is_not_a_function(whole_tree):
    with pytest.raises(FormulaError, match="is not a function ROOT's formulas know"):
        whole_tree.arrays(["GetNoSuchThing()"])


def test_a_members_name_is_a_name_of_the_tree_with_its_own_dimensions(whole_tree):
    names = TreeNames(whole_tree)
    assert "tlv.fE" in names and "fE" in names
    assert names["tlv.fE"] == 0
    assert names["TArrayF"] == 1
    assert names["numbers"] == 1  # a branch, still read as its own type
    assert isinstance(names.whole, Whole) and "tlv.fE" in names.whole


def test_a_member_is_a_name_only_when_the_tree_holds_an_object_whole(whole_tree):
    plain = TreeNames(whole_tree)["numbers"]
    assert plain == 1  # a std::vector branch is no object, and offers no members


# -- a column of a member --------------------------------------------------------


def test_a_member_can_be_selected_and_cut_on_like_any_column(whole_tree):
    got = whole_tree.arrays(["tlv.fE", "fEntries"], cut="fEntries > 2")
    assert got["tlv.fE"].tolist() == [22, 23, 24]
    assert got["fEntries"].tolist() == [3, 4, 5]


def test_a_members_entries_can_be_named_as_any_columns_can(whole_tree):
    got = whole_tree.arrays(["fEntries"], entries=np.array([0, 4]))
    assert got["fEntries"].tolist() == [1, 5]


def test_a_member_of_a_string_type_comes_back_as_a_string(whole_tree):
    assert whole_tree.arrays(["fName"])["fName"].tolist() == ["h1"] * 5


# -- drawing one ------------------------------------------------------------------


def test_a_member_is_drawn_as_a_branch_of_that_name_is(whole_tree):
    histogram = draw(whole_tree, "tlv.fE>>hE")
    assert histogram.entries == 5
    assert histogram.mean() == pytest.approx(22.0)


def test_a_method_of_an_object_is_drawn_just_as_its_member_is(whole_tree):
    histogram = draw(whole_tree, "GetEntries()")
    assert histogram.entries == 5
    assert histogram.mean() == pytest.approx(3.0)


def test_a_whole_numbered_member_is_binned_a_whole_number_wide(whole_tree):
    names = TreeNames(whole_tree)
    assert is_integer(compile_formula("fNcells", names), whole_tree, names.whole)
    assert not is_integer(compile_formula("fEntries", names), whole_tree, names.whole)
    # and without being told, which is how a draw of one from the ROOT namespace asks
    assert is_integer(compile_formula("fNcells", names), whole_tree)


# -- the learning itself ----------------------------------------------------------


class _Branch:
    """One branch, as learning reads it: what it is called, and what an entry holds."""

    def __init__(self, typename: str, values: Any, raises: bool = False) -> None:
        self.typename = typename
        self.record = type("Record", (), {"branches": []})()
        self._values = values
        self._raises = raises
        self.read: list[tuple[int, int]] = []

    def array(self, start: int = 0, stop: int | None = None) -> Any:
        self.read.append((start, 99 if stop is None else stop))
        if self._raises:
            raise UnsupportedFeatureError(f"{self.typename} is not read here")
        return self._values


class _Tree:
    """A tree of branches only: what learning asks a source for, and nothing else."""

    def __init__(self, branches: dict[str, Any]) -> None:
        self.branches = branches
        self.friends: dict[str, Any] = {}

    def keys(self) -> list[str]:
        return list(self.branches)

    def __len__(self) -> int:
        return 5

    def __getitem__(self, name: str) -> Any:
        return self.branches[name]

    def __contains__(self, name: object) -> bool:
        return name in self.branches


def test_a_branch_that_is_no_whole_object_is_no_member_of_anything():
    names = TreeNames(_Tree({"x": _Branch("float64", np.arange(5.0))}))
    assert list(names.whole) == []


def test_a_branch_that_cannot_be_read_offers_the_names_of_the_others():
    event = _Branch("dict", [{"fN": np.int32(1)}, {"fN": np.int32(2)}])
    broken = _Branch("dict", [], raises=True)
    names = TreeNames(_Tree({"event": event, "broken": broken}))
    assert "fN" in names.whole
    assert "broken" not in names.whole


def test_an_object_that_holds_a_collection_offers_its_elements_members_both_ways():
    # ROOT reads ``fTracks.fPx`` and, as fPx is the momentum of every track, ``fPx``.
    tracks = [[{"fPx": 1.0}, {"fPx": 2.0}], [{"fPx": 3.0}]]
    whole = learned({"event": [{"fTracks": tracks[0]}, {"fTracks": tracks[1]}]})
    assert whole.paths["fTracks.fPx"] == ("event", ("fTracks", "fPx"))
    assert whole.paths["fPx"] == ("event", ("fTracks", "fPx"))
    assert whole.dims["fPx"] == 1
    assert whole.read("fPx", [{"fTracks": tracks[0]}, {"fTracks": tracks[1]}]) == [
        [1.0, 2.0],
        [3.0],
    ]


def test_a_branch_of_the_tree_wins_over_a_member_of_the_same_name():
    whole = learned({"event": [{"x": 1.0}, {"x": 2.0}]}, taken=["x", "event.x"])
    assert "x" not in whole and "event.x" not in whole


def test_the_first_whole_object_of_the_tree_wins_a_name_two_of_them_hold():
    # Two objects with a member of one name: ROOT searches the branches in order.
    whole = learned({"a": [{"fN": 1.0}], "b": [{"fN": 2.0}]})
    assert whole.paths["fN"] == ("a", ("fN",))


def test_a_getter_and_the_member_it_names_are_one_member():
    whole = learned({"h": [{"fH": {"fXaxis": {"fXmax": 4.0}}}]})
    assert whole.member(("GetXmax",)) is None  # nothing of that name at the top
    assert whole.member(("fH", "GetXaxis", "GetXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("fH", "fXaxis", "fXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("fH", "GetXaxis", "fXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("fH", "GetNope")) is None


def test_what_an_object_holds_is_known_from_the_part_of_a_name_that_leads_into_it():
    whole = learned({"h": [{"fH": {"fXaxis": {"fXmax": 4.0}}}]})
    assert whole.owner(("fH", "GetMean")) == "fH"
    assert whole.owner(("fH", "fXaxis", "fXmax")) == "fH.fXaxis"  # the deepest object
    assert whole.owner(("nope", "GetMean")) is None


def test_a_getter_is_spelt_by_the_member_it_names_and_members_by_themselves():
    assert spelled("GetNtrack") == ("GetNtrack", "fNtrack")
    assert spelled("IsValid") == ("IsValid", "fIsValid")
    assert spelled("fNtrack") == ("fNtrack",)
    assert spelled("size") == ("size",)


def test_a_name_is_read_off_the_object_as_a_number_a_string_or_a_row():
    whole = learned({"e": [{"fI": np.int32(1), "fS": "a", "fR": [1.0, 2.0]}]})
    assert whole.read("fI", [{"fI": np.int32(1)}]).tolist() == [1]
    assert whole.read("fS", [{"fS": "a"}]) == ["a"]
    assert whole.read("fR", [{"fR": [1.0, 2.0]}]) == [[1.0, 2.0]]
    assert whole.read("fR", [{"fR": np.array([1.0, 2.0])}])[0].tolist() == [1.0, 2.0]


def test_a_char_array_member_is_the_c_string_it_holds_and_not_its_characters():
    # ``char fType[20]`` streams as twenty ``int8``s; ROOT's ``TLeafC`` reads it as a
    # string, which is what ``fType == "type1"`` in ROOT's own tree tests selects on.
    entry = {"fType": np.frombuffer(b"type1" + b"\x00" * 15, dtype=np.int8)}
    whole = learned({"event": [entry]})
    assert whole.read("fType", [entry, entry]) == ["type1", "type1"]
    assert whole.dims["fType"] == 0  # one string an entry, not twenty numbers
    assert whole.whole_number("fType") is False


def test_a_full_char_array_is_every_character_of_it_and_a_short_one_all_it_has():
    full = {"fType": np.frombuffer(b"0123456789abcdefghij", dtype=np.int8)}  # no NUL
    short = {"fType": np.frombuffer(b"ab", dtype=np.int8)}
    whole = learned({"event": [full, short]})
    assert whole.read("fType", [full, short]) == ["0123456789abcdefghij", "ab"]


def test_a_row_of_unsigned_bytes_is_numbers_and_not_a_string():
    # ``UChar_t`` is a number in ROOT and only ``char`` is a string, so a row of
    # them stays what it is.
    entry = {"fAllBits": np.frombuffer(bytes(4), dtype=np.uint8)}
    whole = learned({"event": [entry]})
    assert whole.dims["fAllBits"] == 1
    assert whole.read("fAllBits", [entry])[0].tolist() == [0, 0, 0, 0]


def test_a_member_no_entry_holds_and_one_that_is_a_row_in_only_some_are_refused():
    whole = learned({"e": [{"fI": 1.0}]})
    with pytest.raises(Refused, match="the entries differ in holding it"):
        whole.read("fI", [{"fI": 1.0}, {}])
    with pytest.raises(Refused, match="the entries differ in holding it"):
        whole.read("fI", [{"fI": 1.0}, {"fI": [1.0]}])


def test_what_a_member_is_reaches_through_the_bases_an_object_is_made_of():
    # ``TH1`` is a base of the histogram, and its members are read at its level.
    whole = learned({"h": [{"TH1": {"fEntries": 1.0}, "TArrayF": [0.0]}]})
    assert whole.read("fEntries", [{"TH1": {"fEntries": 1.0}}]).tolist() == [1]
    assert whole.whole_number("fEntries") is False


def test_learning_reads_the_first_entries_of_a_whole_object_and_no_more(monkeypatch):
    branch = _Branch("dict", [{"fN": 1.0}] * SAMPLE)
    TreeNames(_Tree({"event": branch}))
    assert branch.read == [(0, SAMPLE)]


def test_an_entry_that_holds_no_object_at_all_offers_no_names_from_it():
    whole = learned({"e": [{"fX": 1.0}, 5.0]})
    assert list(whole) == ["fX", "e.fX"]
    with pytest.raises(Refused, match="the entries differ in holding it"):
        whole.read("fX", [{"fX": 1.0}, {}])


def test_a_whole_object_reads_says_what_it_leads_to_and_how_many():
    whole = learned({"e": [{"fX": 1.0}]})
    assert repr(whole) == "<Whole 2 members of 1 objects>"
    assert whole.branch("fX") == "e"


class _NamedTree(dict):
    """A tree's names as ``compile_formula`` takes them, with its objects' members beside them."""

    def __init__(self, dims: dict[str, int], whole: Whole) -> None:
        super().__init__(dims)
        self.whole = whole


def test_a_calls_arguments_index_the_member_its_getter_names():
    # ``GetClosestDistance(2)`` is ROOT's way of writing ``fClosestDistance[2]``.
    whole = Whole({"fDist": ("e", ("fDist",))}, {"fDist": 1}, {"fDist": "f"})
    formula = compile_formula("GetDist(2)", _NamedTree({"e": 0, "fDist": 1}, whole))
    columns = {"e": [{}, {}], "fDist": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}
    assert formula.evaluate(columns).tolist() == [3.0, 6.0]


def test_a_size_after_a_call_is_the_length_of_the_collection_it_gave():
    whole = Whole({"fList": ("e", ("fList",))}, {"fList": 1}, {"fList": "f"})
    formula = compile_formula("GetList().size()", _NamedTree({"e": 0, "fList": 1}, whole))
    columns = {"e": [{}, {}], "fList": [[1.0, 2.0], [3.0]]}
    assert formula.evaluate(columns).tolist() == [2, 1]


def test_a_member_named_in_a_name_that_no_entry_leads_to_is_a_refusal():
    whole = Whole({"a.b": ("e", ("a", "b"))}, {"a.b": 0}, {"a.b": "f"})
    with pytest.raises(Refused, match="the entries differ in holding it"):
        whole.read("a.b", [{"a": 1.0}])


# -- a method that is arithmetic over members ----------------------------------


def test_a_histograms_mean_is_the_sums_of_weights_it_was_filled_with(whole_tree):
    # TH1 keeps no fMean: TH1::GetMean divides fTsumwx by the fTsumw it accumulated,
    # and ROOT's formula reads the two members a file still holds.
    got = whole_tree.arrays(["h1.GetMean()", "h1.fTsumwx", "h1.fTsumw"])
    assert got["h1.GetMean()"].tolist() == (got["h1.fTsumwx"] / got["h1.fTsumw"]).tolist()
    assert got["h1.GetMean()"].tolist() == [0.0, 0.5, 1.0, 1.5, 1.5]


def test_a_method_is_offered_only_where_the_object_holds_every_member_it_reads():
    # An object that is not quite a histogram answers for no mean: one of the two
    # sums is missing, so there is nothing to divide.
    half = learned({"t": [{"h": {"TH1": {"fTsumwx": 3.0}}}]})
    assert "h.GetMean" not in half and "GetMean" not in half
    both = learned({"t": [{"h": {"TH1": {"fTsumwx": 3.0, "fTsumw": 2.0}}}]})
    assert both.read("h.GetMean", [{"h": {"TH1": {"fTsumwx": 3.0, "fTsumw": 2.0}}}])[0] == 1.5


def test_a_mean_of_a_histogram_nothing_was_filled_into_is_zero():
    # ROOT answers 0 rather than dividing by nothing, and a draw of it fills that bin.
    empty = {"h": {"TH1": {"fTsumwx": 0.0, "fTsumw": 0.0}}}
    whole = learned({"t": [empty]})
    assert whole.read("h.GetMean", [empty]).tolist() == [0.0]
    assert whole.dims["h.GetMean"] == 0
    assert not whole.whole_number("h.GetMean")  # a mean is not binned whole-number wide


def test_a_member_object_is_not_a_member_a_method_computes_from():
    # fTsumwx here is an object, not a number the class summed, so GetMean stays away.
    odd = learned({"t": [{"h": {"TH1": {"fTsumwx": {"fX": 1.0}, "fTsumw": 2.0}}}]})
    assert "h.GetMean" not in odd


def test_a_method_reached_through_a_collection_is_a_bare_name_too():
    # ROOT reads fPx of a whole fTracks as the tracks' momentum, and a method of one
    # of its objects as h.GetMean is read of the object a loop is at.
    tracks = {"fTracks": [{"Track": {"h": {"TH1": {"fTsumwx": 4.0, "fTsumw": 2.0}}}}]}
    whole = learned({"t": [tracks]})
    assert "fTracks.h.GetMean" in whole and "GetMean" in whole
    assert whole.member(["GetMean"]) == "GetMean"
    assert whole.owner(["fTracks", "h", "nothing"]) == "fTracks.h"
    assert whole.branch("fTracks.h.GetMean") == "t"
    assert whole.branch("GetMean") == "t"


def test_a_branch_of_the_tree_wins_over_a_method_of_the_same_name():
    # A name the tree has a branch for is read from the branch, as ROOT reads it.
    tree = {"h": {"TH1": {"fTsumwx": 4.0, "fTsumw": 2.0}}}
    assert "h.GetMean" not in learned({"t": [tree]}, taken=["h.GetMean", "GetMean"])


# -- a getter whose member ROOT's class names otherwise ----------------------------


def _event_holding_a_histogram():
    """An object shaped as ``Event`` is: its histogram is ``fH``, and ``GetHistogram`` gives it."""
    return {"event": [{"fH": {"fXaxis": {"fXmax": 1.0}}, "fEvtHdr": {"fRun": 3}, "fNtrack": 4}]}


def test_a_getter_no_member_is_spelt_after_is_the_member_the_rest_of_the_name_reads():
    # ROOT's Event::GetHistogram() returns fH. A file holds the member and no method
    # body, so which member it gives is read off what the rest of the name needs: only
    # fH holds an fXaxis with an fXmax, and GetHistogram().GetXaxis().GetXmax() is it.
    whole = learned(_event_holding_a_histogram())
    assert whole.member(("GetHistogram", "GetXaxis", "GetXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("GetHistogram", "GetXaxis", "fXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("event", "GetHistogram", "GetXaxis", "GetXmax")) == (
        "event.fH.fXaxis.fXmax"
    )


def test_a_getter_the_rest_of_the_name_reads_off_no_member_is_still_no_name():
    whole = learned(_event_holding_a_histogram())
    assert whole.member(("GetHistogram", "GetNope")) is None  # nothing holds what follows
    assert whole.member(("GetNope", "GetXmax")) is None  # a getter that names no member here
    assert whole.member(("fH", "GetRMS")) is None  # a method of the histogram, not a member
    assert whole.member(("GetHistogram",)) is None  # and nothing says which member alone


def test_a_getter_the_member_it_names_is_never_read_as_another_member():
    # GetXmax names fXmax, so the rest of the name is never asked to choose for it.
    axis = {"fXaxis": {"fXmax": 4.0, "fXmin": 0.0}, "fXmax": 9.0}
    whole = learned({"e": [{"fH": axis}]})
    assert whole.member(("fH", "GetXmax")) == "fH.fXmax"
    assert whole.member(("fH", "GetXaxis", "GetXmax")) == "fH.fXaxis.fXmax"


def test_a_call_a_file_cannot_answer_names_the_member_the_rest_of_it_reads():
    event = _Branch("dict", [{"fH": {"fXaxis": {"fXmax": 1.0}}}] * 3)
    whole = TreeNames(_Tree({"event": event})).whole
    assert whole.member(("GetHistogram", "GetXaxis", "GetXmax")) == "fH.fXaxis.fXmax"
    assert whole.member(("event", "GetHistogram", "GetXaxis", "GetXmax")) == (
        "event.fH.fXaxis.fXmax"
    )


# -- an object ROOT draws rather than reads: a TBits ------------------------------


class _Objects(dict):
    """A tree of one whole-object branch, as a draw asks a source for the objects."""

    def __init__(self, objects: list[dict[str, Any]], branch: str = "event") -> None:
        super().__init__({branch: _Branch("dict", objects)})
        self.objects = objects
        self.friends: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.objects)


def _bits(set_bits: Sequence[int], count: int = 20) -> dict[str, Any]:
    """A ``TBits`` as a file holds one whole: the bytes ``fAllBits`` and the count ``fNbits``."""
    raw = np.zeros((count + 7) // 8, np.uint8)
    for bit in set_bits:
        raw[bit // 8] |= 1 << bit % 8
    return {"fNbits": np.uint32(count), "fNbytes": np.uint32(len(raw)), "fAllBits": raw}


def _triggering() -> _Objects:
    """The same two objects as ``Event``: a trigger mask each, and one track's apiece."""
    return _Objects(
        [
            {"fTriggerBits": _bits([2, 5]), "fTracks": [{"fPx": 1.5, "fTriggerBits": _bits([5])}]},
            {
                "fTriggerBits": _bits([7, 8, 9]),
                "fTracks": [{"fPx": -2.5, "fTriggerBits": _bits([1])}],
            },
        ]
    )


def test_a_tbits_is_drawn_as_the_number_of_each_bit_it_holds_set():
    # TSelectorDraw::ProcessFillObject fills a set bit at a time: one fill a bit.
    histogram = draw(_triggering(), "event.fTriggerBits")
    assert histogram.entries == 5  # the two bits of the first object, the three of the second
    assert histogram.mean() == pytest.approx(6.2)  # 2, 5, 7, 8 and 9


def test_a_tbits_of_a_collection_is_drawn_a_set_bit_at_a_time_too():
    # One mask a track, and a fill for each bit either track holds set: 5, then 1.
    histogram = draw(_triggering(), "fTracks.fTriggerBits")
    assert histogram.entries == 2
    assert histogram.mean() == pytest.approx(3.0)


def test_a_tbits_is_drawn_through_the_getter_that_names_its_member():
    # ROOT's Event::GetTriggerBits() returns fTriggerBits, and a file holds the member.
    histogram = draw(_triggering(), "event.GetTriggerBits()")
    assert histogram.entries == draw(_triggering(), "event.fTriggerBits").entries == 5


def test_a_bit_of_every_element_of_a_collection_is_asked_after_one_by_one():
    # ``fTracks.fTriggerBits.TestBitNumber(5)`` is one value a track, so it selects
    # with ``fPx`` a track at a time, as ROOT's own treeformula/event test draws it.
    tree = _triggering()
    histogram = draw(tree, "fTracks.fPx", "fTracks.fTriggerBits.TestBitNumber(5)")
    assert histogram.entries == 1  # only the first entry's track has bit 5 set
    assert histogram.mean() == pytest.approx(1.5)


def test_a_bit_of_an_object_that_is_no_collection_selects_an_entry_at_a_time():
    tree = _triggering()
    histogram = draw(tree, "event.fTracks.fPx", "event.fTriggerBits.TestBitNumber(7)")
    assert histogram.entries == 1  # the second entry's mask has bit 7 set
    assert histogram.mean() == pytest.approx(-2.5)


def test_a_tbits_is_binned_a_whole_number_wide_as_root_bins_one():
    names = TreeNames(_triggering())
    assert is_integer(compile_formula("event.fTriggerBits", names), _triggering(), names.whole)


def test_a_tbits_an_entry_does_not_hold_is_a_refusal():
    whole = Whole(
        {"fBits": ("e", ("fBits",))},
        {"fBits": 1},
        {"fBits": "i"},
        bits={"fBits": "set"},
    )
    with pytest.raises(Refused, match="one entry does not hold it"):
        whole.read("fBits", [{"fBits": 1.0}, {"fBits": 2.0}])


def test_asking_after_a_bit_of_a_member_that_is_no_tbits_is_no_such_name():
    whole = learned({"e": [{"fN": 1.0}]})
    assert whole.tested(("fN",)) is None
    assert Whole().tested(("fN",)) is None


# -- learning what a tree's objects hold, once ------------------------------------


def test_a_tree_says_what_its_objects_hold_once_however_many_expressions_ask(
    whole_tree, monkeypatch
):
    """Learning a table reads the first entries of every whole branch, so a tree learns
    once: what a class holds does not change between two draws of the same tree, and the
    draw that learned the table reads those same entries again for their values."""
    inside, cleared = [], {}
    monkeypatch.setattr(select, "_LEARNED", cleared)
    original = select._learned

    def counting(source, names):
        inside.append(source)
        return original(source, names)

    monkeypatch.setattr(select, "_learned", counting)
    first = whole_tree.arrays(["tlv.fE"])["tlv.fE"].tolist()
    again = whole_tree.arrays(["tlv.fE", "fEntries"])["tlv.fE"].tolist()
    assert inside == [whole_tree], "one table for the tree, however many expressions"
    assert first == again == [20, 21, 22, 23, 24], "and the same members behind it"


def test_a_tree_another_tree_displaces_is_learned_again_when_it_is_asked_for(
    whole_tree, monkeypatch
):
    """A session has a few files open, not a run's worth, so the tables kept are bounded -
    and a tree whose table went is learned again rather than read against a stranger's."""
    inside = []
    monkeypatch.setattr(select, "_LEARNED", {})
    monkeypatch.setattr(select, "LEARNED_KEPT", 1)
    original = select._learned

    def counting(source, names):
        inside.append(source)
        return original(source, names)

    monkeypatch.setattr(select, "_learned", counting)
    with open_root(str(DATA / "object-branches-6.40.root")) as handle:
        other = handle["t"]  # the same file, a tree of its own: a source of its own
        for tree in (whole_tree, other, whole_tree):
            assert tree.arrays(["tlv.fE"])["tlv.fE"].tolist() == [20, 21, 22, 23, 24]
    assert inside == [whole_tree, other, whole_tree]
