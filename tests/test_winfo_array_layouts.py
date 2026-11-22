"""The arrays ROOT builds its histograms on, as many ROOT files now hold one.

The arrays ROOT builds its histograms on, described as ROOT 6.40.04's own file
describes them. ``roottest/root/io/bigevent``'s ``Event`` holds ``TArrayF
fArrayF`` by value and ``TArrayI *fArrayI``, its ``Track`` a ``TArrayI fInts``:
each class streams itself - ``TArray::Streamer`` writes version 1, ``fN`` and
then that many values - so the member is an *object*, a ``TStreamerObjectAny``
naming the class whose bytes are those.

The layout is deliberately not in :data:`xrdroot.winfo.INFOS`, and these tests
hold it out: ``INFOS`` is what the writer walks for the classes a file
describes, and ``TArrayF`` is a base of every ``TH1F``. A file describes the
arrays only when a tree carries one - a bare ``TObject`` holding a ``TArrayF``
and a ``TArrayI *``, which the oracle wrote through ACLiC, made a file of one
``TStreamerInfo``, the holder's - so the writer carries them, and the donor here
is a ``TTree`` whose branch element is a ``ClassDef`` class holding one of each.
"""

from __future__ import annotations

from xrdroot.winfo import ARRAY_LAYOUTS, INFOS
from xrdroot.writer import _closure

#: The counted ``fArray`` of each array: its title, as ROOT 6.40.04 titles it, the
#: streamer type of the pointer and the width of one element.
COUNTED = {
    "TArrayC": ("[fN] Array of fN chars", 41, 1, "char*"),
    "TArrayS": ("[fN] Array of fN shorts", 42, 2, "short*"),
    "TArrayI": ("[fN] Array of fN 32 bit integers", 43, 4, "int*"),
    "TArrayF": ("[fN] Array of fN floats", 45, 4, "float*"),
    "TArrayD": ("[fN] Array of fN doubles", 48, 8, "double*"),
}

#: Each class's checksum and version, as a ROOT 6.40.04 tree file says them. They are
#: also what ``TClass::GetCheckSum`` answers in memory, because ROOT 6.40.04 spells the
#: count ``int`` in both; a ROOT 6.22-era file spells it ``Int_t`` and answers
#: ``0x03f126b3`` for ``TArray`` alone, which is a layout of its own and not this one.
SUMS = {
    "TArray": (0x007021B2, 1),
    "TArrayC": (0xAE879936, 1),
    "TArrayS": (0x035C9314, 1),
    "TArrayI": (0xD9D571C7, 1),
    "TArrayF": (0x5A0BF6F1, 1),
    "TArrayD": (0x7139EF34, 1),
}


def test_no_array_class_is_in_the_table_of_what_a_file_written_here_describes():
    assert [name for name in SUMS if name in INFOS] == []


def test_a_histogram_written_here_describes_no_array_class():
    # An INFOS entry for TArrayF would put it, and TArray through it, into the
    # streamer record of every file that holds a TH1F, whose base it is.
    for histogram in ("TH1C", "TH1F", "TH1D", "TH2F", "TH3S"):
        described = _closure({histogram: None})
        assert described[0] == histogram
        assert [name for name in described if name.startswith("TArray")] == []


def test_the_abstract_base_is_the_count_and_nothing_else():
    assert ARRAY_LAYOUTS["TArray"] == (SUMS["TArray"][0], SUMS["TArray"][1],
                                        ARRAY_LAYOUTS["TArray"][2])
    _sum, _version, elements = ARRAY_LAYOUTS["TArray"]
    assert [element[:4] for element in elements] == [
        ("TStreamerBasicType", "fN", "Number of array elements", 6)]


def test_every_array_is_version_one_and_counts_its_values_with_its_abstract_base():
    assert sorted(ARRAY_LAYOUTS) == sorted(SUMS)
    for name, (sum_, version) in SUMS.items():
        assert ARRAY_LAYOUTS[name][:2] == (sum_, version), name
    for name, (title, code, width, typename) in COUNTED.items():
        base, counted = ARRAY_LAYOUTS[name][2]
        # A base element carries the base's checksum, which is how a reader finds the
        # layout the count ``fN`` comes from, and the base's version after it.
        assert (base[0], base[1], base[8], base[9]) == (
            "TStreamerBase", "TArray", "BASE", (1,))
        assert base[7][1] == ARRAY_LAYOUTS["TArray"][0]
        # The title, the pointer's streamer type, the width of one element, its type.
        assert (counted[0], counted[1], counted[2], counted[3]) == (
            "TStreamerBasicPointer", "fArray", title, code)
        # The counter is the base's member, so the counted array names the class it is in.
        assert (counted[4], counted[8], counted[9]) == (
            width, typename, (1, "fN", "TArray"))


def test_the_count_is_an_int_of_four_bytes_where_an_older_file_said_int_t():
    (element,) = ARRAY_LAYOUTS["TArray"][2]
    assert element[3] == 6 and element[4] == 4 and element[5:] == (0, 0, (0, 0, 0, 0, 0),
                                                                   "int", ())
