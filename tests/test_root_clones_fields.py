"""A clones array written field by field: a base's fields each in its place, or a refusal.

``TStreamerInfo::WriteBufferClones`` writes a base class's members each for
every object, so reading such an array takes each base apart into its own
fields - and a base that streams itself has no fields to take, which makes
the array unreadable that way.
"""

from __future__ import annotations

from xrdroot.interp import _fields
from xrdroot.streamers import Member
from xrdroot.wclasses import checksum


class _Source:
    def __init__(self, described):
        self.described = described

    def streamers(self):
        return self.described


def test_a_base_that_streams_itself_has_no_fields_to_read_one_by_one():
    source = _Source({"Held": {"TList": Member("TList", "", 0, "BASE", 0)}})
    assert _fields("Held", source, ()) is None


def test_a_base_of_plain_members_reads_its_fields_into_its_own_dictionary():
    source = _Source({
        "Held": {"Base": Member("Base", "", 0, "BASE", 0), "fX": Member("fX", "", 8, "double", 0)},
        "Base": {"fN": Member("fN", "", 3, "int", 0)},
    })  # fmt: skip
    labels = [label for label, _step in _fields("Held", source, ())]
    assert labels == ["Base", "fX"]


def test_an_array_member_counts_its_length_in_the_checksum():
    element = ("TStreamerBasicType", "fA", "", 28, 24, 3, 1, (3, 0, 0, 0, 0), "double", ())
    plain = ("TStreamerBasicType", "fA", "", 8, 8, 0, 0, (0, 0, 0, 0, 0), "double", ())
    assert checksum("A", (element,)) == (checksum("A", (plain,)) * 3 + 3) & 0xFFFFFFFF
