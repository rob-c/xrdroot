"""A record copied under a name of another length: its references moved to its new key.

ROOT points back inside a record by place, counted from the start of its
key, so a record renamed is only readable once every such place has moved
by the difference in key length. These hold the move to reading back the
same, and to refusing whatever it cannot account for.
"""

from __future__ import annotations

import io
import pathlib
import struct

import pytest

from xrdroot import create, open_root
from xrdroot.buffer import CLASS_MASK, NEW_CLASS_TAG
from xrdroot.errors import FormatError, UnsupportedFeatureError
from xrdroot.merging import copy
from xrdroot.merging import relocate as relocating
from xrdroot.merging.records import renamed_record

DATA = pathlib.Path(__file__).parent / "data"


def test_a_record_with_references_reads_back_the_same_under_a_longer_name(tmp_path):
    out = tmp_path / "out.root"
    copy(DATA / "tformula.root", out, {"fconv": "a_much_longer_name", "func1": "f"})
    with open_root(str(out)) as moved, open_root(str(DATA / "tformula.root")) as there:
        assert moved.key("a_much_longer_name").keylen != there.key("fconv").keylen
        assert repr(moved["a_much_longer_name"]) == repr(there["fconv"])  # read again the same
        assert repr(moved["f"].members) == repr(there["func1"].members)


def test_every_reference_moves_by_the_difference_in_key_length():
    with open_root(str(DATA / "tformula.root")) as there:
        key = there.key("fconv")
        payload = key.payload(there._source)
        moved = relocating.relocated(key, there._source, key.keylen + 7)
        walked = relocating._walked(payload, key.keylen, key.classname, there._source)
    assert walked.places
    for at in walked.places:
        (before,) = struct.unpack_from(">I", payload, at - key.keylen)
        (after,) = struct.unpack_from(">I", moved, at - key.keylen)
        assert after == (before & CLASS_MASK) | ((before & ~CLASS_MASK) + 7)


def noting(words: list[int], tail: bytes = b"") -> relocating._Noting:
    return relocating._Noting(struct.pack(f">{len(words)}I", *words) + tail, 100)


def test_an_object_pointed_at_before_it_was_read_makes_the_record_unmovable():
    buf = noting([0x12345])
    buf.any({})
    assert not buf.whole and buf.places == [100]


def test_a_reference_to_the_record_itself_and_a_null_are_left_as_they_are():
    for word in (0, relocating.SELF):
        buf = noting([word])
        buf.any({})
        assert buf.whole and buf.places == []


def test_a_class_named_in_full_and_then_pointed_at_is_noted_where_pointed():
    named = noting([0x40000008, NEW_CLASS_TAG], b"TNamed\0" + b"\0" * 8)
    assert named.cstring_at(104) == "TNamed"
    named.any({"TNamed": lambda buf: buf.take(1)})
    assert named.places == [] and named.begun == {102}
    named.refs[106] = "TNamed"
    named.pos = 100
    named.data = struct.pack(">II", 0x40000004, CLASS_MASK | 106) + b"\0" * 8
    named.any({"TNamed": lambda buf: None})
    assert named.places == [104] and named.whole


def test_a_class_nothing_reads_is_stepped_over_and_makes_the_record_unmovable():
    buf = noting([0x40000008, NEW_CLASS_TAG], b"TOther\0\0")
    buf.any({})
    assert not buf.whole


def test_skipping_ahead_makes_the_record_unmovable():
    buf = noting([0x40000002, 0])
    buf.resume(None)
    buf.resume(100)
    assert buf.whole
    buf.resume(104)
    assert not buf.whole
    again = noting([0x40000006, 0x00010000])
    again.skip_record()
    assert not again.whole


def test_a_class_its_file_does_not_describe_or_that_streams_itself_is_refused(monkeypatch):
    with open_root(str(DATA / "tformula.root")) as there:
        key = there.key("fconv")
        source = there._source
        with pytest.raises(UnsupportedFeatureError, match="is not read here"):
            relocating._walked(b"", 10, "NoSuchClass", source)

        class Failing:
            def value(self, buf, at):
                raise FormatError("not as described")

        monkeypatch.setattr("xrdroot.interp.whole_object", lambda name, source: Failing())
        with pytest.raises(UnsupportedFeatureError, match="streams itself its own way"):
            relocating._walked(b"", 10, key.classname, source)

        class Short:
            def value(self, buf, at):
                return None

        monkeypatch.setattr("xrdroot.interp.whole_object", lambda name, source: Short())
        with pytest.raises(UnsupportedFeatureError, match="in a way its file does not describe"):
            relocating._walked(b"left", 10, key.classname, source)


def test_a_record_that_cannot_be_moved_is_written_afresh_when_its_class_can_be(monkeypatch):
    def refusing(*args):
        raise UnsupportedFeatureError("no")

    monkeypatch.setattr(relocating, "relocated", refusing)
    with open_root(str(DATA / "gauss-h1.root")) as there, create(io.BytesIO()) as out:
        with pytest.raises(UnsupportedFeatureError):
            renamed_record(out, "longer_name", there.key("h1d"), there._source, {})
    target = io.BytesIO()
    copy(DATA / "gauss-h1.root", target, {"h1d": "longer_name"})
    with open_root(target) as f:
        assert f.keys() == ["longer_name"]
