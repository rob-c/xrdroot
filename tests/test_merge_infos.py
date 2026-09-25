"""Another file's description of its classes, made to stand alone and carried over.

Every file ROOT wrote here has its whole streamer list walked, each
``TStreamerInfo`` written again with its classes named in full, and put
into a new file - where it reads back as the same classes, and where a file
being added to takes on the ones it did not already describe. What cannot
be made to stand alone - an entry that refers to an object met before - is
left out rather than guessed at.
"""

from __future__ import annotations

import io
import pathlib
import shutil

import pytest

from xrdroot import FormatError, create, open_root, update
from xrdroot.buffer import NEW_CLASS_TAG, Buffer
from xrdroot.file import Source
from xrdroot.merging.infos import _Loose, _Relocator, read_entries
from xrdroot.writer import WBuffer, _info_entries, _key_fields, _keylen

DATA = pathlib.Path(__file__).parent / "data"
ROOT_WRITTEN = sorted(path.name for path in DATA.glob("*.root"))


@pytest.mark.parametrize("name", ROOT_WRITTEN)
def test_every_class_a_file_describes_is_carried_and_reads_back_the_same(name):
    with open_root(str(DATA / name)) as f:
        entries = read_entries(f._source)
        described = f._source.streamers()
    assert {classname for classname, _version in entries} == set(described)
    buf = io.BytesIO()
    with create(buf) as out:
        out._carried.update(entries)
        out["note"] = "carried"
    with open_root(io.BytesIO(buf.getvalue())) as again:
        back = again._source.streamers()
    for classname, members in described.items():
        assert [repr(m) for m in back[classname].values()] == [
            repr(m) for m in members.values()
        ], classname


def test_a_file_being_added_to_takes_on_the_classes_it_did_not_describe(tmp_path):
    target = tmp_path / "update.root"
    shutil.copy(DATA / "gauss-h1.root", target)
    with open_root(str(DATA / "tformula.root")) as f:
        entries = read_entries(f._source)
    with update(str(target)) as out:
        out._carried.update(entries)
        out["note"] = "more"
    with open_root(str(target)) as again:
        assert "TF1Convolution" in again._source.streamers()
        assert "TH1D" in again._source.streamers()


def test_a_file_being_added_to_that_already_describes_everything_carried_is_left_so(tmp_path):
    target = tmp_path / "update.root"
    shutil.copy(DATA / "gauss-h1.root", target)
    with open_root(str(target)) as f:
        entries = read_entries(f._source)
        seek_info = f._source.info
    with update(str(target)) as out:
        out._carried.update(entries)
        out["note"] = "more"
    with open_root(str(target)) as again:
        assert again._source.info == seek_info


def test_a_file_that_describes_nothing_carries_nothing():
    assert read_entries(Source(io.BytesIO(b""), "empty", owned=False)) == {}


def _streamer_file(*slots: bytes) -> Source:
    """A file of nothing but a streamer list holding ``slots``, each with its option."""
    payload = WBuffer()
    index = payload.start(5)
    payload.tobject()
    payload.string("")
    payload.i32(len(slots))
    for slot in slots:
        payload.raw(slot)
        payload.string("")
    payload.end(index)
    keylen = _keylen("TList", "StreamerInfo", "")
    key = WBuffer()
    body = bytes(payload.data)
    _key_fields(key, 100, 0, (keylen + len(body), len(body), keylen), 1, 0)
    for text in ("TList", "StreamerInfo", ""):
        key.string(text)
    record = bytes(key.data) + body
    source = Source(io.BytesIO(bytes(100) + record), "crafted", owned=False)
    source.info = (100, len(record))
    return source


def test_an_entry_that_points_back_at_an_object_is_left_out_with_all_after_it():
    good = _info_entries(["TNamed"])[:-1]  # the entry, without its option byte
    loose = (5).to_bytes(4, "big")  # a reference to the object at place 5
    assert list(read_entries(_streamer_file(good, loose, good))) == [("TNamed", 1)]


def test_a_class_named_by_a_place_where_none_was_named_is_refused():
    buf = WBuffer()
    buf.u32(0x40000008)
    buf.u32(0x80000063)  # a class said to be named at place 0x63, where nothing was
    buf.u32(0)
    with pytest.raises(FormatError, match="reference to nowhere"):
        _Relocator(Buffer(bytes(buf.data))).slot()
    with pytest.raises(_Loose):
        _Relocator(Buffer(NEW_CLASS_TAG.to_bytes(4, "big"))).slot()


def test_an_empty_place_and_a_record_without_a_length_are_written_again_as_they_were():
    assert _Relocator(Buffer(bytes(4))).slot() == bytes(4)
    body = WBuffer()
    body.u16(3)  # a TObjArray record with no byte count in front of it
    body.tobject()
    body.string("")
    body.i32(0)
    body.i32(0)
    slot = WBuffer()
    index = slot.tag("TObjArray")
    slot.raw(bytes(body.data))
    slot.end(index)
    again = _Relocator(Buffer(bytes(slot.data))).slot()
    reread = Buffer(again)
    reread.u32()
    reread.u32()
    assert reread.cstring() == "TObjArray"
    version, end = reread.header()
    assert version == 3 and end is not None
