"""Files past 2 GB: ROOT's wide layout, reached without writing 2 GB.

Where the wide layout starts is :data:`xrdroot.writer.BIG`, read each time
a record is written, so these lower it to a few kilobytes and get the layout
a file would have past 2 GB from a file of a few kilobytes. That is a legal
ROOT file in its own right - the wide forms say nothing about how big the
numbers in them are - so it reads back as any other does, which is what is
checked here, along with each field being where ROOT puts it.
"""

from __future__ import annotations

import io
import struct

import numpy as np
import pytest

from xrdroot import Graph, Histogram, create, open_root
from xrdroot import writer as writing
from xrdroot.buffer import Buffer
from xrdroot.file import Key
from xrdroot.winfo import WRITER_VERSION

#: Where these tests have the wide layout start instead of at 2 GB.
LOW = 1_500


@pytest.fixture
def low(monkeypatch):
    monkeypatch.setattr(writing, "BIG", LOW)


def walk(data: bytes) -> list[Key]:
    """Every record in a file, in order, by stepping from key to key."""
    (begin,) = struct.unpack_from(">i", data, 8)
    keys, at = [], begin
    while at < len(data):
        key = Key(Buffer(data[at : at + 1024]))
        keys.append(key)
        at += key.nbytes
    return keys


def written() -> bytes:
    buf = io.BytesIO()
    with create(buf, compression=None) as out:
        out["small"] = "before the line"
        out["h"] = Histogram.new("h", [0, 1, 2, 3], [4, 5, 6])
        out["runs/g"] = Graph.new("g", [1, 2], [3, 4])
        out.tree("runs/events", {"x": "d"}, basket_size=256).extend({"x": np.arange(200.0)})
        out["late"] = "after the line"
    return buf.getvalue()


def test_a_file_past_the_line_reads_back_whole(low):
    data = written()
    assert len(data) > 3 * LOW
    with open_root(io.BytesIO(data)) as back:
        assert back["small"] == "before the line"
        assert back["late"] == "after the line"
        assert back["h"].values().tolist() == [4, 5, 6]
        assert back["runs/g"].y.tolist() == [3, 4]
        assert back["runs/events"]["x"].array().tolist() == list(np.arange(200.0))


def test_keys_stay_small_until_the_line_and_are_wide_after_it_as_roots_are(low):
    keys = walk(written())
    small = [key.version for key in keys if key.seek_key <= LOW]
    assert small and set(small) == {4}
    assert {key.version for key in keys if key.seek_key > LOW} == {1004}


def test_a_wide_key_is_eight_bytes_longer_whatever_it_introduces(low):
    wide = [key for key in walk(written()) if key.seek_key > LOW]
    assert {key.classname for key in wide} >= {"TBasket", "TTree", "TList", "TFile", "string"}
    for key in wide:
        small = 26 + sum(1 + len(text) for text in (key.classname, key.name, key.title))
        assert key.keylen - small in (8, 8 + 19)  # a basket keeps 19 more of its own


def test_the_wide_header_puts_every_field_where_root_reads_it(low):
    data = written()
    version, begin = struct.unpack_from(">ii", data, 4)
    assert (version, begin) == (WRITER_VERSION + 1_000_000, 100)
    end, seek_free, nbytes_free, nfree, _name, units, codes, seek_info, nbytes_info = (
        struct.unpack_from(">qqiiiBiqi", data, 12)
    )
    assert (end, nfree, units, codes) == (len(data), 1, 8, 0)
    assert seek_free + nbytes_free == end
    info = Key(Buffer(data[seek_info : seek_info + 1024]))
    assert (info.name, info.nbytes, info.version) == ("StreamerInfo", nbytes_info, 1004)
    assert struct.unpack_from(">H", data, 57)[0] == 1  # the UUID, after the wide fields


def test_the_free_list_says_where_the_file_ends_in_the_wide_form(low):
    data = written()
    end, seek_free = struct.unpack_from(">qq", data, 12)
    key = Key(Buffer(data[seek_free:]))
    body = data[seek_free + key.keylen :]
    assert struct.unpack(">hqq", body) == (1001, end, LOW + writing.GROW)


def test_directory_records_go_wide_when_what_they_point_at_is_past_the_line(low):
    data = written()
    runs = next(key for key in walk(data) if key.classname == "TDirectory")
    assert runs.seek_key <= LOW  # its key went out early, in the small form
    record = data[runs.seek_key + runs.keylen :][:60]
    version, *_times, nbytes_keys, nbytes_name, seek_dir, seek_parent, seek_keys = (
        struct.unpack_from(">hIIiiqqq", record)
    )
    assert version == 1005
    assert (nbytes_name, seek_dir, seek_parent) == (runs.keylen, runs.seek_key, 100)
    listed = Key(Buffer(data[seek_keys : seek_keys + nbytes_keys]))
    assert (listed.classname, listed.name, listed.seek_pdir) == ("TDirectory", "runs", seek_dir)
    assert struct.unpack_from(">H", record, 42)[0] == 1  # the UUID fills the room
    (nbytes_name,) = struct.unpack_from(">i", data, 12 + 24)  # past two wide places
    assert struct.unpack_from(">h", data, 100 + nbytes_name) == (1005,)  # the file's own


def test_the_free_list_ends_where_root_would_have_moved_it_to():
    big = writing.BIG
    assert writing._tail(100) == big
    assert writing._tail(big) == big
    assert writing._tail(big + 1) == big + writing.GROW
    assert writing._tail(big + writing.GROW) == big + writing.GROW
    assert writing._tail(big + writing.GROW + 1) == big + 2 * writing.GROW


def test_a_file_that_ends_just_past_the_line_is_written_wide_to_the_end(monkeypatch):
    buf = io.BytesIO()
    with create(buf) as out:
        out["s"] = "x"
    small = len(buf.getvalue())
    for line in range(small - 12, small + 1):
        monkeypatch.setattr(writing, "BIG", line)
        buf = io.BytesIO()
        with create(buf) as out:
            out["s"] = "x"
        data = buf.getvalue()
        wide = struct.unpack_from(">i", data, 4)[0] > 1_000_000
        assert wide == (len(data) > line)
        with open_root(io.BytesIO(data)) as back:
            assert back["s"] == "x"
