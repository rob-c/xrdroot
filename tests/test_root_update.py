"""Updating a ROOT file that is already there, including ones ROOT wrote.

Every file in ``tests/data`` that ROOT wrote takes an update here and keeps
everything it had: the same names of the same classes, the same streamer
information and more, and the objects it held reading back as they did. The
promise about failure is checked byte for byte - a ``with`` block that
raises leaves the file exactly as it was, after megabytes of it had gone out.
"""

from __future__ import annotations

import io
import pathlib
import shutil
import struct

import numpy as np
import pytest

from xrdroot import (
    FormatError,
    Histogram,
    UnsupportedFeatureError,
    create,
    open_root,
    update,
    wupdate,
)
from xrdroot import writer as writing
from xrdroot.buffer import Buffer
from xrdroot.file import Key

DATA = pathlib.Path(__file__).parent / "data"
#: The one file here whose top directory is too small for an update: go-hep
#: wrote it with a record of 42 bytes, where ROOT leaves 60.
CRAMPED = "g4-like.root"
ROOT_WRITTEN = sorted(path.name for path in DATA.glob("*.root") if path.name != CRAMPED)


def copied(tmp_path: pathlib.Path, name: str) -> pathlib.Path:
    target = tmp_path / name
    shutil.copy(DATA / name, target)
    return target


def infos(path) -> dict:
    with open_root(str(path)) as f:
        return f._source.streamers()


def header(data: bytes) -> tuple:
    """The header's fields after the version and fBEGIN, small or wide."""
    wide = struct.unpack_from(">i", data, 4)[0] > 1_000_000
    return struct.unpack_from(">qqiiiBiqi" if wide else ">iiiiiBiii", data, 12)


def free_list(data: bytes) -> list[tuple[int, int]]:
    _end, seek_free, nbytes_free, nfree, *_rest = header(data)
    buf = Buffer(data[seek_free : seek_free + nbytes_free])
    Key(buf)
    entries = []
    for _ in range(nfree):
        wide = buf.u16() > 1000
        entries.append((buf.i64(), buf.i64()) if wide else (buf.i32(), buf.i32()))
    return entries


def small_file(**objects) -> bytes:
    buf = io.BytesIO()
    with create(buf) as out:
        for name, obj in objects.items():
            out[name] = obj
    return buf.getvalue()


# -- what an update keeps and adds ------------------------------------------


@pytest.mark.parametrize("name", ROOT_WRITTEN)
def test_every_file_root_wrote_here_takes_an_update_and_keeps_what_it_had(tmp_path, name):
    path = copied(tmp_path, name)
    with open_root(str(path)) as before:
        classes = before.classnames()
    old = infos(path)
    with update(str(path)) as f:
        f["xrdroot/added"] = Histogram.new("added", [0, 1, 2], [5, 6])
        f["note"] = "added by an update"
    with open_root(str(path)) as after:
        assert {name: after.classnames()[name] for name in classes} == classes
        assert after["xrdroot/added"].values().tolist() == [5, 6]
        assert after["note"] == "added by an update"
    new = infos(path)
    assert set(old) <= set(new)
    assert {name: list(members) for name, members in old.items()} == {
        name: list(new[name]) for name in old
    }


def test_an_update_reads_back_old_and_new_side_by_side(tmp_path):
    path = copied(tmp_path, "dirs-6.14.00.root")
    with open_root(str(path)) as before:
        h1 = before["dir1/dir11/h1"].values().tolist()
    with update(str(path)) as f:
        f["dir1/note"] = "into a directory ROOT made"
        f["dir1/dir11/h2"] = Histogram.new("h2", [0, 1], [7])
        f.tree("fresh/events", {"n": "i"}).extend({"n": np.arange(4, dtype="i")})
    with open_root(str(path)) as back:
        assert back.keys() == ["dir1", "dir2", "dir3", "fresh"]
        assert back["dir1"].keys() == ["dir11", "note"]
        assert back["dir1/dir11/h1"].values().tolist() == h1
        assert back["dir1/dir11/h2"].values().tolist() == [7]
        assert back["dir1/note"] == "into a directory ROOT made"
        assert back["fresh/events"]["n"].array().tolist() == [0, 1, 2, 3]
        assert back.version == 61400  # still says which ROOT made it


def test_a_name_already_there_gets_its_next_cycle(tmp_path):
    path = tmp_path / "cycles.root"
    path.write_bytes(small_file(s="first"))
    with update(str(path)) as f:
        f["s"] = "second"
    with update(str(path)) as f:
        f["s"] = "third"
        f["d/s"] = "elsewhere"
    with open_root(str(path)) as back:
        assert [back[f"s;{cycle}"] for cycle in (1, 2, 3)] == ["first", "second", "third"]
        assert back["s"] == "third"
        assert back["d/s"] == "elsewhere"


def test_a_directory_already_there_is_rewritten_only_when_something_goes_in(tmp_path):
    path = copied(tmp_path, "dirs-6.14.00.root")
    original = path.read_bytes()
    with open_root(str(path)) as before:
        dir2, dir3 = before._key("dir2"), before._key("dir3")
    with update(str(path)) as f:
        f["dir3/s"] = "x"
        f.mkdir("dir2")  # asked for, and nothing put in it
    data = path.read_bytes()
    record = slice(dir2.seek_key, dir2.seek_key + dir2.nbytes)
    assert data[record] == original[record]
    record = slice(dir3.seek_key, dir3.seek_key + dir3.nbytes)
    assert data[record] != original[record]


def test_what_an_update_replaced_becomes_free_space_marked_the_way_root_marks_it(tmp_path):
    path = copied(tmp_path, "dirs-6.14.00.root")
    original = path.read_bytes()
    end, seek_free, nbytes_free, _n, _nn, _u, _c, seek_info, nbytes_info = header(original)
    with update(str(path)) as f:
        f["dir2/h"] = Histogram.new("h", [0, 1], [1])  # a class the file does not describe yet
    data = path.read_bytes()
    gaps = free_list(data)
    assert gaps[-1] == (len(data), writing.BIG)
    freed = set()
    for first, last in gaps[:-1]:
        assert struct.unpack_from(">i", data, first) == (-(last - first + 1),)
        freed.update(range(first, last + 1))
    assert set(range(seek_info, seek_info + nbytes_info)) <= freed
    assert set(range(seek_free, seek_free + nbytes_free)) <= freed
    assert end < len(data) == header(data)[0]


def test_updates_one_after_another_join_their_gaps(tmp_path):
    path = tmp_path / "again.root"
    path.write_bytes(small_file(s="x"))
    for step in range(3):
        with update(str(path)) as f:
            f[f"h{step}"] = Histogram.new(f"h{step}", [0, 1], [step])
    data = path.read_bytes()
    gaps = free_list(data)[:-1]
    assert gaps == writing._merged(gaps)
    with open_root(io.BytesIO(data)) as back:
        assert [back[f"h{step}"].values().tolist() for step in range(3)] == [[0], [1], [2]]
        assert back["s"] == "x"
    assert writing._merged([(10, 19), (40, 49), (20, 29), (15, 25)]) == [(10, 29), (40, 49)]


def test_a_gap_too_small_for_a_marker_is_listed_but_left_unmarked(tmp_path):
    path = tmp_path / "tiny.root"
    path.write_bytes(small_file(s="x"))
    original = path.read_bytes()
    with update(str(path)) as f:
        # Three bytes a reuse had left over, as ROOT's own lists can hold:
        # a four-byte marker there would run into whatever follows.
        f._gaps.append((200, 202))
        f["t"] = "y"
    data = path.read_bytes()
    assert (200, 202) in free_list(data)
    assert data[200:204] == original[200:204]


# -- the streamer information -----------------------------------------------


def test_nothing_new_to_describe_leaves_the_streamer_information_where_it_was(tmp_path):
    path = copied(tmp_path, "gauss-h1.root")
    _e, _f, _nf, _n, _nn, _u, _c, seek_info, nbytes_info = header(path.read_bytes())
    with update(str(path)) as f:
        f["again"] = "a string needs no class described"
    assert header(path.read_bytes())[7:] == (seek_info, nbytes_info)


def test_an_update_past_the_line_moves_every_reference_in_the_streamer_information(
    tmp_path, monkeypatch
):
    path = copied(tmp_path, "std-containers-split00.root")
    old = infos(path)
    monkeypatch.setattr(writing, "BIG", 1_000)
    with update(str(path)) as f:
        f["h"] = Histogram.new("h", [0, 1], [1])
    data = path.read_bytes()
    seek_info = header(data)[7]
    assert Key(Buffer(data[seek_info : seek_info + 1024])).version > 1000
    new = infos(path)
    assert {name: list(members) for name, members in old.items()} == {
        name: list(new[name]) for name in old
    }
    assert "TH1D" in new
    with open_root(str(path)) as back:
        assert back["h"].values().tolist() == [1]


def test_a_file_with_no_streamer_information_gets_one_when_it_needs_one(tmp_path):
    path = tmp_path / "bare.root"
    data = bytearray(small_file(s="x"))
    struct.pack_into(">ii", data, 12 + 28, 0, 0)  # say there is none
    path.write_bytes(bytes(data))
    with update(str(path)) as f:
        f["t"] = "still none needed"
    assert header(path.read_bytes())[7:] == (0, 0)
    with update(str(path)) as f:
        f["h"] = Histogram.new("h", [0, 1], [3])
    assert "TH1D" in infos(path)
    with open_root(str(path)) as back:
        assert back["h"].values().tolist() == [3]
        assert (back["s"], back["t"]) == ("x", "still none needed")


def listing(*entries: bytes) -> bytes:
    """A ``TList`` holding ``entries``, each already a whole object slot."""
    buf = writing.WBuffer()
    index = buf.start(5)
    buf.tobject()
    buf.string("")
    buf.i32(len(entries))
    for entry in entries:
        buf.raw(entry)
        buf.u8(0)
    buf.end(index)
    return bytes(buf.data)


def test_the_streamer_list_walk_follows_every_kind_of_slot_root_writes():
    named = writing.WBuffer()
    at = named.tag("TObjString")  # a class the walk steps over by its length
    named.raw(b"\x00" * 6)
    named.end(at)
    rules = writing.WBuffer()
    at = rules.tag("TList")  # a list inside the list, as ROOT keeps its rules in
    rules.raw(listing(bytes(named.data)))
    rules.end(at)
    keylen = 60  # every place in the list counts from the start of its key
    again = writing.WBuffer()
    again.u32(0x40000000 | 6)  # a length, then a class named before
    again.u32(0x80000000 | (keylen + 21 + 4 + 2))  # the first slot's name, past the map offset
    again.raw(b"\x00\x00")
    data = listing(
        bytes(named.data),
        struct.pack(">I", 0),  # nothing at all
        struct.pack(">I", keylen + 21 + 2),  # an object met before: the first one
        bytes(rules.data),
        bytes(again.data),
    )
    infos = wupdate._Infos(data, keylen, "")
    assert infos.known == set()
    places = [place - keylen for place in infos._refs]
    assert len(places) == 2  # the object met before, and the class named before
    expected = bytearray(data)
    for place in places:
        (was,) = struct.unpack_from(">I", data, place)
        struct.pack_into(">I", expected, place, was + 8)
    assert infos.merged(b"", 0, keylen + 8) == bytes(expected)


def test_the_streamer_list_walk_refuses_what_it_could_not_step_over():
    with pytest.raises(FormatError, match="written with no length"):
        wupdate._Infos(listing(struct.pack(">I", 0x80000000 | 30)), 0, "")
    with pytest.raises(FormatError, match="written with no length"):
        wupdate._Infos(listing(struct.pack(">I", 0xFFFFFFFF)), 0, "")
    nowhere = struct.pack(">II", 0x40000000 | 4, 0x80000000 | 999)
    with pytest.raises(FormatError, match="a reference to nowhere"):
        wupdate._Infos(listing(nowhere), 0, "")
    unclassed = struct.pack(">II", 0x40000000 | 4, 7)
    with pytest.raises(FormatError, match="a reference to nowhere"):
        wupdate._Infos(listing(unclassed), 0, "")


# -- compression ------------------------------------------------------------


def test_an_update_carries_on_with_the_files_compression_unless_told_otherwise(tmp_path):
    path = tmp_path / "zipped.root"
    buf = io.BytesIO()
    with create(buf, compression="lzma", level=3) as out:
        out["s"] = "x"
    path.write_bytes(buf.getvalue())
    with update(str(path)) as f:
        f["h"] = Histogram.new("h", [0, 1], [1])
        assert (f._algorithm, f._level) == ("lzma", 3)
    assert header(path.read_bytes())[6] == 203
    with update(str(path), level=7) as f:
        assert (f._algorithm, f._level) == ("lzma", 7)
    assert wupdate._setting(wupdate.AS_FILED, None, 1) == ("zlib", 1)  # ROOT's 0: its default
    assert wupdate._setting(wupdate.AS_FILED, None, 301) == ("zlib", 1)  # ROOT's old zlib


def test_an_update_told_otherwise_says_so_in_the_header(tmp_path):
    path = tmp_path / "zipped.root"
    path.write_bytes(small_file(s="x"))
    with update(str(path), compression="lzma") as f:
        assert (f._algorithm, f._level) == ("lzma", 6)
    assert header(path.read_bytes())[6] == 206
    with update(str(path), compression=None) as f:
        f["raw"] = "y" * 500
    assert header(path.read_bytes())[6] == 0
    with update(str(path)) as f:
        assert f._algorithm is None
    with open_root(str(path)) as back:
        assert (back["raw"], back["s"]) == ("y" * 500, "x")
    with pytest.raises(ValueError, match="compression must be one of"):
        update(str(path), compression="gzip")


# -- failing, and refusing ---------------------------------------------------


def test_a_with_block_that_raises_leaves_the_file_byte_for_byte_as_it_was(tmp_path, monkeypatch):
    monkeypatch.setattr(writing, "WRITE_BEHIND", 1_024)
    path = copied(tmp_path, "dirs-6.14.00.root")
    original = path.read_bytes()
    with pytest.raises(RuntimeError):
        with update(str(path), compression=None) as f:
            f["dir1/note"] = "x"
            tree = f.tree("dir2/events", {"x": "d"}, basket_size=512)
            tree.extend({"x": np.arange(20_000.0)})
            assert len(path.read_bytes()) > len(original) + 100_000  # it had gone out
            raise RuntimeError("the analysis fell over")
    assert path.read_bytes() == original
    handle = io.BytesIO(original)
    with pytest.raises(RuntimeError):
        with update(handle) as f:
            f["s"] = "x"
            raise RuntimeError("likewise, for a handle")
    assert handle.getvalue() == original
    assert not handle.closed


def test_update_takes_an_open_handle_and_leaves_it_open():
    handle = io.BytesIO(small_file(s="x"))
    with update(handle) as f:
        assert f.name == "<file>"
        f["t"] = "y"
    assert not handle.closed
    with open_root(io.BytesIO(handle.getvalue())) as back:
        assert (back["s"], back["t"]) == ("x", "y")


def test_update_writes_over_the_wire_like_anything_else():
    from xrdclient.testing import FakeServer

    with FakeServer(files={"/data/x.root": small_file(s="there already")}) as server:
        with update(str(server.url / "data" / "x.root")) as f:
            f["d/t"] = "added over the wire"
        with open_root(io.BytesIO(server.contents("/data/x.root"))) as back:
            assert back["s"] == "there already"
            assert back["d/t"] == "added over the wire"


def test_update_refuses_what_is_not_a_root_file_and_lets_go_of_it(tmp_path):
    path = tmp_path / "page.html"
    path.write_bytes(b"<html>" + bytes(200))
    with pytest.raises(FormatError, match="not a ROOT file"):
        update(str(path))
    with pytest.raises(FormatError, match="not a ROOT file"):
        update(io.BytesIO(b"<html>" + bytes(200)))


def test_update_refuses_a_file_that_is_not_the_length_its_header_says(tmp_path):
    path = tmp_path / "longer.root"
    path.write_bytes(small_file(s="x") + b"trailing")
    with pytest.raises(FormatError, match="its header says it ends at"):
        update(str(path))
    data = small_file(s="x")
    with pytest.raises(FormatError, match="still being written, or was cut short"):
        update(io.BytesIO(data[:-5]))


def test_update_refuses_a_file_whose_top_directory_has_no_room_for_its_record(tmp_path):
    path = copied(tmp_path, CRAMPED)
    original = path.read_bytes()
    with pytest.raises(UnsupportedFeatureError, match="in 42 bytes, where ROOT since version 4"):
        update(str(path))
    assert path.read_bytes() == original


def test_update_refuses_a_directory_with_no_key_list(tmp_path):
    data = bytearray(small_file(s="x"))
    (nbytes_name,) = struct.unpack_from(">i", data, 28)
    struct.pack_into(">i", data, 100 + nbytes_name + 26, 0)  # the file's own fSeekKeys
    with pytest.raises(FormatError, match="no key list to read"):
        update(io.BytesIO(bytes(data)))


def test_update_refuses_a_target_that_cannot_seek():
    class OneWay(io.RawIOBase):
        def writable(self) -> bool:
            return True

        def write(self, data) -> int:  # type: ignore[override]
            return len(data)

    with pytest.raises(ValueError, match="cannot seek, and an update has to read"):
        update(OneWay())


def test_a_file_with_no_free_list_is_given_one(tmp_path):
    data = bytearray(small_file(s="x"))
    struct.pack_into(">ii", data, 12 + 4, 0, 0)  # fSeekFree and fNbytesFree
    handle = io.BytesIO(bytes(data))
    with update(handle) as f:
        f["t"] = "y"
    written = handle.getvalue()
    (old_keys, _last), tail = free_list(written)  # the replaced key list, and the end
    assert tail == (len(written), writing.BIG)
    (nbytes_name,) = struct.unpack_from(">i", data, 28)
    assert struct.unpack_from(">i", data, 100 + nbytes_name + 26) == (old_keys,)
    with open_root(io.BytesIO(written)) as back:
        assert (back["s"], back["t"]) == ("x", "y")
