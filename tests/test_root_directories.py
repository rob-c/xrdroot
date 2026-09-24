"""Directories: made by name or by path, laid out the way ROOT lays them out.

The layout is checked against the one directory tree in ``tests/data`` that
ROOT itself wrote, record by record: the key each directory has in its
parent, the ``TDirectory`` record behind it, and the key list it keeps, with
every place in the file pointing where ROOT's point. What is written is also
read back by path, which is the promise that matters to whoever wrote it.
"""

from __future__ import annotations

import io
import pathlib
import struct

import numpy as np
import pytest

from xrdroot import Histogram, create, open_root
from xrdroot.buffer import Buffer
from xrdroot.file import Key

DATA = pathlib.Path(__file__).parent / "data"


class OneWay(io.RawIOBase):
    """A target that only takes bytes in order, like a pipe or an upload."""

    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:  # type: ignore[override]
        self.data += data
        return len(data)


def key_at(data: bytes, seek: int) -> Key:
    return Key(Buffer(data[seek : seek + 1024]))


def record_of(data: bytes, seek: int) -> dict:
    """A ``TDirectory`` record, every field of it, small or wide."""
    raw = data[seek : seek + 60]
    (version,) = struct.unpack_from(">h", raw)
    form = ">hIIiiqqqH16s" if version > 1000 else ">hIIiiiiiH16s"
    fields = struct.unpack_from(form, raw)
    names = ("version", "created", "modified", "nbytes_keys", "nbytes_name")
    names += ("seek_dir", "seek_parent", "seek_keys", "uuid_version", "uuid")
    record = dict(zip(names, fields))
    record["padding"] = raw[struct.calcsize(form) :]
    return record


def shapes(data: bytes, seek: int, parent: int, path: str = "") -> list[tuple]:
    """Every subdirectory below the record at ``seek``, as what does not move.

    Positions are compared as relations - the record says where its own key
    is, the key list's key says which directory it belongs to - so that two
    files with different contents can be held to the same layout.
    """
    record = record_of(data, seek)
    listed = key_at(data, record["seek_keys"])
    buf = Buffer(data[record["seek_keys"] : record["seek_keys"] + record["nbytes_keys"]])
    Key(buf)
    keys = [Key(buf) for _ in range(buf.i32())]
    found = []
    for key in keys:
        if key.classname != "TDirectory":
            continue
        inner = record_of(data, key.seek_key + key.keylen)
        sub = key_at(data, inner["seek_keys"])
        found.append(
            (
                f"{path}/{key.name}".lstrip("/"),
                key.title == key.name,
                key.objlen,
                key.nbytes - key.keylen,
                key.seek_pdir == parent,
                inner["version"],
                inner["nbytes_name"] == key.keylen,
                inner["seek_dir"] == key.seek_key,
                inner["seek_parent"],  # the file's own, however deep: ROOT's choice
                inner["uuid_version"],
                inner["padding"],
                (sub.classname, sub.name, sub.title),
                sub.seek_pdir == key.seek_key,
                sub.objlen == sub.nbytes - sub.keylen,  # stored raw
            )
        )
        found += shapes(data, key.seek_key + key.keylen, key.seek_key, found[-1][0])
    assert listed.seek_pdir in (parent, 100)
    return found


def top(data: bytes) -> tuple[int, int]:
    """Where a small-headed file's top record is, and where its key is."""
    (begin,) = struct.unpack_from(">i", data, 8)
    (nbytes_name,) = struct.unpack_from(">i", data, 28)
    return begin + nbytes_name, begin


def test_directories_are_laid_out_exactly_as_root_lays_out_its_own():
    root = (DATA / "dirs-6.14.00.root").read_bytes()
    buf = io.BytesIO()
    with create(buf) as out:
        out.mkdir("dir1/dir11")["h1"] = Histogram.new("h1", [0, 1], [1])
        out.mkdir("dir2")
        out.mkdir("dir3")
    ours = buf.getvalue()
    assert shapes(ours, *top(ours)) == shapes(root, *top(root))
    assert [shape[0] for shape in shapes(ours, *top(ours))] == [
        "dir1",
        "dir1/dir11",
        "dir2",
        "dir3",
    ]


def test_a_directory_record_goes_out_before_what_is_put_in_it():
    buf = io.BytesIO()
    with create(buf) as out:
        run = out.mkdir("run")
        run["h"] = Histogram.new("h", [0, 1], [1])
    with open_root(io.BytesIO(buf.getvalue())) as back:
        directory = back._key("run")
        inside = back["run"]._key("h")
    assert directory.seek_key < inside.seek_key
    assert inside.seek_pdir == directory.seek_key


def test_a_path_makes_every_directory_it_passes_through():
    buf = io.BytesIO()
    with create(buf) as out:
        out["a/b/c/h"] = Histogram.new("h", [0, 1, 2], [3, 4])
        out["a/b/note"] = "beside c"
        out["a/top"] = "in a"
    with open_root(io.BytesIO(buf.getvalue())) as back:
        assert back.keys() == ["a"]
        assert back["a"].keys() == ["b", "top"]
        assert back["a/b"].keys() == ["c", "note"]
        assert back["a/b/c/h"].values().tolist() == [3, 4]
        assert back["a/b/note"] == "beside c"
        assert back["a"]["top"] == "in a"
        assert back.classnames() == {"a": "TDirectory"}


def test_a_tree_goes_into_a_directory_and_its_baskets_say_so():
    buf = io.BytesIO()
    with create(buf) as out:
        tree = out.tree("runs/4711/events", {"x": float}, basket_size=64)
        tree.extend({"x": np.arange(40.0)})
        out.mkdir("runs")["frame"] = {"n": np.arange(3, dtype="i")}
    data = buf.getvalue()
    with open_root(io.BytesIO(data)) as back:
        events = back["runs/4711/events"]
        assert events["x"].array().tolist() == list(np.arange(40.0))
        assert back["runs/frame"]["n"].array().tolist() == [0, 1, 2]
        where = back["runs"]._key("4711").seek_key
        seeks = events["x"].record.basket_seek
    assert len(seeks) > 1
    assert all(key_at(data, seek).seek_pdir == where for seek in seeks)


def test_mkdir_gives_back_what_is_already_there_and_makes_nothing_twice():
    buf = io.BytesIO()
    with create(buf) as out:
        first = out.mkdir("a/b")
        assert out.mkdir("a/b") is first
        assert out.mkdir("a").mkdir("b") is first
        assert first.mkdir("c").path == "a/b/c"
        assert (first.name, first.title) == ("b", "b")
    with open_root(io.BytesIO(buf.getvalue())) as back:
        assert back.keys() == ["a"]
        assert back["a"].keys() == ["b"]
        assert back["a/b"].keys() == ["c"]


def test_a_name_written_twice_in_a_directory_becomes_its_second_cycle_there():
    buf = io.BytesIO()
    with create(buf) as out:
        out["d/s"] = "first"
        out["d/s"] = "second"
        out["s"] = "the top's own"
    with open_root(io.BytesIO(buf.getvalue())) as back:
        assert back["d/s"] == "second"
        assert back["d/s;1"] == "first"
        assert back["s;1"] == "the top's own"


def test_a_directory_and_an_object_cannot_share_a_name():
    with create(io.BytesIO()) as out:
        out["h"] = "an object"
        with pytest.raises(ValueError, match="already holds an object"):
            out.mkdir("h")
        with pytest.raises(ValueError, match="already holds an object"):
            out["h/inside"] = "x"
        out.mkdir("d")
        with pytest.raises(ValueError, match="'d' is a directory in the top of the file"):
            out["d"] = "x"
        with pytest.raises(ValueError, match=r"'e' is a directory in 'd'.*write into it"):
            out.mkdir("d/e")
            out.tree("d/e", {"x": float})


def test_a_path_a_reader_could_never_ask_for_makes_no_directory_on_the_way():
    buf = io.BytesIO()
    with create(buf) as out:
        with pytest.raises(ValueError, match="an old cycle"):
            out["a/b;1"] = "x"
        with pytest.raises(ValueError, match="could never be asked for"):
            out.mkdir("a//b")
        with pytest.raises(ValueError, match="could never be asked for"):
            out["/a"] = "x"
        with pytest.raises(ValueError, match="too long for a key"):
            out.mkdir("a/" + "n" * 255)
        with pytest.raises(ValueError, match="the name must be a str"):
            out.mkdir(7)
    with open_root(io.BytesIO(buf.getvalue())) as back:
        assert back.keys() == []


def test_a_directory_says_where_it_is_and_refuses_writes_once_the_file_is_closed():
    out = create(io.BytesIO())
    run = out.mkdir("runs/4711")
    assert repr(run) == "<WritableDirectory 'runs/4711' in '<file>', 0 keys so far>"
    run["s"] = "x"
    assert repr(run) == "<WritableDirectory 'runs/4711' in '<file>', 1 keys so far>"
    assert repr(out) == "<WritableFile '<file>', 1 keys so far>"
    out.close()
    assert run.closed
    assert repr(run) == "<WritableDirectory 'runs/4711' in '<file>', closed>"
    with pytest.raises(ValueError, match="this file is closed"):
        run["t"] = "y"
    with pytest.raises(ValueError, match="this file is closed"):
        out.mkdir("runs")


def test_directories_go_to_a_target_that_cannot_seek_in_one_piece():
    target = OneWay()
    with create(target) as out:
        out["a/b/s"] = "through a pipe"
        assert not target.data
    with open_root(io.BytesIO(bytes(target.data))) as back:
        assert back["a/b/s"] == "through a pipe"


def test_a_with_block_that_raises_takes_back_the_directories_too():
    buf = io.BytesIO()
    with pytest.raises(RuntimeError):
        with create(buf) as out:
            out["a/b/s"] = "x"
            raise RuntimeError("gone wrong")
    assert buf.getvalue() == b""
