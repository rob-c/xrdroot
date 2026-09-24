"""Friends: other trees of the same entries, read beside a tree as though part of it.

``join1.root`` to ``join4.root`` are ROOT's own, from go-hep: three trees of
ten entries each whose values say which tree and which entry they are
(``b10`` of ``j1`` is ``101`` to ``110``), and a fourth file of two trees,
one of eleven entries and one sharing column names with ``j1`` and ``j2``.

None of them records a friend: ROOT only writes a ``TFriendElement`` into a
tree that was given one, and no file in the corpus was. So the recorded ones
are made here by writing a tree and then writing its record again with a
friend list in it, laid out the way ``TFriendElement`` streams itself.
"""

from __future__ import annotations

import pathlib
import shutil
import struct

import numpy as np
import pytest

from crafted import Out
from xrdroot import create, open_root
from xrdroot.buffer import Buffer
from xrdroot.friends import friend_target
from xrdroot.objects import FriendRecord, _tree_friends

DATA = pathlib.Path(__file__).parent / "data"


def opened(name: str):
    return open_root(str(DATA / f"{name}.root"))


def test_a_friend_s_column_is_asked_for_by_its_alias_and_the_branch():
    with opened("join1") as one, opened("join2") as two:
        tree = one["j1"]
        tree.add_friend(two["j2"], "second")
        assert tree["second.b20"].array(0, 3).tolist() == [201.0, 202.0, 203.0]
        assert tree["b10"].array(0, 1).tolist() == [101.0]  # its own columns are as they were
        assert list(tree.friends) == ["second"]


def test_a_friend_s_column_is_there_by_its_bare_name_when_only_it_has_one():
    with opened("join1") as one, opened("join2") as two, opened("join3") as three:
        tree = one["j1"]
        tree.add_friend(two["j2"])
        tree.add_friend(three["j3"])
        assert tree["b32"].array(8, 10) == ["j3-309", "j3-310"]
        assert "b21" in tree and "j2.b21" in tree and "nothing" not in tree
        batch = tree.arrays(["b10", "j2.b21", "b32"], 2, 4)
        assert batch["j2.b21"].tolist() == [203, 204]
        assert batch["b32"] == ["j3-303", "j3-304"]


def test_a_name_two_friends_share_has_to_be_asked_for_by_alias():
    with opened("join2") as two, opened("join4") as four:
        tree = two["j2"]
        tree.add_friend(four["j42"])
        assert tree["b22"].array(0, 1) == ["j2-201"]  # its own comes first
        with opened("join1") as one:
            first = one["j1"]
            first.add_friend(two["j2"])
            first.add_friend(four["j42"])
            with pytest.raises(KeyError, match="'b22' is a branch of 2 friends"):
                first["b22"]
            assert "b22" not in first
            assert first["j42.b22"].array(0, 1) == ["j4-2-401"]


def test_a_friend_of_a_different_length_is_refused():
    with opened("join1") as one, opened("join4") as four:
        with pytest.raises(ValueError, match="'j41' has 11 entries and 'j1' has 10"):
            one["j1"].add_friend(four["j41"])


def test_two_friends_under_one_alias_are_refused():
    with opened("join1") as one, opened("join2") as two, opened("join3") as three:
        tree = one["j1"]
        tree.add_friend(two["j2"], "f")
        with pytest.raises(ValueError, match="already has a friend called 'f'"):
            tree.add_friend(three["j3"], "f")


def test_a_name_neither_the_tree_nor_a_friend_has_is_refused_by_name():
    with opened("join1") as one:
        with pytest.raises(KeyError, match="not a branch of 'j1' or of any of its friends"):
            one["j1"]["nothing"]


def test_a_friend_s_columns_are_read_for_the_entries_picked():
    with opened("join1") as one, opened("join2") as two:
        tree = one["j1"]
        tree.add_friend(two["j2"])
        picked = tree.arrays(["b10", "b21"], entries=[7, 2])
        assert picked["b10"].tolist() == [108.0, 103.0]
        assert picked["b21"].tolist() == [208, 203]


def test_a_chain_can_be_a_friend_of_a_tree_as_long():
    from xrdroot import chain

    with opened("join1") as one, chain("j2", [DATA / "join2.root"]) as friends:
        tree = one["j1"]
        tree.add_friend(friends, "c")
        assert tree["c.b21"].array(0, 2).tolist() == [201, 202]


# -- friends ROOT recorded ----------------------------------------------------


def friend_list(*friends: tuple[str, str, str]) -> bytes:
    """A ``TList`` of ``TFriendElement``, each its alias, file and tree name.

    It is what the tree's ``fFriends`` points at, so it names its class first.
    """
    out = Out()
    pointer = out.tag("TList")
    at = out.start(5)
    out.tobject()
    out.string("")
    out.pack("i", len(friends))
    for alias, file_name, tree_name in friends:
        tag = out.tag("TFriendElement")
        element = out.start(2)
        named = out.start(1)
        out.tobject()
        out.string(alias)
        out.string(file_name)
        out.end(named)
        out.string(tree_name)
        out.pack("?", False)  # fOwnFile
        out.end(element)
        out.end(tag)
        out.string("")
    out.end(at)
    out.end(pointer)
    return bytes(out.data)


def befriended(path: pathlib.Path, name: str, *friends: tuple[str, str, str]) -> None:
    """Write the tree ``name`` of ``path`` again, its record now naming ``friends``.

    The tree's record gains a friend list where its null ``fFriends`` pointer
    was; the new record goes on the end of the file, and the tree's entry in
    the key list is pointed at it - a key is the same length wherever it
    points, so nothing else in the file moves.
    """
    data = bytearray(path.read_bytes())
    with open_root(str(path)) as handle:
        key = handle._key(name)
        payload = key.payload(handle._source)
    assert not key.compressed and payload[-12:-8] == b"\x00" * 4  # the null fFriends
    listed = friend_list(*friends)
    grown = bytearray(payload[:-12] + listed + payload[-8:])
    count = struct.unpack_from(">I", grown, 0)[0] + len(listed) - 4
    struct.pack_into(">I", grown, 0, count)
    header = bytes(data[key.seek_key : key.seek_key + key.keylen])
    moved = bytearray(header)
    struct.pack_into(">iHi", moved, 0, key.keylen + len(grown), key.version, len(grown))
    struct.pack_into(">i", moved, 18, len(data))
    at = data.find(header, key.seek_key + key.keylen)
    assert at > 0
    data[at : at + len(header)] = moved
    data += moved + grown
    path.write_bytes(bytes(data))


def write_tree(path: pathlib.Path, name: str, **columns) -> None:
    with create(str(path), compression=None) as out:
        out[name] = columns


def test_a_friend_recorded_in_another_file_is_found_beside_this_one(tmp_path):
    write_tree(tmp_path / "main.root", "events", x=np.arange(4, dtype=np.int32))
    write_tree(tmp_path / "extra.root", "weights", w=np.arange(4) * 0.5)
    befriended(tmp_path / "main.root", "events", ("wt", "/far/away/extra.root", "weights"))
    with open_root(str(tmp_path / "main.root")) as handle:
        tree = handle["events"]
        assert tree["wt.w"].array().tolist() == [0.0, 0.5, 1.0, 1.5]
        assert tree["w"].array(1, 2).tolist() == [0.5]
        assert list(tree.friends) == ["wt"]
        opened_friend = handle._source.companions[0]
    assert opened_friend._source.handle.closed  # closed with the file that opened it


def test_a_friend_recorded_with_a_name_relative_to_this_file_is_found(tmp_path):
    (tmp_path / "sub").mkdir()
    write_tree(tmp_path / "main.root", "events", x=np.arange(3, dtype=np.int32))
    write_tree(tmp_path / "sub" / "extra.root", "more", y=np.arange(3, dtype=np.int32))
    befriended(tmp_path / "main.root", "events", ("more", "sub/extra.root", ""))
    with open_root(str(tmp_path / "main.root")) as handle:
        assert handle["events"]["more.y"].array().tolist() == [0, 1, 2]


def test_a_friend_recorded_in_the_same_file_is_read_out_of_it(tmp_path):
    with create(str(tmp_path / "both.root"), compression=None) as out:
        out["events"] = {"x": np.arange(3, dtype=np.int32)}
        out["extra"] = {"y": np.arange(3, dtype=np.int32) * 10}
    befriended(tmp_path / "both.root", "events", ("extra", "", "extra"))
    with open_root(str(tmp_path / "both.root")) as handle:
        assert handle["events"]["y"].array().tolist() == [0, 10, 20]
        assert handle._source.companions == []


def test_a_friend_recorded_under_this_file_s_own_name_is_this_file(tmp_path):
    with create(str(tmp_path / "both.root"), compression=None) as out:
        out["events"] = {"x": np.arange(2, dtype=np.int32)}
        out["extra"] = {"y": np.arange(2, dtype=np.int32)}
    befriended(tmp_path / "both.root", "events", ("e", "/elsewhere/both.root", "extra"))
    with open_root(str(tmp_path / "both.root")) as handle:
        assert handle["events"]["e.y"].array().tolist() == [0, 1]
        assert handle._source.companions == []


def test_a_recorded_friend_that_cannot_be_found_says_where_it_looked(tmp_path):
    write_tree(tmp_path / "main.root", "events", x=np.arange(2, dtype=np.int32))
    befriended(tmp_path / "main.root", "events", ("gone", "gone.root", "t"))
    with open_root(str(tmp_path / "main.root")) as handle:
        with pytest.raises(FileNotFoundError, match=r"is in 'gone\.root', which is neither"):
            handle["events"]["gone.x"]
        with pytest.raises(FileNotFoundError):  # asked again, it is looked for again
            _ = handle["events"].friends
        assert handle["events"]["x"].array().tolist() == [0, 1]  # its own still read


def test_a_friend_s_record_says_what_it_names():
    record = FriendRecord("wt", "weights", "extra.root")
    assert repr(record) == "<FriendRecord 'wt': 'weights' in 'extra.root'>"
    assert repr(FriendRecord("a", "a", "")) == "<FriendRecord 'a': 'a'>"


def test_a_friend_of_a_remote_file_is_looked_for_beside_it_on_the_same_server():
    own = "root://host//store/run/main.root"
    assert friend_target("/tmp/job/extra.root", own) == "root://host//store/run/extra.root"
    assert friend_target("root://other//x.root", own) == "root://other//x.root"
    assert friend_target("/anywhere/main.root", own) is None
    assert friend_target("", own) is None


def test_a_tree_older_than_the_friend_layout_or_not_in_it_has_no_friends():
    assert _tree_friends(Buffer(b""), 15) == []
    assert _tree_friends(Buffer(b"\x00\x01"), 20) == []  # a tail that does not read
    empty = Out()
    at = empty.start(3)
    empty.end(at)
    tail = bytes(empty.data) + struct.pack(">IiiI", 0, 0, 0, 0) + b"\x00\x00\x00\x00"
    assert _tree_friends(Buffer(tail), 20) == []  # a null fFriends
    listed = Buffer(bytes(empty.data) + struct.pack(">IiiI", 0, 0, 0, 0) + friend_list())
    assert _tree_friends(listed, 20) == []


def test_a_friend_list_holding_something_else_keeps_only_the_friends():
    empty = Out()
    at = empty.start(3)
    empty.end(at)
    out = Out()
    listed = out.start(5)
    out.tobject()
    out.string("")
    out.pack("i", 1)
    tag = out.tag("TObjString")
    inner = out.start(1)
    out.end(inner)
    out.end(tag)
    out.string("")
    out.end(listed)
    tail = bytes(empty.data) + struct.pack(">IiiI", 0, 0, 0, 0) + bytes(out.data)
    assert _tree_friends(Buffer(tail), 20) == []


def test_a_tree_file_copied_with_its_friend_finds_it_in_the_new_place(tmp_path):
    write_tree(tmp_path / "main.root", "events", x=np.arange(2, dtype=np.int32))
    write_tree(tmp_path / "extra.root", "more", y=np.arange(2, dtype=np.int32))
    befriended(tmp_path / "main.root", "events", ("more", str(tmp_path / "extra.root"), "more"))
    moved = tmp_path / "moved"
    moved.mkdir()
    for name in ("main.root", "extra.root"):
        shutil.copy(tmp_path / name, moved / name)
    (tmp_path / "extra.root").unlink()
    with open_root(str(moved / "main.root")) as handle:
        assert handle["events"]["y"].array().tolist() == [0, 1]
