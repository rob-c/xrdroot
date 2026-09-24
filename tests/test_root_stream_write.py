"""The writer sends records as it makes them, and takes them back on failure.

A file bigger than memory is the point of writing a tree basket by basket, so
these check that the bytes really do leave before the close - and that the
promise the in-memory writer used to keep for free, no file rather than half
a file, still holds now that half a file has already gone out.
"""

from __future__ import annotations

import io

import pytest

from xrdroot import create, open_root
from xrdroot import writer as writing


class Watched(io.BytesIO):
    """A target that remembers how long it had grown by each write."""

    def __init__(self, prefix: bytes = b"") -> None:
        super().__init__(prefix)
        self.seek(len(prefix))
        self.lengths: list[int] = []

    def write(self, data) -> int:  # type: ignore[override]
        count = super().write(data)
        self.lengths.append(len(self.getvalue()))
        return count


class OneWay(io.RawIOBase):
    """A target that only takes bytes in order, like a pipe or an upload."""

    def __init__(self) -> None:
        self.data = bytearray()

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:  # type: ignore[override]
        self.data += data
        return len(data)


def fill(out, entries: int) -> None:
    tree = out.tree("events", {"x": "d", "n": "i"}, basket_size=4096)
    for entry in range(entries):
        tree.fill(x=entry * 0.5, n=entry)


def test_baskets_leave_for_the_target_before_the_file_is_closed(monkeypatch):
    monkeypatch.setattr(writing, "WRITE_BEHIND", 16_384)
    target = Watched()
    with create(target) as out:
        fill(out, 20_000)
        assert target.lengths, "nothing went out while the tree was being filled"
        assert target.lengths[-1] > 30_000
    with open_root(io.BytesIO(target.getvalue())) as back:
        tree = back["events"]
        assert len(tree) == 20_000
        assert tree["n"].array()[-1] == 19_999
        assert tree["x"].array()[12_345] == 6172.5


def test_a_target_that_cannot_seek_is_written_whole_at_the_close():
    target = OneWay()
    with create(target) as out:
        fill(out, 3_000)
        assert not target.data
    with open_root(io.BytesIO(bytes(target.data))) as back:
        assert len(back["events"]) == 3_000
    target = OneWay()
    with pytest.raises(RuntimeError):
        with create(target) as out:
            fill(out, 3_000)
            raise RuntimeError("nothing of this was sent, so nothing is taken back")
    assert not target.data


def test_a_failure_takes_back_what_had_already_been_sent(monkeypatch, tmp_path):
    monkeypatch.setattr(writing, "WRITE_BEHIND", 1_024)
    target = Watched()
    with pytest.raises(RuntimeError):
        with create(target) as out:
            fill(out, 5_000)
            assert target.lengths
            raise RuntimeError("the analysis fell over half way")
    assert target.getvalue() == b""
    path = tmp_path / "half.root"
    with pytest.raises(RuntimeError):
        with create(str(path)) as out:
            fill(out, 5_000)
            raise RuntimeError("likewise")
    assert path.read_bytes() == b""


def test_a_handle_that_already_held_something_keeps_it_and_the_file_follows():
    prefix = b"not a ROOT file, and not ours to touch"
    target = Watched(prefix)
    with create(target) as out:
        out["s"] = "after the prefix"
    data = target.getvalue()
    assert data.startswith(prefix)
    with open_root(io.BytesIO(data[len(prefix) :])) as back:
        assert back["s"] == "after the prefix"
    target = Watched(prefix)
    with pytest.raises(RuntimeError):
        with create(target) as out:
            out["s"] = "never to be seen"
            raise RuntimeError("gone wrong")
    assert target.getvalue() == prefix
