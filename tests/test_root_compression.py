"""LZ4 and XXH64 twice over: in C when the ``lz4`` extra is installed, in Python always.

The accelerated codec and checksum are only worth having if they agree with
the Python ones they stand in for, so that is most of what is asserted here:
the same digests for the same bytes, the same bytes back from the same block,
and blocks from either encoder undone by either decoder. The Python paths are
reached by taking the accelerated ones away, so both stay exercised on a
machine that has the extra installed.
"""

from __future__ import annotations

import pathlib
import struct

import pytest

from xrdroot import FormatError
from xrdroot import compression as codec
from xrdroot.compression import _checksum, _lz4, _lz4_pack, _xxh64, compress, decompress

pytest.importorskip("lz4.block")
xxhash = pytest.importorskip("xxhash")

DATA = pathlib.Path(__file__).parent / "data"


def basket(rows: int) -> bytes:
    """Something shaped like a real basket: big-endian floats and counters."""
    return b"".join(struct.pack(">dii", i * 0.25, i % 7, i // 3) for i in range(rows))


SHAPES = [
    b"",
    b"x",
    b"twelve bytes",
    b"abcd" * 64,
    b"a" * 1000,
    bytes(range(256)) * 3,
    basket(2000),
]


@pytest.fixture
def pure(monkeypatch):
    """The module with its accelerators taken away, as if the extra were absent."""
    monkeypatch.setattr(codec, "_lz4_fast", None)
    monkeypatch.setattr(codec, "_lz4_fast_pack", None)
    monkeypatch.setattr(codec, "_xxh64_fast", None)


def framed(body: bytes, size: int, checksum: int | None = None) -> bytes:
    """A ROOT LZ4 block around ``body``, with its true checksum unless told otherwise."""
    if checksum is None:
        checksum = xxhash.xxh64_intdigest(body)
    payload = checksum.to_bytes(8, "big") + body
    return b"L4\x01" + len(payload).to_bytes(3, "little") + size.to_bytes(3, "little") + payload


def test_the_python_xxh64_and_the_c_one_agree_on_every_length_that_matters():
    data = basket(40)
    for length in [*range(0, 80), 255, 256, 1023, len(data)]:
        assert _xxh64(data[:length]) == xxhash.xxh64_intdigest(data[:length])


def test_the_checksum_comes_out_the_same_with_the_extra_and_without(monkeypatch):
    data = basket(500)
    fast = _checksum(data)
    monkeypatch.setattr(codec, "_xxh64_fast", None)
    assert _checksum(data) == fast == _xxh64(data)


def test_the_c_decoder_gives_the_same_bytes_as_the_python_one_for_either_encoder():
    import lz4.block

    for data in SHAPES:
        for body in (_lz4_pack(data), lz4.block.compress(data, store_size=False)):
            assert _lz4(body, len(data)) == data
            assert codec._unpack_lz4(framed(body, len(data))[9:], len(data)) == data


@pytest.mark.parametrize("level", [1, 9])
def test_lz4_round_trips_with_the_extra_at_a_fast_level_and_a_high_one(level):
    for data in SHAPES:
        packed = compress(data, "lz4", level)
        assert decompress(packed, len(data)) == data


def test_lz4_round_trips_without_the_extra(pure):
    for data in SHAPES:
        assert decompress(compress(data, "lz4"), len(data)) == data


def test_a_block_written_without_the_extra_is_read_with_it_and_the_other_way(monkeypatch):
    data = basket(3000)
    fast = compress(data, "lz4")
    monkeypatch.setattr(codec, "_lz4_fast_pack", None)
    monkeypatch.setattr(codec, "_xxh64_fast", None)
    slow = compress(data, "lz4")
    assert decompress(fast, len(data)) == data
    monkeypatch.undo()
    assert decompress(slow, len(data)) == data


def test_the_high_compression_mode_is_what_level_four_and_up_ask_for():
    data = basket(5000)
    assert len(compress(data, "lz4", 9)) < len(compress(data, "lz4", 1))


def test_an_lz4_block_that_disagrees_with_its_checksum_is_refused():
    data = basket(100)
    body = _lz4_pack(data)
    with pytest.raises(FormatError, match="does not match the checksum"):
        decompress(framed(body, len(data), checksum=_xxh64(body) ^ 1), len(data))


def test_without_xxhash_the_checksum_is_passed_over_as_it_always_was(pure):
    data = basket(100)
    body = _lz4_pack(data)
    assert decompress(framed(body, len(data), checksum=0), len(data)) == data


def test_the_c_decoder_refuses_a_damaged_block_as_a_format_error():
    with pytest.raises(FormatError, match="would not decode"):
        decompress(framed(b"\xf0", 4), 4)


def test_the_c_decoder_will_not_hand_back_fewer_bytes_than_were_promised():
    import lz4.block

    body = lz4.block.compress(b"abcd" * 100, store_size=False)
    with pytest.raises(FormatError, match="gave 400 bytes where 500 were promised"):
        decompress(framed(body, 500), 500)


def test_blocks_root_itself_wrote_pass_the_checksum_check_on_the_way_in():
    raw = (DATA / "dirs-6.14.00.root").read_bytes()
    checked, pos = 0, raw.find(b"L4\x01")
    while pos != -1:
        packed = int.from_bytes(raw[pos + 3 : pos + 6], "little")
        unpacked = int.from_bytes(raw[pos + 6 : pos + 9], "little")
        block = raw[pos + 9 : pos + 9 + packed]
        if packed > 8 and xxhash.xxh64_intdigest(block[8:]) == int.from_bytes(block[:8], "big"):
            assert decompress(raw[pos : pos + 9 + packed], unpacked) == _lz4(block[8:], unpacked)
            checked += 1
        pos = raw.find(b"L4\x01", pos + 1)
    assert checked >= 2
