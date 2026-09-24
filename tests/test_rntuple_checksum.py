"""XXH3-64 in Python, held to the digests the C library and ROOT itself give.

An RNTuple that carries a wrong checksum is one ROOT refuses, so the Python
XXH3 the writer falls back on has to be exactly right. It is held here to a
table of digests the ``xxhash`` package computed - one per length range, since
XXH3 is a different recipe for each - and, with nothing installed at all, to
the checksums ROOT wrote into the envelopes of the files under ``tests/data``.
"""

from __future__ import annotations

import pathlib
import struct

import pytest

import xrdroot
from xrdroot.rntuple import checksum as xxh3
from xrdroot.rntuple.envelope import Anchor, read_block

DATA = pathlib.Path(__file__).parent / "data" / "rntuple"

#: Bytes that are not all alike, so that a lane mixed up with its neighbour shows.
SAMPLE = bytes((i * 7 + 3) % 251 for i in range(3000))

#: The digest of ``SAMPLE[:n]``, as ``xxhash.xxh3_64_intdigest`` gives it: every
#: length range XXH3 has a recipe for, and both sides of each boundary.
DIGESTS = [
    (0, 0x2D06800538D394C2),
    (1, 0x13E608BC156DEFED),
    (2, 0x1C9074B93943B86C),
    (3, 0xA9088DDA485B481C),
    (4, 0x6D9253B16C8B1ED3),
    (8, 0x60539DB630471163),
    (9, 0xFEFF668361D723A8),
    (16, 0xB8C859B0F030B585),
    (17, 0x714A04408E79B80F),
    (32, 0x19FF4EE1D6BA1A55),
    (33, 0x3E44983AD21679C8),
    (64, 0x38BCDE5122F74956),
    (65, 0x95A166C5957453D9),
    (96, 0x75D654BDAEE123DF),
    (97, 0x1296F9E2421AB74C),
    (128, 0x4634AE6A253A60E4),
    (129, 0xC095B9B1B087722D),
    (200, 0xA369F2930049476F),
    (240, 0x887AF00281F75D38),
    (241, 0x82B1DE299F6E411E),
    (1024, 0xF75E768C7CDD54B2),
    (1025, 0x667A5EABE344E5DF),
    (2048, 0x9E5E4A8160109A5D),
    (3000, 0xB54504A9E625239F),
]


@pytest.mark.parametrize(("length", "digest"), DIGESTS)
def test_the_python_xxh3_gives_the_digest_the_c_library_gives(length, digest):
    assert xxh3.xxh3_64(SAMPLE[:length]) == digest


def test_the_python_xxh3_and_the_installed_one_agree_on_every_length_up_to_two_blocks():
    xxhash = pytest.importorskip("xxhash")
    for length in range(0, 2200, 7):
        data = SAMPLE[:length]
        assert xxh3.xxh3_64(data) == xxhash.xxh3_64_intdigest(data), length


def test_the_python_xxh3_matches_the_checksums_root_wrote_into_its_envelopes():
    with xrdroot.open_root(DATA / "test_stl_containers_rntuple_v1-0-0-0.root") as f:
        key = f._key("ntuple")
        anchor = Anchor.parse(key.payload(f._source), "ntuple")
        for link in (anchor.header, anchor.footer):
            raw = read_block(f._source, link, anchor.max_key)
            assert xxh3.xxh3_64(raw[:-8]) == struct.unpack("<Q", raw[-8:])[0]
        payload = key.payload(f._source)
        assert xxh3.xxh3_64(payload[6:70]) == struct.unpack(">Q", payload[70:78])[0]


def test_without_xxhash_checksums_are_computed_in_python_and_not_checked_on_the_way_in(
    monkeypatch,
):
    monkeypatch.setattr(xxh3, "_fast", None)
    assert not xxh3.verifying()
    assert xxh3.checksum(SAMPLE[:100]) == xxh3.xxh3_64(SAMPLE[:100])


def test_with_xxhash_the_c_checksum_is_used_and_checked_on_the_way_in(monkeypatch):
    monkeypatch.setattr(xxh3, "_fast", lambda data: 42)
    assert xxh3.verifying()
    assert xxh3.checksum(b"anything") == 42
