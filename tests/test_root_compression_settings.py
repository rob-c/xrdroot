"""ROOT's compression settings, packed as ROOT packs them, byte for byte.

``compressed-settings-6.40.root`` was written by ROOT 6.40 (see
``compressed_settings.C`` beside it): a histogram and a small ``TNamed`` per
setting, each named for the setting it was compressed with. Packing what
each holds again must give ROOT's bytes exactly - zlib at the level, xz with
a CRC32 check and a dictionary a quarter of the block, zstd at twice the
level, LZ4's high-compression mode from level four. LZ4's fast mode is left
out: LZ4 1.10, which ROOT 6.40 is built with, packs differently there from
the versions the ``lz4`` wheel bundles.
"""

from __future__ import annotations

import pathlib

import pytest

import xrdroot
from xrdroot.compression import CODES, compress, decompress, spelled

DATA = pathlib.Path(__file__).parent / "data"
FILE = DATA / "compressed-settings-6.40.root"
SETTINGS = [101, 106, 201, 207, 209, 404, 409, 501, 505, 509]


def _records(name: str) -> tuple[bytes, int]:
    """The record of key ``name`` as it is on file, and how long it is unpacked."""
    with xrdroot.open_root(FILE) as found:
        key = found.key(name)
    raw = FILE.read_bytes()[key.seek_key + key.keylen : key.seek_key + key.nbytes]
    return raw, key.objlen


@pytest.mark.parametrize("setting", SETTINGS)
def test_every_setting_packs_what_root_packed_byte_for_byte(setting):
    if setting // 100 == CODES["zstd"]:
        pytest.importorskip("compression.zstd", reason="zstd is in the standard library from 3.14")
    if setting // 100 == CODES["lz4"]:
        pytest.importorskip("lz4.block")
    packed, objlen = _records(f"h{setting}")
    algorithm, level = spelled(setting)
    assert algorithm is not None
    assert compress(decompress(packed, objlen), algorithm, level) == packed


@pytest.mark.parametrize("setting", SETTINGS)
def test_an_object_of_256_bytes_or_fewer_is_left_raw_as_root_leaves_it(setting, tmp_path):
    packed, objlen = _records(f"n{setting}")
    assert len(packed) == objlen
    with xrdroot.create(tmp_path / "small.root", compression="zlib", level=9) as out:
        out["small"] = "a" * 250
        out["large"] = "a" * 260
    with xrdroot.open_root(tmp_path / "small.root") as found:
        small, large = found.key("small"), found.key("large")
    assert small.nbytes - small.keylen == small.objlen
    assert large.nbytes - large.keylen < large.objlen


def test_a_setting_names_its_algorithm_in_its_hundreds_and_level_zero_stores_raw():
    assert spelled(505) == ("zstd", 5)
    assert spelled(1) == ("zlib", 1)
    assert spelled(207) == ("lzma", 7)
    assert spelled(400) == (None, None)
    assert spelled(901) == ("zlib", 1)


def test_zstd_past_its_highest_level_is_held_there_as_zstd_holds_it():
    pytest.importorskip("compression.zstd", reason="zstd is in the standard library from 3.14")
    data = b"abc" * 300
    assert compress(data, "zstd", 99) == compress(data, "zstd", 11)
    assert decompress(compress(data, "zstd", 99), len(data)) == data
