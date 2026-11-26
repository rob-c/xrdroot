"""Compression by ROOT's names: ``RCompressionSetting``, and a branch compressed its own way.

``TBranch::SetCompressionSettings`` packs one branch's baskets otherwise than
its file, and says so in the branch's ``fCompress``, which ``hadd`` and
``GetCompressionSettings`` read back - what roottest's ``root/io/hadd`` checks.
"""

from __future__ import annotations

import struct
from array import array

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.compression import algorithm
from xrdroot.pycxx.runtime import Cell
from xrdroot.pycxx.runtime import array as c_array


def _basket_algorithm(raw: bytes, seek: int) -> str:
    """What compressed the basket whose key starts at ``seek``."""
    keylen = struct.unpack(">h", raw[seek + 14 : seek + 16])[0]
    return algorithm(raw[seek + keylen : seek + keylen + 2])


def test_the_settings_root_recommends_and_the_names_of_its_algorithms():
    settings = ROOT.RCompressionSetting
    assert settings.EDefaults.kUseAnalysis == 404
    assert settings.EDefaults.kUseCompiledDefault == 101
    assert settings.ELevel.kDefaultZSTD == 5
    assert settings.AlgorithmToString(settings.EAlgorithm.kZSTD) == "zstd"
    assert settings.AlgorithmToString(-1) == "Undefined compression algorithm"
    assert settings.AlgorithmFromCompressionSettings(505) == settings.EAlgorithm.kZSTD
    assert settings.AlgorithmFromCompressionSettings(950) == settings.EAlgorithm.kUndefined


def test_a_setting_is_spelled_from_an_algorithm_and_a_level_as_root_spells_it():
    assert ROOT.CompressionSettings(ROOT.RCompressionSetting.EAlgorithm.kZSTD, 5) == 505
    assert ROOT.CompressionSettings(7, 150) == 99
    assert ROOT.CompressionSettings(-1, 5) == -95
    assert ROOT.ROOT.CompressionSettings(1, 1) == 101


def _branch_file(path: str) -> None:
    """A tree whose branch ``x`` is LZ4 at level 4 in a file of zlib at level 1."""
    out = ROOT.TFile(path, "RECREATE")
    out.SetCompressionSettings(101)
    tree = ROOT.TTree("t", "t")
    x, y = array("i", [0]), array("i", [0])
    branch = tree.Branch("x", x, "x/I")
    tree.Branch("y", y, "y/I")
    assert branch.GetCompressionSettings() == 101
    branch.SetCompressionSettings(404)
    for value in range(2000):
        x[0] = y[0] = value % 7
        tree.Fill()
    assert tree.GetBranch("x").GetCompressionSettings() == 404
    assert tree.GetBranch("y").GetCompressionSettings() == 101
    tree.Write()
    out.Close()


def test_a_branch_compressed_its_own_way_packs_its_baskets_so(tmp_path):
    path = str(tmp_path / "branch.root")
    _branch_file(path)
    with xrdroot.open_root(path) as found:
        x_record = found["t"].branches["x"].record
        y_record = found["t"].branches["y"].record
        assert (x_record.compress, y_record.compress) == (404, 101)
        raw = (tmp_path / "branch.root").read_bytes()
        assert [
            _basket_algorithm(raw, seek)
            for seek in (x_record.basket_seek[0], y_record.basket_seek[0])
        ] == ["lz4", "zlib"]
        assert found["t"].arrays(["x"])["x"][:8].tolist() == [0, 1, 2, 3, 4, 5, 6, 0]


def test_a_branch_read_says_how_it_was_compressed_and_can_be_told_otherwise(tmp_path):
    path = str(tmp_path / "branch.root")
    _branch_file(path)
    back = ROOT.TFile(path)
    read = back.Get("t").GetBranch("x")
    assert read.GetCompressionSettings() == 404
    assert read.GetCompressionAlgorithm() == 4 and read.GetCompressionLevel() == 4
    read.SetCompressionLevel(9)
    assert read.GetCompressionSettings() == 409
    read.SetCompressionAlgorithm(5)
    assert read.GetCompressionSettings() == 509
    back.Close()


def test_a_tree_in_no_file_and_a_branch_given_no_setting_take_roots_default():
    tree = ROOT.TTree("free", "free")
    tree.SetDirectory(0)
    value = np.zeros(1, dtype=np.float64)
    branch = tree.Branch("v", value, "v/D")
    assert branch.GetCompressionSettings() == 101
    branch.SetCompressionSettings()
    assert branch.GetCompressionSettings() == 101
    objects = ROOT.TTree("objects", "objects")
    objects.SetDirectory(0)
    held = ROOT.std.vector("double")()
    assert objects.Branch("vec", held).GetCompressionSettings() == 101
    objects._directory = type("Raw", (), {"GetCompressionSettings": lambda self: 0})()
    assert objects.GetBranch("vec").GetCompressionSettings() == 0


def test_a_tree_handed_to_writeobject_is_written_under_the_name_given(tmp_path):
    path = str(tmp_path / "written.root")
    with ROOT.TFile(
        path, "RECREATE", "myfile", ROOT.RCompressionSetting.EDefaults.kUseAnalysis
    ) as out:
        tree = ROOT.TTree("mytree", "mytree")
        x = array("i", [42])
        tree.Branch("x", x, "x/I")
        tree.Fill()
        out.WriteObject(tree, "renamed")
    with xrdroot.open_root(path) as found:
        assert found.compression == 404
        assert found["renamed"].arrays(["x"])["x"].tolist() == [42]


def test_a_file_told_a_level_zlib_has_not_got_keeps_it_and_packs_at_its_top(tmp_path):
    """``IoBigEventGenerator`` passes 101 to ``SetCompressionLevel``, as roottest's
    ``root/io/compression`` does, and ROOT keeps that as zlib at level 99: the file
    says 199 while its blocks were made at zlib's own top level, 9."""
    path = str(tmp_path / "level99.root")
    out = ROOT.TFile(path, "RECREATE")
    out.SetCompressionLevel(101)
    assert out.GetCompressionSettings() == 199
    tree = ROOT.TTree("t", "t")
    x = array("i", [0])
    tree.Branch("x", x, "x/I")
    for value in range(2000):
        x[0] = value % 7
        tree.Fill()
    tree.Write()
    out.Close()
    with xrdroot.open_root(path) as found:
        assert found.compression == 199
        record = found["t"].branches["x"].record
        raw = (tmp_path / "level99.root").read_bytes()
        assert _basket_algorithm(raw, record.basket_seek[0]) == "zlib"
        assert found["t"].arrays(["x"])["x"].sum() == sum(value % 7 for value in range(2000))


# -- R__zipMultipleAlgorithm: ROOT's own zipper, and the block it makes --

@pytest.mark.parametrize(
    ("algorithm", "signature"),
    [(1, b"ZL"), (2, b"XZ"), (3, b"CS"), (4, b"L4"), (5, b"ZS")],
)
def test_the_zipper_writes_a_block_each_algorithm_can_read_back(
    algorithm: int, signature: bytes
) -> None:
    """``R__zipMultipleAlgorithm`` against the kit's own reader, one block per algorithm."""
    data = bytes([7] * 8 + [1, 2, 3, 4]) * 4
    source = c_array("char", len(data))
    for index, value in enumerate(data):
        source[index] = value - 256 if value > 127 else value
    target = c_array("char", 96)
    size, room, written = Cell(len(data), "int"), Cell(96, "int"), Cell(-1, "int")
    ROOT.R__zipMultipleAlgorithm(6, size, source, room, target, written, algorithm)
    block = bytes(int(value) & 0xFF for value in target[:written.value])
    assert block[:2] == signature
    assert xrdroot.compression.decompress(block, len(data)) == data


def test_the_zipper_leaves_the_buffer_alone_when_the_block_will_not_fit() -> None:
    """``*irep`` stays 0, which is how a caller knows to store the object raw."""
    source = c_array("char", 32)
    target = c_array("char", 32)
    size, room, written = Cell(32, "int"), Cell(12, "int"), Cell(-1, "int")
    ROOT.R__zipMultipleAlgorithm(6, size, source, room, target, written, 1)
    assert written.value == 0 and all(int(value) == 0 for value in target)


def test_the_zipper_does_nothing_for_a_source_or_a_level_too_small_to_bother() -> None:
    source, target = c_array("char", 16), c_array("char", 16)
    for size, room, level in ((10, 16, 6), (16, 9, 6), (16, 16, 0)):
        written = Cell(-1, "int")
        ROOT.R__zipMultipleAlgorithm(level, Cell(size, "int"), source, Cell(room, "int"),
                                     target, written, 1)  # fmt: skip
        assert written.value == 0


def test_the_unknown_algorithm_number_is_zlib_as_roots_own_default_is() -> None:
    data = bytes(range(40))
    source = c_array("char", len(data))
    for index, value in enumerate(data):
        source[index] = value
    target = c_array("char", 96)
    written = Cell(-1, "int")
    ROOT.R__zipMultipleAlgorithm(9, Cell(len(data), "int"), source, Cell(96, "int"), target,
                                 written, 0)  # fmt: skip
    assert bytes(int(value) & 0xFF for value in target[:2]) == b"ZL"
