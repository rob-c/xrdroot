"""RNTuples read together - chained and joined - and a vector read as a collection."""

from __future__ import annotations

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.ntuple.views import RNTupleLocalRange


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _write(path: str, name: str, columns: dict[str, object], kinds: dict[str, object]) -> None:
    with xrdroot.create(path) as f:
        f.rntuple(name, kinds).extend(columns)


def test_a_chain_reads_rntuples_one_after_another_numbering_on() -> None:
    _write("a.root", "a", {"x": [1.0, 2.0]}, {"x": "double"})
    _write("b.root", "b", {"x": [3.0]}, {"x": "double"})
    second = ROOT.RNTupleOpenSpec("b", "b.root")
    specs = ROOT.std.vector["RNTupleOpenSpec"]([["a", "a.root"], second])
    processor = ROOT.RNTupleProcessor.CreateChain(specs)
    x = processor.RequestField["double"]("x")
    seen = [(index, processor.GetCurrentProcessorNumber(), x.value) for index in processor]
    assert seen == [(0, 0, 1.0), (1, 0, 2.0), (2, 1, 3.0)]
    assert processor.GetNEntriesProcessed() == 3


def test_a_join_reads_each_primary_entry_with_the_auxiliary_one_of_its_keys() -> None:
    _write("main.root", "main", {"i": [0, 1, 5], "px": [1.0, 2.0, 3.0]},
           {"i": "std::uint32_t", "px": "float"})  # fmt: skip
    _write("aux.root", "aux", {"i": [1, 0], "py": [10.0, 20.0]},
           {"i": "std::uint32_t", "py": "float"})  # fmt: skip
    processor = ROOT.RNTupleProcessor.CreateJoin(["main", "main.root"], ["aux", "aux.root"], ["i"])
    px = processor.RequestField["float"]("px")
    py = processor.RequestField["float"]("aux.py")
    seen = [(px.value, py.value) for _ in processor]
    assert seen == [(1.0, 20.0), (2.0, 10.0), (3.0, 10.0)]  # no match for 5: the last stays
    assert processor.GetNEntriesProcessed() == 3
    by_number = ROOT.RNTupleProcessor.CreateJoin(["main", "main.root"], ["aux", "aux.root"], [])
    py = by_number.RequestField["float"]("aux.py")
    assert [py.value for _ in by_number] == [10.0, 20.0, 20.0]


def test_a_vector_read_as_a_collection_gives_sizes_ranges_and_items() -> None:
    _write("v.root", "v", {"v": [[1, 2, 3], [], [4]]}, {"v": "std::vector<std::int32_t>"})
    options = ROOT.RNTupleReadOptions()
    options.SetClusterCache(ROOT.RNTupleReadOptions.EClusterCache.kOff)
    assert options.GetClusterCache() == 0
    reader = ROOT.RNTupleReader.Open("v", "v.root", options)
    collection = ROOT.RNTupleCollectionView(reader.GetCollectionView("v"))
    assert [collection(entry) for entry in range(3)] == [3, 0, 1]
    items = collection.GetView["int"]("_0")
    span = collection.GetCollectionRange(0)
    it, end = RNTupleLocalRange.RIterator(span.begin()), span.end()
    walked = []
    while it != end:
        walked.append(items(it))
        it += 1
    assert walked == [1, 2, 3] and list(span) == [0, 1, 2] and span.size() == 3
    assert int(it) == 3 and it == span.end() and it != 3
    empty = RNTupleLocalRange(ROOT.kInvalidDescriptorId, ROOT.kInvalidNTupleIndex,
                              ROOT.kInvalidNTupleIndex)  # fmt: skip
    assert empty.size() == 0 and ROOT.NTupleSize_t(4) == 4
