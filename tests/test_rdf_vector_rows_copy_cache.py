"""Python's native deepcopy slot cache must not disable native RDF row boxing."""

import copy
from typing import Any

import numpy as np
import pytest

from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf import entrywise, rvec, vectorrows
from xrdroot.tree import Jagged


def _assert_copy_storage(kind: Any, source: Any) -> None:
    first, second = copy.deepcopy(source), copy.deepcopy(source)
    assert type(kind.__slotnames__) is list
    assert kind.__slotnames__ == ['_data', '_size']
    assert first._data is not source._data and second._data is not first._data
    assert first.capacity() == second.capacity() == source.capacity()


def _assert_independent_rows(actual: Any, expected: Any, content: Any) -> None:
    assert actual is not None
    assert len(actual) == len(expected)
    for found, original in zip(actual, expected, strict=True):
        assert type(found) is type(original)
        assert found.data().tobytes() == original.data().tobytes()
        assert not np.shares_memory(found.data(), content)


@pytest.mark.parametrize('name', list(stl.CPP_NAMES))
@pytest.mark.parametrize('order', ['<', '>'])
def test_real_deepcopies_leave_row_boxing_bit_exact_and_independent(
    monkeypatch: pytest.MonkeyPatch, name: str, order: str,
) -> None:
    content = np.arange(20).astype(np.dtype(name).newbyteorder(order))[::2]
    column = Jagged(content, [0, 3, 3, 7])
    kind = rvec.RVec[column.content.dtype]
    monkeypatch.delattr(kind, '__slotnames__', raising=False)
    source = kind([0, 1])
    source.reserve(17)
    _assert_copy_storage(kind, source)
    expected = [rvec._made(np.asarray(row)) for row in column]
    actual = vectorrows.rows(column, rvec._made)
    _assert_independent_rows(actual, expected, content)
    actual[0][0] = 1
    assert content[0] == 0 and list(actual[1]) == []
    assert [each.data().tobytes() for each in entrywise._entries(column)] == [
        each.data().tobytes() for each in expected
    ]


@pytest.mark.parametrize('cached', [None, (), ('_data', '_size'), {}, [], ['_data'],
                                    ['_data', '_size', 'extra'], ['_size', '_data'],
                                    ['_data', '_data'], [0, '_size']])
def test_malformed_slot_caches_keep_the_original_row_constructor(
    monkeypatch: pytest.MonkeyPatch, cached: Any,
) -> None:
    column = Jagged(np.array([1., 2., 3.]), [0, 2, 3])
    kind = rvec.RVec[column.content.dtype]
    monkeypatch.setattr(kind, '__slotnames__', cached, raising=False)
    assert vectorrows.rows(column, rvec._made) is None
    assert [list(row) for row in entrywise._entries(column)] == [[1., 2.], [3.]]


@pytest.mark.parametrize('custom', ['list', 'name', 'descriptor'])
def test_foreign_cache_operations_are_never_entered_during_eligibility(
    monkeypatch: pytest.MonkeyPatch, custom: str,
) -> None:
    calls = []
    class Cache(list):
        def __len__(self):
            calls.append('len')
            raise AssertionError('A foreign cache must not be entered.')
        def __iter__(self):
            calls.append('iter')
            raise AssertionError('A foreign cache must not be entered.')
        def __eq__(self, other):
            calls.append('eq')
            raise AssertionError('A foreign cache must not be compared.')
    class Name(str):
        def __eq__(self, other):
            calls.append('name eq')
            raise AssertionError('A foreign slot name must not be compared.')
    def descriptor(self):
        calls.append('descriptor')
        raise AssertionError('A cache descriptor must not be read.')
    column = Jagged(np.array([1., 2.]), [0, 2])
    kind = rvec.RVec[column.content.dtype]
    cache = (Cache(['_data', '_size']) if custom == 'list' else
             [Name('_data'), '_size'] if custom == 'name' else property(descriptor))
    monkeypatch.setattr(kind, '__slotnames__', cache, raising=False)
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []


def test_the_valid_copy_cache_does_not_allow_other_specialization_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    column = Jagged(np.array([1., 2.]), [0, 2])
    kind = rvec.RVec[column.content.dtype]
    copy.deepcopy(kind([1, 2]))
    monkeypatch.setattr(kind, 'foreign_metadata', 'extra', raising=False)
    assert vectorrows.rows(column, rvec._made) is None
    assert [list(row) for row in entrywise._entries(column)] == [[1., 2.]]


@pytest.mark.parametrize(('name', 'integer', 'bits'), [
    ('float32', 'uint32', [0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 1]),
    ('float64', 'uint64', [0x8000000000000000, 0x7ff0000000000000,
                         0xfff0000000000000, 0x7ff8000000001234, 1]),
])
def test_real_copy_cache_keeps_signed_zero_and_nan_payload_bits_in_boxed_rows(
    name: str, integer: str, bits: list[int],
) -> None:
    content = np.array(bits, dtype=integer).view(name)
    column = Jagged(content, [0, 3, 5])
    kind = rvec.RVec[column.content.dtype]
    copy.deepcopy(kind([0, 1]))
    assert type(kind.__slotnames__) is list
    assert kind.__slotnames__ == ['_data', '_size']
    rows = vectorrows.rows(column, rvec._made)
    assert rows is not None
    assert [row.data().view(integer).tolist() for row in rows] == [bits[:3], bits[3:]]
    assert all(not np.shares_memory(row.data(), content) for row in rows)
    rows[0][0] = 1
    assert content.view(integer).tolist() == bits
    assert rows[1].data().view(integer).tolist() == bits[3:]
