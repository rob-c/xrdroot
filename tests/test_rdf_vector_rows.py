"""Preparing native vector rows keeps scalar construction and its observable hooks."""
import functools

import numpy as np
import pytest

from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf import entrywise, rvec, vectorrows
from xrdroot.tree import Jagged


@pytest.mark.parametrize('name', list(stl.CPP_NAMES))
@pytest.mark.parametrize('order', ['<', '>'])
@pytest.mark.parametrize('offsets', [[0, 3, 3, 7], [7, 3, -2, 11], [0, 0, 0]])
def test_native_rows_keep_values_types_and_independent_storage(name, order, offsets):
    dtype = np.dtype(name, metadata={'column': 'native rows'}).newbyteorder(order)
    content = np.arange(24).astype(dtype)[::2]
    column = Jagged(content, offsets)
    expected = [rvec._made(np.asarray(row)) for row in column]
    actual = vectorrows.rows(column, rvec._made)
    assert actual is not None
    assert len(actual) == len(expected)
    for found, original in zip(actual, expected, strict=True):
        assert type(found) is type(original)
        assert np.asarray(found).tobytes() == np.asarray(original).tobytes()
        assert not np.shares_memory(found.data(), content)


@pytest.mark.parametrize('name', ['float32', 'float64'])
def test_native_float_rows_keep_signed_zero_infinities_and_nan_payloads(name):
    array = np.array([0., -0., np.inf, -np.inf, np.nan], dtype=name)
    column = Jagged(array, [0, 5])
    found = vectorrows.rows(column, rvec._made)
    assert found is not None
    assert found[0].data().tobytes() == rvec._made(array).data().tobytes()


def test_empty_batches_do_not_build_a_vector_specialization(monkeypatch):
    calls = []
    monkeypatch.setattr(rvec, 'RVec', lambda *args: calls.append(args))
    assert entrywise._entries(Jagged([], [0])) == []
    assert calls == []


def test_vector_rows_keep_a_replaced_result_maker(monkeypatch):
    calls = []
    def maker(array):
        calls.append(array.tolist())
        return tuple(array)
    monkeypatch.setattr(entrywise, '_made', maker)
    result = entrywise._entries(Jagged([1., 2., 3.], [0, 2, 3]))
    assert result == [(1., 2.), (3.,)]
    assert calls == [[1., 2.], [3.]]


@pytest.mark.parametrize('name', ['__init__', '_construct', 'assign'])
@pytest.mark.parametrize('wrapped', [False, True])
def test_custom_vector_constructors_keep_their_original_calls(monkeypatch, name, wrapped):
    calls = []
    original = getattr(stl._Vector, name)
    def callback(self, *args):
        calls.append(np.asarray(args[0]).tolist())
        return original(self, *args)
    if wrapped:
        callback = functools.wraps(original)(callback)
    monkeypatch.setattr(stl._Vector, name, callback)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert [list(v) for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [[1., 2.], [3.]]


def test_custom_jagged_indexing_keeps_its_call_order(monkeypatch):
    calls = []
    original = Jagged.__getitem__
    def callback(self, index):
        calls.append(index)
        return original(self, index)
    monkeypatch.setattr(Jagged, '__getitem__', callback)
    column = Jagged([1., 2., 3.], [0, 2, 3])
    assert [list(v) for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [0, 1, 2]


def test_a_subclass_iterator_keeps_the_general_preparation_path():
    calls = []
    class Rows(Jagged):
        def __iter__(self):
            calls.append('iter')
            yield np.array([7., 8.], dtype='f4')
    column = Rows([], [0])
    assert [list(v) for v in entrywise._entries(column)] == [[7., 8.]]
    assert calls == ['iter']


def test_class_creation_hooks_are_not_entered_by_fast_path_eligibility(monkeypatch):
    calls = []
    def callback(cls):
        calls.append(cls)
    monkeypatch.setattr(rvec._RVec, '__init_subclass__', classmethod(callback))
    monkeypatch.delitem(rvec.RVec._made, ('double',), raising=False)
    column = Jagged([1., 2., 3.], [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [list(v) for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert len(calls) == 1


def test_writing_a_prepared_row_never_changes_the_input_or_another_row():
    column = Jagged(np.array([1., 2., 3., 4.], dtype='f4'), [0, 2, 4])
    vectors = entrywise._entries(column)
    vectors[0][0] = 99
    assert list(vectors[1]) == [3., 4.]
    assert column.content.tolist() == [1., 2., 3., 4.]


@pytest.mark.parametrize('name', ['asarray', 'zeros'])
def test_numpy_constructor_callbacks_keep_their_scalar_call_count(monkeypatch, name):
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    calls = []
    original = getattr(np, name)
    def callback(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(np, name, callback)
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    result = entrywise._entries(column)
    assert [v.data().tolist() for v in result] == [[1., 2.], [3.]]
    assert len(calls) == (4 if name == 'asarray' else 2)


def test_a_changed_storage_precision_keeps_the_original_cast(monkeypatch):
    monkeypatch.setattr(rvec.RVecF, 'dtype', np.dtype('float64'))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    vectors = entrywise._entries(column)
    assert [v.data().dtype for v in vectors] == [np.dtype('float64')] * 2
    assert [v.data().tolist() for v in vectors] == [[1., 2.], [3.]]


def test_storage_descriptors_are_only_entered_by_the_original_constructor(monkeypatch):
    calls = []
    def dtype(self):
        calls.append('dtype')
        return np.dtype('float32')
    monkeypatch.setattr(rvec.RVecF, 'dtype', property(dtype))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == ['dtype'] * 4


def test_custom_specialization_objects_are_not_entered_during_eligibility(monkeypatch):
    calls = []
    class Factory:
        def __call__(self, values):
            calls.append(tuple(values))
            return tuple(values)
    monkeypatch.setitem(rvec.RVec._made, ('float',), Factory())
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert entrywise._entries(column) == [(1., 2.), (3.,)]
    assert calls == [(1., 2.), (3.,)]


def test_custom_specialization_builders_keep_their_original_cache_misses(monkeypatch):
    calls = []
    def builder(kind):
        calls.append(kind)
        return rvec._rvec_class(kind)
    monkeypatch.setattr(rvec, 'RVec', stl._Template('RVec', 1, builder))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [np.dtype('float32')]


def test_custom_cpp_spelling_lookups_keep_their_scalar_call_count(monkeypatch):
    calls = []
    class Names(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)
    monkeypatch.setattr(stl, 'CPP_NAMES', Names(stl.CPP_NAMES))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == ['float32'] * 2


def test_custom_dtype_lookups_keep_their_scalar_call_count(monkeypatch):
    calls = []
    class Names(dict):
        def get(self, key):
            calls.append(key)
            return super().get(key)
    monkeypatch.setattr(stl, '_DTYPE_NAMES', Names(stl._DTYPE_NAMES))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [np.dtype('float32').num] * 2


def test_vector_classes_with_custom_metadata_keep_ordinary_construction(monkeypatch):
    monkeypatch.setattr(rvec.RVecF, 'custom_metadata', 'extra', raising=False)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]


def test_callable_result_makers_keep_their_per_row_calls(monkeypatch):
    calls = []
    original = rvec._made
    class Maker:
        def __call__(self, row):
            calls.append(row.tolist())
            return original(row)
    maker = Maker()
    monkeypatch.setattr(entrywise, '_made', maker)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, maker) is None
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [[1., 2.], [3.]]


def test_subclassed_offset_arrays_keep_their_original_accesses():
    calls = []
    class Offsets(np.ndarray):
        def __getitem__(self, index):
            calls.append(index)
            return super().__getitem__(index)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    column.offsets = column.offsets.view(Offsets)
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [0, 1, 1, 2]


def test_custom_template_caches_keep_their_original_accesses(monkeypatch):
    calls = []
    class Cache(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)
    monkeypatch.setattr(rvec.RVec, '_made', Cache(rvec.RVec._made))
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [('float',)] * 2


def test_custom_template_objects_keep_their_original_lookups(monkeypatch):
    calls = []
    original = rvec.RVec
    class Factory:
        def __getitem__(self, key):
            calls.append(key)
            return original[key]
    monkeypatch.setattr(rvec, 'RVec', Factory())
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert calls == []
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == [np.dtype('float32')] * 2


def test_custom_template_dictionaries_are_not_entered_by_the_native_guard(monkeypatch):
    calls = []
    class Template(stl._Template):
        @property
        def __dict__(self):
            calls.append(1)
            raise AssertionError('The native guard entered a custom dictionary descriptor.')
    template = Template('RVec', 1, rvec._rvec_class)
    monkeypatch.setattr(rvec, 'RVec', template)
    monkeypatch.setattr(stl, '_Template', Template)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
    assert calls == []


def test_noninteger_arities_keep_their_original_conversion(monkeypatch):
    monkeypatch.setattr(rvec.RVec, '_arity', True)
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    assert vectorrows.rows(column, rvec._made) is None
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]


def test_foreign_numpy_metadata_is_not_compared_during_native_detection(monkeypatch):
    class Foreign:
        def __eq__(self, other):
            raise AssertionError('The native guard compared foreign NumPy metadata.')
    column = Jagged(np.array([1., 2., 3.], dtype='f4'), [0, 2, 3])
    monkeypatch.setattr(np.asarray, '__module__', Foreign())
    assert vectorrows.rows(column, rvec._made) is None
    assert [v.data().tolist() for v in entrywise._entries(column)] == [[1., 2.], [3.]]
