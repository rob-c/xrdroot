"""Scalar reads retain NumPy's values and foreign indexing callbacks."""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf.rvec import RVec


def _ordinary_scalar(vector, index):
    return vector.data()[vector._checked(index)].item()


def _observed(storage, calls, checked):
    class Observed(stl._Vector):
        def data(self):
            calls.append("data")
            return storage

        def _checked(self, index):
            calls.append(("checked", index))
            return checked(index)

    return Observed()


def _same_number(actual, expected, dtype):
    assert type(actual) is type(expected)
    assert np.asarray(actual, dtype=dtype).tobytes() == np.asarray(expected, dtype=dtype).tobytes()


@pytest.mark.parametrize(
    "kind",
    ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64",
     "uint64", "float32", "float64"],
)
@pytest.mark.parametrize("size", [1, 4, 4096])
def test_numeric_scalar_reads_match_the_original_full_storage_and_return_type(kind, size):
    dtype = np.dtype(kind)
    values = np.arange(size).astype(dtype)
    if dtype.kind == "u":
        values[-1] = np.iinfo(dtype).max
    for factory in (stl.std.vector, RVec):
        vector = factory[dtype](values)
        vector.reserve(size + 7)
        before = vector._data.tobytes()
        for index in (0, size - 1, -1, -size):
            _same_number(vector[index], _ordinary_scalar(vector, index), dtype)
        assert vector._data.tobytes() == before
        assert vector._size == size


@pytest.mark.parametrize("bits,word,dtype", [
    (32, "uint32", "float32"), (64, "uint64", "float64"),
])
@pytest.mark.parametrize("step", [1, 2, -1])
def test_scalar_reads_preserve_signed_zero_nan_payload_and_subnormal_bits(bits, word, dtype, step):
    raw = [0x80000000, 0x3F800000, 0x7FC00042, 0x40000000, 1, 0] if bits == 32 else [
        0x8000000000000000, 0x3FF0000000000000, 0x7FF8000000000042, 0x4000000000000000, 1, 0,
    ]
    storage = np.asarray(raw, dtype=word).view(dtype)[::step]
    calls = []
    vector = _observed(storage, calls, int)
    before = storage.tobytes()
    for index in range(storage.size):
        expected = _ordinary_scalar(vector, index)
        calls.clear()
        _same_number(vector[index], expected, storage.dtype)
        assert calls == ["data", ("checked", index)]
    assert storage.tobytes() == before


@pytest.mark.parametrize(
    "dtype",
    [np.dtype("float16"), np.dtype(">f8"), np.dtype(">u8"), np.dtype(np.longdouble)],
)
def test_non_native_byte_order_and_long_float_reads_keep_original_values_and_types(dtype):
    storage = np.asarray([0, 1, 2], dtype=dtype)
    calls = []
    vector = _observed(storage, calls, int)
    for index in (0, 1, -1):
        expected = _ordinary_scalar(vector, index)
        actual = vector[index]
        assert type(actual) is type(expected)
        assert actual == expected
        assert bool(np.signbit(actual)) == bool(np.signbit(expected))


def test_array_subclasses_keep_getitem_and_do_not_acquire_an_item_callback():
    calls = []

    class Array(np.ndarray):
        def __getitem__(self, index):
            calls.append(("getitem", index))
            return super().__getitem__(index)

        def item(self, *args):
            calls.append(("item", args))
            return super().item(*args)

    storage = np.asarray([1.0, -0.0]).view(Array)
    vector = _observed(storage, calls, int)
    expected = _ordinary_scalar(vector, -1)
    original_calls = calls[:]
    calls.clear()
    _same_number(vector[-1], expected, np.dtype("float64"))
    assert calls == original_calls == ["data", ("checked", -1), ("getitem", -1)]


def test_foreign_storage_retains_its_index_and_item_callbacks_after_the_index_check():
    calls = []
    marker = object()

    class Value:
        def item(self):
            calls.append("item")
            return marker

    class Storage:
        def __getitem__(self, index):
            calls.append(("getitem", index))
            return Value()

    vector = _observed(Storage(), calls, lambda index: index)
    assert _ordinary_scalar(vector, 2) is marker
    original_calls = calls[:]
    calls.clear()
    assert vector[2] is marker
    assert calls == original_calls == ["data", ("checked", 2), ("getitem", 2), "item"]


def test_object_arrays_retain_the_stored_objects_item_callback():
    calls = []

    class Item:
        def item(self):
            calls.append("stored item")
            return 37

    storage = np.asarray([Item()], dtype=object)
    vector = _observed(storage, calls, int)
    assert _ordinary_scalar(vector, 0) == 37
    original_calls = calls[:]
    calls.clear()
    assert vector[0] == 37
    assert calls == original_calls == ["data", ("checked", 0), "stored item"]


@pytest.mark.parametrize("storage,index", [
    (np.asarray([[1.0], [2.0]]), 0), (np.asarray([1.0]), 2),
    (np.asarray([1.0]), -2), (np.asarray([[1.0, 2.0]]), 0),
    (np.asarray(["abc"]), 0),
])
def test_non_scalar_or_out_of_bounds_storage_keeps_the_original_result_or_exact_error(
    storage, index,
):
    calls = []
    vector = _observed(storage, calls, lambda given: given)
    try:
        expected = _ordinary_scalar(vector, index)
    except Exception as error:
        original_calls = calls[:]
        calls.clear()
        with pytest.raises(type(error)) as actual:
            vector[index]
        assert type(actual.value) is type(error)
        assert str(actual.value) == str(error)
    else:
        original_calls = calls[:]
        calls.clear()
        actual = vector[index]
        assert type(actual) is type(expected)
        assert actual == expected
    assert calls == original_calls == ["data", ("checked", index)]


def test_a_foreign_checked_index_keeps_the_original_index_protocol():
    calls = []

    class Index:
        def __index__(self):
            calls.append("index")
            return 0

    vector = _observed(np.asarray([7.0]), calls, lambda given: Index())
    expected = _ordinary_scalar(vector, 4)
    original_calls = calls[:]
    calls.clear()
    assert vector[4] == expected == 7.0
    assert calls == original_calls == ["data", ("checked", 4), "index"]


def test_index_conversion_may_mutate_the_previously_obtained_storage_shape():
    storage = np.asarray([1.0, 2.0])
    calls = []

    def checked(index):
        calls.append("reshape")
        storage.shape = (2, 1)
        return int(index)

    vector = _observed(storage, calls, checked)
    expected = _ordinary_scalar(vector, 1)
    original_calls = calls[:]
    storage.shape = (2,)
    calls.clear()
    assert vector[1] == expected == 2.0
    assert storage.shape == (2, 1)
    assert calls == original_calls == ["data", ("checked", 1), "reshape"]


def test_slice_reads_keep_the_original_data_slice_and_copy_callbacks():
    calls = []
    marker = object()

    class Slice:
        def copy(self):
            calls.append("copy")
            return marker

    class Storage:
        def __getitem__(self, index):
            calls.append(("slice", index.start, index.stop, index.step))
            return Slice()

    vector = _observed(Storage(), calls, lambda index: index)
    assert vector[1:4:2] is marker
    assert calls == ["data", ("slice", 1, 4, 2), "copy"]


@pytest.mark.parametrize("checked", [np.int64(0), False])
def test_non_builtin_checked_indices_keep_the_original_result_or_exact_error(checked):
    calls = []
    vector = _observed(np.asarray([1.0, 2.0]), calls, lambda given: checked)
    try:
        expected = _ordinary_scalar(vector, 0)
    except Exception as error:
        original_calls = calls[:]
        calls.clear()
        with pytest.raises(type(error)) as actual:
            vector[0]
        assert type(actual.value) is type(error)
        assert str(actual.value) == str(error)
    else:
        original_calls = calls[:]
        calls.clear()
        actual = vector[0]
        assert type(actual) is type(expected)
        assert actual == expected
    assert calls == original_calls == ["data", ("checked", 0)]
