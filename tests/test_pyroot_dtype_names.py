"""Numeric dtype spelling stays canonical without recomputing NumPy's descriptive name."""
import numpy as np
import pytest

from xrdroot.cppstd.containers import _canonical
from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf.rvec import RVec, _rvec_class


@pytest.mark.parametrize('name', list(stl.CPP_NAMES))
@pytest.mark.parametrize('order', ['<', '>'])
@pytest.mark.parametrize('metadata', [None, {'description': 'event column'}])
def test_numeric_dtypes_keep_their_cpp_name_and_template_identity(name, order, metadata):
    dtype = np.dtype(name, metadata=metadata) if metadata else np.dtype(name)
    dtype = dtype.newbyteorder(order)
    assert stl.cpp_name(dtype) == stl.CPP_NAMES[name]
    assert stl.std.vector[dtype] is stl.std.vector[np.dtype(name)]
    assert RVec[dtype] is RVec[np.dtype(name)]


@pytest.mark.parametrize('name', ['float16', 'complex64', 'datetime64[ns]', 'S7'])
def test_unknown_numeric_and_non_numeric_dtypes_keep_their_original_refusal(name):
    dtype = np.dtype(name)
    with pytest.raises(KeyError) as error:
        stl.cpp_name(dtype)
    assert error.value.args == (dtype.name,)


def test_dtype_names_do_not_cache_the_mutable_cpp_spelling_table(monkeypatch):
    dtype = np.dtype('float64')
    assert stl.cpp_name(dtype) == 'double'
    monkeypatch.setitem(stl.CPP_NAMES, 'float64', 'changed double')
    assert stl.cpp_name(dtype) == 'changed double'


def test_builtin_alias_dtypes_keep_their_numpy_spelling():
    for name in ('longlong', 'ulonglong', 'intp', 'uintp'):
        dtype = np.dtype(name)
        assert stl.cpp_name(dtype) == stl.CPP_NAMES[dtype.name]


@pytest.mark.parametrize('build', [stl._vector_class, _rvec_class], ids=['std', 'rvec'])
@pytest.mark.parametrize('order', ['<', '>'])
@pytest.mark.parametrize('metadata', [None, {'description': 'event column'}])
@pytest.mark.parametrize('spelling', ['uint64', 'ulonglong'])
def test_a_foreign_dtype_built_first_does_not_become_every_later_classes_dtype(
        build, order, metadata, spelling):
    # A big-endian or annotated dtype's class, built first and cached under "unsigned long",
    # once gave every later RVec['unsigned long'] in the worker that byte order.
    template = stl._Template('vector', 1, build)
    foreign = np.dtype(spelling, metadata=metadata) if metadata else np.dtype(spelling)
    first = template[foreign.newbyteorder(order)]
    assert template['unsigned long'] is first
    assert first.dtype == np.dtype('uint64') and first.dtype.isnative
    assert first.dtype.metadata is None
    words = np.array([0, 2**64 - 1, 2**63, 1], dtype='uint64')
    value = first(words)
    value.assign(value)
    assert value.data().view('uint64').tobytes() == words.tobytes()


def test_only_known_number_dtypes_are_made_canonical():
    given = np.dtype('float16').newbyteorder('>')
    assert _canonical(given) is given
    assert _canonical(stl.string) is stl.string


def test_the_identity_test_then_an_unsigned_long_rvec_self_assignment_in_one_process(
        monkeypatch):
    # The polluter, then the victim, as one xdist worker once ran them: with empty class
    # caches, as in a worker where nothing had yet asked for an unsigned long vector.
    monkeypatch.setattr(RVec, '_made', {})
    monkeypatch.setattr(stl.std.vector, '_made', {})
    for metadata in (None, {'description': 'event column'}):
        test_numeric_dtypes_keep_their_cpp_name_and_template_identity('uint64', '>', metadata)
    words = np.array([0, 2**64 - 1, 2**63, 1], dtype='uint64')
    value = RVec['unsigned long'](words)
    previous = value._data
    value.assign(value)
    assert value.data().view('uint64').tobytes() == previous.view('uint64').tobytes()
    assert previous.tobytes() == words.tobytes()
