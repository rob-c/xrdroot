"""Numeric dtype spelling stays canonical without recomputing NumPy's descriptive name."""
import numpy as np
import pytest

from xrdroot.pyroot import stl
from xrdroot.pyroot.rdf.rvec import RVec


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
