"""The one-collection path keeps the batch evaluator's order and independent storage."""
import numpy as np
import pytest

from xrdroot.pyroot.rdf import rvec


@pytest.mark.parametrize('n',range(6))
@pytest.mark.parametrize('k',range(7))
def test_scalar_combinations_keep_batch_order_and_types(n,k):
    first=rvec.RVecF(np.arange(n,dtype=np.float32))
    original=rvec._OneEntry(rvec._batched.Combinations)
    fast=rvec.VecOps.Combinations
    actual,expected=fast(first,k),original(first,k)
    assert len(actual)==len(expected)
    for a,b in zip(actual,expected,strict=True):
        assert type(a) is type(b)
        assert np.array_equal(np.asarray(a),np.asarray(b))
    if actual and len(actual[0]):
        actual[0][0]=999
        assert list(fast(first,k)[0])==list(expected[0])

@pytest.mark.parametrize('first',[[1,2],(1,2),np.array([1,2]),rvec.RVecD([1,2])])
def test_scalar_collection_spellings_keep_the_same_indices(first):
    assert [list(v) for v in rvec.VecOps.Combinations(first,np.int64(2))]==[[0],[1]]

def test_cartesian_products_keep_the_general_evaluator():
    a,b=rvec.VecOps.Combinations(rvec.RVecF([1,2]),rvec.RVecF([3,4,5]))
    assert list(a)==[0,0,0,1,1,1]
    assert list(b)==[0,1,2,0,1,2]

def test_bad_arities_keep_the_original_error():
    with pytest.raises(TypeError):
        rvec.VecOps.Combinations(rvec.RVecF([1]))

def test_scalar_first_arguments_keep_the_original_error():
    with pytest.raises(TypeError):
        rvec.VecOps.Combinations(1,2)

def test_negative_choices_keep_the_original_error():
    with pytest.raises(ValueError):
        rvec.VecOps.Combinations(rvec.RVecF([1]),-1)


def test_custom_integer_coercions_keep_their_original_order_and_call_count():
    calls = []
    class Vector(rvec.RVecF):
        def data(self):
            calls.append('data')
            return super().data()
    class Count(int):
        def __int__(self):
            calls.append('int')
            return 2
    result = rvec.VecOps.Combinations(Vector([1., 2., 3.]), Count(2))
    assert [list(row) for row in result] == [[0, 0, 1], [1, 2, 2]]
    assert calls == ['data', 'int']


@pytest.mark.parametrize('side', ['first', 'second'])
def test_foreign_class_properties_are_not_inspected_by_fast_path_eligibility(side):
    calls = []
    class Probe:
        @property
        def __class__(self):
            calls.append('class')
            return int
    results = []
    for function in (rvec._OneEntry(rvec._batched.Combinations), rvec.VecOps.Combinations):
        calls.clear()
        args = (Probe(), 2) if side == 'first' else ([1., 2.], Probe())
        with pytest.raises(TypeError) as error:
            function(*args)
        results.append((list(calls), str(error.value)))
    assert results[0] == results[1]


def test_replacing_a_scalar_function_keeps_the_original_callback():
    function = type(rvec.VecOps.Combinations)(rvec._batched.Combinations)
    function._function = lambda *args: 123
    assert function(rvec.RVecF([1., 2.]), 2) == 123


def test_replacing_the_batch_function_does_not_opt_callbacks_into_the_shortcut(monkeypatch):
    def replacement(first, second):
        assert len(first) == 1
        assert second == 2
        return 123

    monkeypatch.setattr(rvec._batched, 'Combinations', replacement)
    function = rvec._one(replacement)
    assert function(rvec.RVecF([1., 2.]), 2) == 123


@pytest.mark.parametrize('side', ['first', 'second'])
def test_foreign_metaclass_equality_is_not_entered_by_fast_path_eligibility(side):
    class Meta(type):
        def __eq__(self, other):
            raise AssertionError('Fast eligibility compared an arbitrary metaclass.')
    class Probe(metaclass=Meta):
        pass
    results = []
    for function in (rvec._OneEntry(rvec._batched.Combinations), rvec.VecOps.Combinations):
        args = (Probe(), 2) if side == 'first' else ([1., 2.], Probe())
        with pytest.raises(TypeError) as error:
            function(*args)
        results.append(str(error.value))
    assert results[0] == results[1]
