"""ROOT's RVec condition overload selects nonzero elements of numeric vectors."""
import numpy as np
import pytest

from xrdroot import Jagged, create
from xrdroot import pyroot as ROOT


@pytest.mark.parametrize('kind', ['bool', 'int', 'unsigned int', 'long', 'float', 'double'])
@pytest.mark.parametrize('conditions,expected', [
    ([1, 0, 1], [11., 47.]), ([0, 0, 0], []), ([1, 1, 1], [11., 23., 47.]),
])
def test_numeric_rvec_conditions_select_nonzero_elements_in_order(kind, conditions, expected):
    # ROOT 6.40.04's RVecN::operator[](RVecN<V>) tests each condition as a boolean.
    values = ROOT.RVecF([11., 23., 47.])
    selected = values[ROOT.RVec[kind](conditions)]
    assert type(selected) is ROOT.RVecF
    assert list(selected) == expected


def test_negative_and_nonunit_integer_conditions_are_true():
    assert list(ROOT.RVecD([11., 23., 47.])[ROOT.RVecI([-1, 0, 2])]) == [11., 47.]


@pytest.mark.parametrize('size', [0, 2, 4])
def test_condition_vectors_must_have_the_same_size_as_the_values(size):
    conditions = ROOT.RVecI([1] * size)
    with pytest.raises(TypeError, match=rf'Cannot index RVecN of size 3.*size \({size}\)\.'):
        ROOT.RVecF([11., 23., 47.])[conditions]


def test_empty_vectors_accept_empty_conditions():
    selected = ROOT.RVecF()[ROOT.RVecI()]
    assert type(selected) is ROOT.RVecF and len(selected) == 0


def test_selected_storage_is_independent_of_both_inputs():
    values, conditions = ROOT.RVecF([11., 23., 47.]), ROOT.RVecI([1, 0, 1])
    selected = values[conditions]
    values[0], conditions[0], selected[1] = 100., 0, 200.
    assert list(values) == [100., 23., 47.]
    assert list(conditions) == [0, 0, 1]
    assert list(selected) == [11., 200.]


@pytest.mark.parametrize('indices', [[0, 2], np.array([0, 2])])
def test_python_and_numpy_indices_keep_their_positional_selection(indices):
    assert list(ROOT.RVecF([11., 23., 47.])[indices]) == [11., 47.]


def test_dataframe_integer_masks_reach_declared_functions_as_conditions(tmp_path):
    ROOT.gInterpreter.Declare('''
        float condition_sum(RVecF values) { return ROOT::VecOps::Sum(values); }
    ''')
    values = Jagged(np.array([11., 23., 47., 2., 3.], dtype='f4'), np.array([0, 3, 5, 5]))
    path = tmp_path / 'conditions.root'
    with create(path) as output:
        output['tree'] = {'values': values}
    frame = ROOT.RDataFrame('tree', str(path)).Define('conditions', 'values > 20')
    with frame._inner:
        assert frame.GetColumnType('conditions') == 'ROOT::VecOps::RVec<int>'
        result = frame.Define('selected_sum', 'condition_sum(values[conditions])')
        assert result.AsNumpy(['selected_sum'])['selected_sum'].tolist() == [70., 0., 0.]
