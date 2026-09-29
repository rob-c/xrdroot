"""``TMVA::Experimental``'s tensors: made, read, written, reshaped, sliced, printed, filled."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT

Experimental = ROOT.TMVA.Experimental
RTensor, MemoryLayout = Experimental.RTensor, Experimental.MemoryLayout


class Buffer:
    """A macro's buffer, its NumPy array under ``name`` as pyroot's arrays and vectors keep it."""

    def __init__(self, name: str, values: list[float]) -> None:
        setattr(self, name, np.asarray(values, dtype=np.float32))


class Listed:
    """A container whose ``data`` is not an array: it is copied, element by element."""

    data = "not an array"

    def __iter__(self):
        return iter([1, 2, 3, 4])


def test_a_tensor_of_a_shape_is_zeros_of_its_element_type():
    made = RTensor["double"]([2, 3])
    assert made.GetShape() == [2, 3] and made.GetSize() == 6
    assert made.GetData().dtype == np.float64 and made(1, 2) == 0.0
    assert RTensor["Float_t"]([1]).dtype == np.float32
    assert RTensor["Double_t"]([1]).dtype == np.float64
    assert RTensor["std::complex"]([1]).dtype == np.float32


@pytest.mark.parametrize("name", ["_array", "_values", "data"])
def test_a_tensor_over_a_buffer_shares_its_memory(name):
    buffer = Buffer(name, [1, 2, 3, 4, 5, 6])
    made = RTensor["float"](buffer, [2, 3])
    made.__setcall__(0, 1, 20.0)
    assert getattr(buffer, name)[1] == 20.0 and made(0, 1) == 20.0


def test_a_tensor_over_a_numpy_array_or_anything_iterable_takes_its_values():
    over = np.arange(6, dtype=np.float32)
    assert RTensor["float"](over, [3, 2])(2, 1) == 5.0
    assert RTensor["int"](Listed(), [2, 2])(1, 0) == 3


def test_a_column_major_tensor_counts_its_first_index_fastest():
    made = RTensor["float"](np.arange(6, dtype=np.float32), [2, 3], MemoryLayout.ColumnMajor)
    assert made(1, 0) == 1.0 and made(0, 1) == 2.0
    assert made.GetMemoryLayout() == MemoryLayout.ColumnMajor
    reshaped = made.Reshape([3, 2])
    assert reshaped(0, 1) == 3.0 and reshaped.GetMemoryLayout() == MemoryLayout.ColumnMajor
    zeros = RTensor["float"]([2, 2], 2)
    assert zeros.GetMemoryLayout() == MemoryLayout.ColumnMajor


def test_a_tensor_reshaped_squeezed_and_sliced_keeps_its_elements():
    made = RTensor["float"](np.arange(12, dtype=np.float32), [3, 1, 4])
    assert made.Reshape([4, 3])(1, 0) == 3.0
    assert made.Squeeze().GetShape() == [3, 4]
    assert RTensor["float"]([1, 1]).Squeeze().GetShape() == [1]
    sliced = made.Slice([[1, 2], [0, 1], [0, 4]])
    assert sliced.GetShape() == [4] and sliced[0] == 4.0


def test_a_tensor_prints_as_tmvas_operator_does():
    made = RTensor["float"](np.array([1.5, 2, 3, 4], dtype=np.float32), [2, 2])
    assert str(made) == "{ { 1.5, 2 } { 3, 4 } }"
    assert repr(made) == "<RTensor (2, 2) { { 1.5, 2 } { 3, 4 } }>"
    assert str(RTensor["bool"](np.array([True, False]), [2])) == "{ 1, 0 }"


def test_a_frame_as_a_tensor_has_an_event_per_row_in_either_layout():
    frame = ROOT.RDataFrame(4).Define("x", "(float)rdfentry_").Define("y", "x*2")
    whole = Experimental.AsTensor["float"](frame)
    assert whole.GetShape() == [4, 2] and whole(3, 1) == 6.0
    chosen = Experimental.AsTensor["double"](frame, ["y"], MemoryLayout.ColumnMajor)
    assert chosen.GetShape() == [4, 1] and chosen.dtype == np.float64
    assert chosen.GetMemoryLayout() == MemoryLayout.ColumnMajor
    assert type(Experimental.AsTensor(frame)).__name__ == "RTensor<float>"
