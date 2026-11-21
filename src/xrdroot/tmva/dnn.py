"""TMVA's DNN CPU backend: the matrices it works on, and the loss functions over them.

TMVA's deep-learning unit tests check the CPU backend - ``TMVA::DNN::TCpu<T>`` -
against a reference implementation written in the test itself, so what is
checked is the arithmetic of the formulas below, to a relative error of 1e-3.
Each formula is TMVA's own, in ``TMVA/DNN/Architectures/Cpu/LossFunctions.hxx``:
the weights are read per row of the matrix, and the normalisation of the mean
squared and cross-entropy losses is over every element, that of the softmax
cross entropy over every row.

A ``Matrix`` is a NumPy array of two dimensions holding TMVA's column-major
matrix - what ``TCpuMatrix<T>`` is, its ``(i, j)`` an element of, and what a
``TMatrixT<T>`` is made from and makes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["LOSSES", "LOSS_GRADIENTS", "Matrix", "matrix_type"]

#: The element types a matrix can be, by their C++ names.
DTYPES = {"float": np.float32, "double": np.float64}


class Matrix:
    """``TCpuMatrix<T>``: a dense matrix the CPU backend works on.

    Made of zeros of a shape - ``(nrows, ncols)`` - or of what it copies: another
    matrix, a ``TMatrixT``, or anything NumPy reads as an array.
    """

    dtype: Any = np.float64

    def __init__(self, *args: Any) -> None:
        if len(args) == 1:
            self.values = np.array(np.asarray(_elements(args[0])), dtype=self.dtype)
        else:
            self.values = np.zeros((int(args[0]), int(args[1])), dtype=self.dtype)

    def __class_getitem__(cls, kind: Any) -> type[Matrix]:
        return matrix_type(kind)

    def GetNrows(self) -> int:
        return int(self.values.shape[0])

    def GetNcols(self) -> int:
        return int(self.values.shape[1])

    def GetNoElements(self) -> int:
        return int(self.values.size)

    def GetSize(self) -> int:
        return int(self.values.size)

    def __call__(self, i: Any, j: Any) -> float:
        return float(self.values[int(i), int(j)])

    def __setcall__(self, i: Any, j: Any, value: Any) -> None:
        self.values[int(i), int(j)] = value

    def GetMatrixArray(self) -> Any:
        """TMVA's raw pointer: the elements in memory order, column by column."""
        return self.values.ravel(order="F")

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        made = np.asarray(self.values, dtype=dtype)
        return made.copy() if copy else made

    def Print(self) -> None:
        """``Print``: the matrix's shape and its elements, as TMVA dumps them."""
        print(f"Matrix {self.GetNrows()}x{self.GetNcols()}")
        for row in self.values:
            print("  " + " ".join(f"{value:g}" for value in row))

    def __repr__(self) -> str:
        return f"<TCpuMatrix {self.GetNrows()}x{self.GetNcols()}>"


def _elements(given: Any) -> Any:
    """What a matrix is copied from: another matrix's elements, a ``TMatrixT``'s, an array."""
    values = getattr(given, "values", None)
    if values is not None:
        return values
    return np.asarray(given, dtype=np.float64)


def _array(given: Any) -> Any:
    """A matrix's elements as NumPy takes them, in the precision the matrix holds - TMVA's
    ``AFloat``, single for a float architecture and double for a double one."""
    return np.asarray(given, dtype=getattr(given, "dtype", np.float64))


def matrix_type(kind: Any) -> type[Matrix]:
    """``TCpuMatrix<T>``: the matrix class of elements of C++ type ``kind``."""
    name = str(kind).replace("Float_t", "float").replace("Double_t", "double")
    return type(f"TCpuMatrix<{name}>", (Matrix,), {"dtype": DTYPES.get(name, np.float64)})


def _rows(y: Any) -> int:
    return int(y.shape[0])


def _weights(weights: Any, rows: int) -> Any:
    """The weight of each row: TMVA reads the matrix's memory ``rows`` elements deep, which
    in its column-major order is the first column."""
    return np.asarray(_elements(weights), dtype=np.float64).ravel(order="F")[:rows]


def mean_squared_error(y: Any, x: Any, w: Any) -> float:
    """``TCpu<A>::MeanSquaredError``: the weighted mean of the squared differences."""
    y, x = _array(y), _array(x)
    norm = 1.0 / float(y.shape[0] * y.shape[1])
    return norm * float(np.sum(_weights(w, _rows(y))[:, None] * (y - x) ** 2))


def mean_squared_error_gradients(dy: Any, y: Any, x: Any, w: Any) -> None:
    y, x = _array(y), _array(x)
    weight = _weights(w, _rows(y))
    norm = 1.0 / float(y.shape[0] * y.shape[1])
    dy.values[:, :] = -2.0 * norm * (y - x) * weight[:, None]


def cross_entropy(y: Any, x: Any, w: Any) -> float:
    """``TCpu<A>::CrossEntropy``: the sigmoid's cross entropy, its logarithm taken as TMVA
    takes it - ``log(1 + exp(-x))``, which stays finite for an ``x`` far from zero."""
    y, x = _array(y), _array(x)
    lr = np.where(x < -75.0, -x, np.where(x > 75.0, np.exp(-x), np.log1p(np.exp(-x))))
    norm = 1.0 / float(y.shape[0] * y.shape[1])
    terms = y * lr + (1.0 - y) * (x + lr)
    return norm * float(np.sum(_weights(w, _rows(y))[:, None] * terms))


def cross_entropy_gradients(dy: Any, y: Any, x: Any, w: Any) -> None:
    y, x = _array(y), _array(x)
    sigmoid = 1.0 / (1.0 + np.exp(-x))
    norm = 1.0 / float(y.shape[0] * y.shape[1])
    dy.values[:, :] = norm * (sigmoid - y) * _weights(w, _rows(y))[:, None]


def _softmax(x: Any) -> Any:
    """``exp(x)`` over the sum of its row, each row's largest element taken out first, as
    TMVA's ratio ``exp(x) / sum(exp(x))`` is the same and cannot overflow."""
    shifted = np.exp(x - x.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def softmax_cross_entropy(y: Any, x: Any, w: Any) -> float:
    """``TCpu<A>::SoftmaxCrossEntropy``: the cross entropy of a softmax over each row."""
    y, x = _array(y), _array(x)
    norm = 1.0 / float(_rows(y))
    per_row = -np.sum(y * np.log(_softmax(x)), axis=1)
    return norm * float(np.sum(per_row * _weights(w, _rows(y))))


def softmax_cross_entropy_gradients(dy: Any, y: Any, x: Any, w: Any) -> None:
    y, x = _array(y), _array(x)
    soft = _softmax(x)
    norm = 1.0 / float(_rows(y))
    dy.values[:, :] = norm * (soft * y.sum(axis=1, keepdims=True) - y) * _weights(
        w, _rows(y))[:, None]


#: ``TMVA::DNN::ELossFunction``'s members without their ``k``, as the function each names.
LOSSES: dict[str, Callable[[Any, Any, Any], float]] = {
    "CrossEntropy": cross_entropy,
    "MeanSquaredError": mean_squared_error,
    "SoftmaxCrossEntropy": softmax_cross_entropy,
}

#: Each loss function's gradient, written into the matrix it is given.
LOSS_GRADIENTS: dict[str, Callable[[Any, Any, Any, Any], None]] = {
    "CrossEntropy": cross_entropy_gradients,
    "MeanSquaredError": mean_squared_error_gradients,
    "SoftmaxCrossEntropy": softmax_cross_entropy_gradients,
}
