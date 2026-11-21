"""ROOT's element actions: ``TElementActionT`` and ``TElementPosActionT``.

A matrix or vector is changed element by element by a class the caller
writes, ``class FillMatrix : public TElementPosActionD``, whose ``Operation``
the matrix calls for each of its elements with a reference to the element -
and, for the position-taking kind, with the element's row and column in
``fI`` and ``fJ``. That is how a ROOT macro fills a matrix with an arbitrary
function of the position without the matrix knowing what the function is, so
the base class is what the macro's class derives from and what ``Apply``
walks. ROOT declares the two as templates over the element's type and names
the four typedefs ``TElementActionD``, ``TElementActionF``,
``TElementPosActionD`` and ``TElementPosActionF``; all four are the one Python
class here, since Python has the one kind of number.

The action is handed the element as a reference - the macro writes
``void Operation(Double_t &element) const { element = ...; }`` - so what
``Apply`` passes into a matrix's own array is a
:class:`~xrdroot.cppstd.cells.ItemRef`.
"""

from __future__ import annotations

from typing import Any

from ...cppstd.cells import Cell, ItemRef
from ...errors import UnsupportedFeatureError

__all__ = [
    "TElementActionT",
    "TElementPosActionT",
    "TElementActionD",
    "TElementActionF",
    "TElementPosActionD",
    "TElementPosActionF",
]


class TElementActionT:
    """``TElementActionT<Element>``: an action on every element, wherever it is.

    ``Operation`` is pure virtual in C++, so a macro always writes it; one that
    is called without it named the base and gave no action, which ROOT cannot
    compile either.
    """

    #: The element type ROOT's ``Element`` parameter stands for: ``Double_t`` here.
    fElementType = "Double_t"

    def __class_getitem__(cls, kind: Any) -> type:
        """``TElementActionT<double>``: these are one class, of the one number Python has."""
        return cls

    def Operation(self, element: Any) -> None:
        raise UnsupportedFeatureError(
            f"{type(self).__name__} was given to Apply with no Operation, which is the method "
            "ROOT calls for each element; give the class the Operation it should run."
        )


class TElementPosActionT(TElementActionT):
    """``TElementPosActionT<Element>``: an action that knows where the element is.

    ``fI`` and ``fJ`` are the row and column ``Apply`` has reached, which a
    macro's ``Operation`` reads; a vector's ``Apply`` sets ``fI`` alone.
    """

    #: The row of the element being operated on, as ``Apply`` walks the matrix.
    fI = 0
    #: The column of it.
    fJ = 0


#: ``TElementActionT<Double_t>``.
TElementActionD = TElementActionT
#: ``TElementActionT<Float_t>``.
TElementActionF = TElementActionT
#: ``TElementPosActionT<Double_t>``.
TElementPosActionD = TElementPosActionT
#: ``TElementPosActionT<Float_t>``.
TElementPosActionF = TElementPosActionT


def _positioned(action: Any, row: int, col: int | None = None) -> None:
    """Tell a position-taking action where it is, as ``Apply`` does before each call."""
    if isinstance(action, TElementPosActionT):
        action.fI = row
        if col is not None:
            action.fJ = col


def applied(matrix: Any, action: Any) -> None:
    """``TMatrixTBase::Apply``, ``TMatrixT``'s: every element of ``matrix`` in row order.

    ``matrix`` is walked by ``GetRowLwb()``'s count, so ``fI`` and ``fJ`` are the C++
    indices the macro computes with.
    """
    values = matrix.values
    for row in range(matrix.GetNrows()):
        for col in range(matrix.GetNcols()):
            _positioned(action, row + matrix.GetRowLwb(), col + matrix.GetColLwb())
            action.Operation(ItemRef(values, (row, col)))


def applied_symmetric(matrix: Any, action: Any) -> None:
    """``TMatrixTSym::Apply``: the upper triangle, each result mirrored below the diagonal.

    ROOT hands the action a fresh ``0`` here rather than the element's own value, since the
    element it will be written to is filled from the action's answer, and it fills both
    halves with that answer.
    """
    values = matrix.values
    for row in range(matrix.GetNrows()):
        for col in range(row, matrix.GetNcols()):
            _positioned(action, row + matrix.GetRowLwb(), col + matrix.GetColLwb())
            element = Cell(0.0, "double")
            action.Operation(element)
            values[row, col] = values[col, row] = element.value


def applied_to_vector(values: Any, action: Any, rowlwb: int) -> None:
    """``TVectorT::Apply``: every element in turn, ``fI`` its index for the positioned kind."""
    for row in range(len(values)):
        _positioned(action, row + rowlwb)
        action.Operation(ItemRef(values, row))
