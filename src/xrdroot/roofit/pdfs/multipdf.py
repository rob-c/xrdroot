"""``RooMultiPdf``: one of several densities, chosen by a category's index.

The tutorial that needs it (``rf619_discrete_profiling``) fits a model whose
shape is one of two, then the other, and asks for a profile over which one it
was - so the density has to *be* the selected one, not a mixture: its value,
its normalisation and its parameters are the selected density's own, and the
index category is what says which.
"""

from __future__ import annotations

from typing import Any

from ..pdf import RooAbsPdf
from ..real import Context

__all__ = ["RooMultiPdf"]


class RooMultiPdf(RooAbsPdf):
    """``RooMultiPdf``: the density the index category selects, from a list of them."""

    def __init__(self, name: Any, title: Any, index: Any, pdfList: Any) -> None:
        super().__init__(name, title)
        from ..collections import as_list

        self._index = index
        self._pdfs = list(as_list(pdfList))
        if not self._pdfs:
            raise ValueError(
                "RooMultiPdf needs at least one density to choose between; "
                "an empty list has nothing to be the selected one."
            )
        # ROOT gives the index category one state per density, named after it and
        # numbered in the order given, so `pdfIndex.setIndex(i)` selects the i-th.
        for position, pdf in enumerate(self._pdfs):
            index.defineType(pdf.GetName(), position)

    # --- which one is selected ----------------------------------------------

    def getCurrentIndex(self) -> int:
        """``getCurrentIndex``: the index category's state, as ROOT's ``int`` cast gives it."""
        return int(self._index.getIndex())

    def setCurrentIndex(self, index: int) -> bool:
        """``setCurrentIndex``: select another density by its index.

        Returns what ROOT's ``RooCategory::setIndex`` returns, which is its own
        backwards convention: ``False`` when the state was set, ``True`` when the
        index was not one of the category's states.
        """
        return bool(self._index.setIndex(int(index)))

    def getCurrentPdf(self) -> Any:
        """``getCurrentPdf``: the selected density itself."""
        return self._pdfs[self.getCurrentIndex()]

    def pdfList(self) -> list[Any]:
        """``pdfList``: every density it may be, in the order given."""
        return list(self._pdfs)

    def leaves(self) -> list[Any]:
        """The selected density's leaves, and the index that says which one it is.

        Generation and normalisation walk a density's leaves to find what it depends
        on, so a density that *is* another has to answer with that one's - and with
        the index, which is a discrete parameter the model fits.
        """
        found = list(self.getCurrentPdf().leaves())
        if all(one is not self._index for one in found):
            found.append(self._index)
        return found

    # --- being a density -----------------------------------------------------

    def compute(self, ctx: Context) -> Any:
        """The selected density's value."""
        return self.getCurrentPdf().compute(ctx)

    def __repr__(self) -> str:
        return (f"<RooMultiPdf {self.GetName()!r} of {len(self._pdfs)} "
                f"at index {self.getCurrentIndex()}>")
