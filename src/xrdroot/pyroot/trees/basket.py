"""``TBasket``, as far as its IO-feature bits go.

A basket's header records, in ``kIOBitCount`` bits, which IO features its
writer used - ``TBufferFile::EIOBits``. That is the whole of ``TBasket`` a test
of this kit reads: a basket is otherwise the writer's and the reader's own
business (:mod:`xrdroot.wtree`, :mod:`xrdroot.pyroot.trees.memory`), and any
other name of the class is refused by the namespace's own sentence.
"""

from __future__ import annotations

from ..core.tiofeatures import IO_BIT_COUNT

__all__ = ["TBasket"]


class TBasket:
    """``TBasket``: the bits a basket's header spends on saying how it was written."""

    #: How many IO-feature bits a basket has room for: one in ROOT 6.40.
    kIOBitCount = IO_BIT_COUNT

    class EIOBits:
        """``TBasket::EIOBits``: the bits a basket of this release sets, being all of them."""

        kSupported = (1 << IO_BIT_COUNT) - 1
