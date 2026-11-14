"""``ROOT.uhi``: the tags a histogram's UHI index is written with.

    >>> h[ROOT.uhi.loc(2.5)]                 # the bin 2.5 falls in   # doctest: +SKIP
    >>> h[ROOT.uhi.underflow]                # the underflow          # doctest: +SKIP
    >>> h[:: ROOT.uhi.rebin(2)]              # every two bins merged  # doctest: +SKIP
    >>> h2[:, ROOT.uhi.sum]                  # y summed away          # doctest: +SKIP

See :mod:`xrdroot.pyroot._pythonization._uhi` for what each does.
"""

from __future__ import annotations

from ._pythonization._uhi.tags import _loc as loc
from ._pythonization._uhi.tags import _overflow as overflow
from ._pythonization._uhi.tags import _rebin as rebin
from ._pythonization._uhi.tags import _sum as sum
from ._pythonization._uhi.tags import _underflow as underflow

#: Nothing for ``from ROOT import *``: the module itself is ``ROOT.uhi``.
__all__: list[str] = []

#: The tags, by the names a script imports them under: ``from ROOT.uhi import loc``.
TAGS = {"loc": loc, "overflow": overflow, "rebin": rebin, "sum": sum, "underflow": underflow}
