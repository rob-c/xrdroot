"""ROOT's histograms as the Unified Histogram Interface (UHI) has them: indexed, plotted, saved.

:class:`UnifiedHistogram` is what a ``TH1`` - and so every histogram class -
adds to ROOT's own methods: UHI's indexing (:mod:`.indexing`), its plotting
protocol (:mod:`.plotting`) and its serialization (:mod:`.serialization`).
``ROOT.uhi`` is the module of tags an index is written with (:mod:`.tags`).
The submodules keep ROOT's own names, which PyROOT's tests import.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .indexing import _getitem, _iter, _setitem
from .plotting import Kind, PlottableAxisBase, _axes, _counts, _kind, _values, _variances
from .serialization import _from_uhi_, _to_uhi_

__all__ = ["UnifiedHistogram"]


class UnifiedHistogram:
    """UHI's protocols on a histogram of ROOT's.

    ``h[index]`` and ``h[index] = value`` index it, ``iter(h)`` runs over
    every bin's content, the flow too; ``kind``, ``axes``, ``values()``,
    ``variances()`` and ``counts()`` are what plotting libraries read;
    ``_to_uhi_()`` and ``_from_uhi_()`` are what UHI's writers and readers
    call, and the constructor takes UHI's dictionary as well.
    """

    def __getitem__(self, index: Any) -> Any:
        return _getitem(self, index)

    def __setitem__(self, index: Any, value: Any) -> None:
        _setitem(self, index, value)

    def __iter__(self) -> Iterator[float]:
        return _iter(self)

    @property
    def kind(self) -> Kind:
        return _kind(self)

    @property
    def axes(self) -> tuple[PlottableAxisBase, ...]:
        return _axes(self)

    def values(self, flow: bool = False, *, writable: bool = False) -> Any:
        return _values(self, flow, writable=writable)

    def variances(self, flow: bool = False) -> Any:
        return _variances(self, flow)

    def counts(self, flow: bool = False) -> Any:
        return _counts(self, flow)

    def _to_uhi_(self) -> dict[str, Any]:
        return _to_uhi_(self)

    @classmethod
    def _from_uhi_(cls, uhi_dict: dict[str, Any]) -> Any:
        return _from_uhi_(cls, uhi_dict)
