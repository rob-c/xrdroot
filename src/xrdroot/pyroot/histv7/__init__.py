"""``ROOT.Experimental``: ROOT 7's histograms - ``RHist``, its axes, its statistics and its
slices - and their conversion to ``TH1``.

``ROOT.Experimental.RHist['int'](axis)`` is ``ROOT::Experimental::RHist<int>``;
``Experimental.Hist.ConvertToTH1I(hist)`` makes the ``TH1I`` of the same
bins, contents and statistics, as ROOT's ``ConvertToTH1`` makes it.
``Experimental.Internal`` has what ROOT's own tests reach for there: the
axes of a histogram, the slice mapper, ``CreateBinIndexRange`` and the
atomic additions. Other names under ``ROOT::Experimental`` are ROOT 7's
graphics, from :mod:`..rcanvas`, and the machine-learning loader; the
rest are refused by name.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from ..core.tiofeatures import EIOUnsupportedFeatures, ExperimentalEIOFeatures
from .axes import RCategoricalAxis, RRegularAxis, RVariableBinAxis
from .contents import RBinWithError
from .convert import ConvertAxis, converted
from .fillers import RHistAutoAxisFiller, RHistConcurrentFiller, RHistFillContext
from .hist import ENGINE, HIST, Template
from .index import (
    CreateBinIndexRange,
    RBinIndex,
    RBinIndexMultiDimRange,
    RBinIndexRange,
    RLinearizedIndex,
)
from .slices import RSliceBinIndexMapper, RSliceSpec
from .stats import ATOMIC, RHistStats, RWeight
from .variant import RAxes, RAxisVariant

__all__ = ["Experimental"]


def AtomicAdd(target: Any, value: Any) -> None:
    """``Internal::AtomicAdd(&target, value)``: ``target += value`` made whole against other
    threads - a bin content's own ``AtomicAdd``, or the value a cell holds."""
    with ATOMIC:
        if hasattr(target, "AtomicAdd"):
            target.AtomicAdd(value)
        else:
            target.value = target.value + value


def AtomicInc(target: Any) -> None:
    """``Internal::AtomicInc(&target)``: ``target += 1``, made whole against other threads."""
    with ATOMIC:
        if hasattr(target, "AtomicInc"):
            target.AtomicInc()
        else:
            target.value = target.value + 1


class _Namespace:
    """A C++ namespace whose names are this object's attributes."""

    def __init__(self, name: str, **names: Any) -> None:
        self._name = name
        self.__dict__.update(names)

    def __repr__(self) -> str:
        return f"<namespace {self._name}>"


def _converter(kind: str) -> Any:
    def convert(hist: Any) -> Any:
        return converted(kind, hist)

    convert.__name__ = convert.__qualname__ = f"ConvertToTH1{kind}"
    convert.__doc__ = f"``ConvertToTH1{kind}(hist)``: the ``TH1{kind}`` of a 1-D histogram."
    return convert


#: ``ROOT::Experimental::Hist``: the conversions to ``TH1``.
Hist = _Namespace("ROOT::Experimental::Hist",
                  Internal=_Namespace("ROOT::Experimental::Hist::Internal",
                                      ConvertAxis=ConvertAxis),
                  **{f"ConvertToTH1{kind}": _converter(kind) for kind in "CSILFD"})  # fmt: skip

#: ``ROOT::Experimental::Internal``: what ROOT's tests reach into.
Internal = _Namespace("ROOT::Experimental::Internal", RAxes=RAxes,
                      RSliceBinIndexMapper=RSliceBinIndexMapper,
                      CreateBinIndexRange=CreateBinIndexRange, AtomicAdd=AtomicAdd,
                      AtomicInc=AtomicInc)  # fmt: skip


class _Experimental:
    """``ROOT::Experimental``, as far as this kit has it: ROOT 7's histograms, and the
    IO-feature bits a basket records (:mod:`xrdroot.pyroot.core.tiofeatures`)."""

    RHist = HIST
    RHistEngine = ENGINE
    RHistAutoAxisFiller = Template(RHistAutoAxisFiller)
    RHistConcurrentFiller = RHistConcurrentFiller
    RHistFillContext = RHistFillContext
    RRegularAxis = RRegularAxis
    RVariableBinAxis = RVariableBinAxis
    RCategoricalAxis = RCategoricalAxis
    RAxisVariant = RAxisVariant
    RBinIndex = RBinIndex
    RBinIndexRange = RBinIndexRange
    RBinIndexMultiDimRange = RBinIndexMultiDimRange
    RLinearizedIndex = RLinearizedIndex
    RBinWithError = RBinWithError
    RWeight = RWeight
    RHistStats = RHistStats
    RSliceSpec = RSliceSpec
    Hist = Hist
    Internal = Internal
    EIOFeatures = ExperimentalEIOFeatures
    EIOUnsupportedFeatures = EIOUnsupportedFeatures

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name == "ML":  # the machine-learning data loader, TMVA's part of the namespace
            from ..tmva import Experimental as learning

            return learning.ML
        from .. import rcanvas

        if name in rcanvas.__all__:  # ROOT 7's graphics
            return getattr(rcanvas, name)
        raise UnsupportedFeatureError(f"ROOT::Experimental::{name} is not supported: of ROOT 7's "
                                      "classes, xrdroot has the histograms, RHist and its axes, "
                                      "and the graphics, RCanvas and what it draws.")

    def __repr__(self) -> str:
        return "<namespace ROOT::Experimental>"


#: ``ROOT.Experimental``.
Experimental = _Experimental()
