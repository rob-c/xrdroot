"""RooFit under ``import ROOT``: :mod:`xrdroot.roofit`'s classes by ROOT's names.

``ROOT.RooRealVar``, ``ROOT.RooGaussian``, ``ROOT.RooFit.Save()``,
``ROOT.RooMsgService.instance()`` are the engine's own classes and
functions: RooFit's names are the engine's names already, so this module
only gathers them, and connects the two things the engine leaves to the
kit - what drawing a frame puts on the pad (the ``Draw`` hook of
:mod:`xrdroot.pyroot.core`), and what a frame's axis is (``TAxis``, when the
core part has one).
"""

from __future__ import annotations

from typing import Any

from ...roofit import histograms as _histograms
from ...roofit.plot import frame as _frame
from ...roofit.plot import params as _params
from ...roofit.registry import classes
from ..core import draw_hook
from .commands import RooFit

__all__ = ["RooFit"]


def _gather() -> None:
    for name, cls in classes().items():
        if not name.startswith("Roo"):
            continue  # TMatrixDSym and the like are the core part's to give
        globals()[name] = cls
        __all__.append(name)


def _axis() -> Any:
    """The core part's ``TAxis`` over a frame's axis members, or the engine's stand-in."""
    try:
        from ..core.axes import TAxis  # the swap point: the core part's TAxis, when it is there
    except ImportError:
        return _frame.Axis
    return TAxis._of  # pragma: no cover - once the core part is merged


def _pave() -> Any:
    """The graphics part's ``TPaveText``, which a ``paramOn`` box is, or the engine's stand-in."""
    try:
        from ..graphics.paves import TPaveText
    except ImportError:  # pragma: no cover - the graphics part is always there
        return _params.Pave
    return TPaveText


def _histogram_wrapper() -> Any:
    """The core part's wrapping of an :class:`xrdroot.Histogram` as a ``TH1``, or none (the swap
    point).

    The histogram is put in ``gStyle``'s attributes first, as ``TH1::Build``
    puts every histogram ROOT books, the ones ``createHistogram`` makes too.
    """
    try:
        from ..core.wrapping import wrap
    except ImportError:
        return lambda made: made
    from ..core.histcore import _in_style

    def styled(made: Any) -> Any:
        return wrap(_in_style(made))

    styled.wraps = wrap  # type: ignore[attr-defined]
    return styled


_gather()
_histograms.set_wrapper(_histogram_wrapper())
_params.set_pave(_pave())
_frame.set_drawer(draw_hook)
_frame.set_axis(_axis())
