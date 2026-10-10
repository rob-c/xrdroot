"""ROOT's graphics, in batch mode: canvases, pads, styles, colours, text, shapes and legends.

    >>> import xrdroot.pyroot as ROOT                                   # doctest: +SKIP
    >>> c = ROOT.TCanvas("c", "c", 800, 600)                            # doctest: +SKIP
    >>> c.Divide(2, 1); c.cd(1); h.Draw("E1")                           # doctest: +SKIP
    >>> ROOT.gStyle.SetOptStat(0); c.SaveAs("c.png")                    # doctest: +SKIP

A pad drawn here is kept as a saved one is read, and drawn by the same
code, :mod:`xrdroot.canvas`, as a canvas ROOT saved: what it holds, by the
option each was drawn with, and what ROOT adds when it paints - frame,
title, stats box - made from ``gStyle`` as ``THistPainter`` makes them.
Importing this installs the hook ``TObject::Draw`` calls, so drawing any
object puts it on the current pad.
"""

from __future__ import annotations

import atexit

from . import hook
from .canvas import TCanvas
from .colors import PALETTE_NUMBERS, TColor
from .compare import compare_images
from .decorations import TFrame
from .gradients import TColorGradient, TLinearGradient, TRadialGradient
from .legend import TLegend, TLegendEntry
from .output import close_books
from .pads import TPad, gPad
from .paves import TPave, TPaveLabel, TPaveStats, TPavesText, TPaveText
from .pie import TPie, TPieSlice
from .shapes import (
    TArc,
    TArrow,
    TBox,
    TCrown,
    TEllipse,
    TLine,
    TMarker,
    TPolyLine,
    TPolyMarker,
    TWbox,
)
from .style import TStyle, get_style, gStyle, set_style
from .text import TGaxis, TLatex, TMathText, TText

__all__ = [
    # pads
    "TCanvas", "TPad", "gPad", "TFrame",
    # styles and colours
    "TStyle", "gStyle", "TColor", "TColorGradient", "TLinearGradient", "TRadialGradient",
    # text and paves
    "TText", "TLatex", "TMathText", "TPave", "TPaveText", "TPavesText", "TPaveLabel", "TPaveStats",
    "TLegend", "TLegendEntry", "TGaxis",
    # shapes
    "TLine", "TArrow", "TBox", "TWbox", "TEllipse", "TArc", "TCrown", "TMarker",
    "TPolyLine", "TPolyMarker", "TPie", "TPieSlice",
    # the palettes by name
    *PALETTE_NUMBERS,
    # for the harness and gROOT
    "compare_images", "set_style", "get_style",
]  # fmt: skip

globals().update(PALETTE_NUMBERS)

hook.install()
atexit.register(close_books)
