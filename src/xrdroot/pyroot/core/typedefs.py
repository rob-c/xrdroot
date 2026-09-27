"""ROOT's number types by name: ``ROOT.Float_t(i + 1) / nq``, ``ROOT.Int_t(x)``.

C++ macros cast with them, and PyROOT offers them as types a number is
converted by; here each is the Python type that conversion gives - ``float``
for the floating types, ``int`` for the whole ones and ``bool`` for
``Bool_t`` - so a cast truncates or widens as C++'s does.
"""

from __future__ import annotations

__all__ = [
    "Float_t", "Double_t", "Size_t", "Coord_t", "Axis_t", "Stat_t", "Real_t",
    "Int_t", "UInt_t", "Short_t", "UShort_t", "Long_t", "ULong_t", "Ssiz_t",
    "Width_t", "Color_t", "Style_t", "Marker_t", "Font_t", "Version_t", "Option_t",
    "Bool_t",
]  # fmt: skip

Float_t = Double_t = Size_t = Coord_t = Axis_t = Stat_t = Real_t = float
Int_t = UInt_t = Short_t = UShort_t = Long_t = ULong_t = Ssiz_t = int
Width_t = Color_t = Style_t = Marker_t = Font_t = Version_t = int
Option_t = str
Bool_t = bool
