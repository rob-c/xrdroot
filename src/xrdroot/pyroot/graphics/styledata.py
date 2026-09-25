"""The numbers a ``TStyle`` holds, and what each of ROOT's named styles makes them.

:data:`FIELDS` are ``Modern``'s - ROOT's default since 5.30 - so a style
is ``Modern`` and whatever its entry of :data:`STYLES` changes. An entry
``"LabelSize:XYZ"`` is the field of each axis named after the colon; one
with nothing after it, ``"TitleSize:"``, is the pad's title's.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AXIS_FIELDS", "FIELDS", "STYLES", "opt_stat"]

#: Every field of a style that is one number or string, as ``Modern`` has it.
FIELDS: dict[str, Any] = {
    "OptStat": 1111, "OptFit": 0, "OptTitle": 1, "OptDate": 0, "OptFile": 0,
    "OptLogx": 0, "OptLogy": 0, "OptLogz": 0,
    "StatX": 0.98, "StatY": 0.935, "StatW": 0.2, "StatH": 0.16, "StatFont": 42,
    "StatFontSize": 0.0, "StatBorderSize": 1, "StatColor": 0, "StatStyle": 1001,
    "StatTextColor": 1, "StatFormat": "6.4g", "FitFormat": "5.4g",
    "TitleX": 0.5, "TitleY": 0.995, "TitleW": 0.0, "TitleH": 0.0, "TitleAlign": 23,
    "TitleBorderSize": 0, "TitleFillColor": 0, "TitleStyle": 0, "TitleTextColor": 1,
    "CanvasColor": 0, "CanvasBorderMode": 0, "CanvasBorderSize": 2,
    "CanvasDefH": 500, "CanvasDefW": 700, "CanvasDefX": 10, "CanvasDefY": 10,
    "PadColor": 0, "PadBorderMode": 0, "PadBorderSize": 2,
    "PadBottomMargin": 0.1, "PadTopMargin": 0.1, "PadLeftMargin": 0.1, "PadRightMargin": 0.1,
    "PadGridX": 0, "PadGridY": 0, "PadTickX": 0, "PadTickY": 0,
    "FrameFillColor": 0, "FrameFillStyle": 1001, "FrameLineColor": 1, "FrameLineWidth": 1,
    "FrameLineStyle": 1, "FrameBorderMode": 0, "FrameBorderSize": 1,
    "HistLineColor": 602, "HistLineStyle": 1, "HistLineWidth": 1, "HistFillColor": 0,
    "HistFillStyle": 1001, "HistMinimumZero": 0, "HistTopMargin": 0.05,
    "FuncColor": 2, "FuncStyle": 1, "FuncWidth": 2,
    "GridColor": 0, "GridStyle": 3, "GridWidth": 1,
    "LegendBorderSize": 1, "LegendFillColor": 0, "LegendFont": 42, "LegendTextSize": 0.0,
    "LineColor": 1, "LineStyle": 1, "LineWidth": 1, "FillColor": 19, "FillStyle": 1001,
    "MarkerColor": 1, "MarkerStyle": 1, "MarkerSize": 1.0,
    "TextColor": 1, "TextFont": 62, "TextSize": 0.05, "TextAlign": 11, "TextAngle": 0.0,
    "EndErrorSize": 2.0, "ErrorX": 0.5, "NumberContours": 20, "PaintTextFormat": "g",
    "BarWidth": 1.0, "BarOffset": 0.0, "DrawBorder": 0, "StripDecimals": 1,
    "TimeOffset": 788918400.0, "ScreenFactor": 1.0, "HatchesLineWidth": 1,
    "HatchesSpacing": 1.0, "LineScalePS": 3.0, "ColorModelPS": 0, "Legoinnerr": 0.5,
    "ShowEventStatus": 0, "ShowEditor": 0, "ShowToolBar": 0, "ImageScaling": 1.0,
    "CapLinePS": 0, "JoinLinePS": 0, "IsReading": 1, "Grayscale": 0,
}  # fmt: skip

#: Every field a style keeps once per axis - and once more for the pad's title.
AXIS_FIELDS: dict[str, Any] = {
    "LabelSize": 0.035, "LabelFont": 42, "LabelColor": 1, "LabelOffset": 0.005,
    "TitleSize": 0.035, "TitleFont": 42, "TitleColor": 1, "TitleOffset": 1.0,
    "TickLength": 0.03, "Ndivisions": 510, "AxisColor": 1,
}  # fmt: skip

#: What ROOT's styles before ``Modern`` share: grey, bevelled, in bold Helvetica.
_CLASSIC: dict[str, Any] = {
    "TitleX": 0.01, "TitleAlign": 13, "TitleBorderSize": 2, "TitleFillColor": 19,
    "TitleStyle": 1001, "StatFont": 62, "StatBorderSize": 2, "StatColor": 19,
    "StatX": 0.98, "StatY": 0.995, "CanvasColor": 19, "CanvasBorderMode": 1,
    "PadColor": 19, "PadBorderMode": 1, "FrameFillColor": 19, "FrameBorderMode": 1,
    "HistLineColor": 1, "FillColor": 19, "TextFont": 62, "LegendBorderSize": 4,
    "LegendFont": 62, "FuncWidth": 3, "LegendFillColor": 19,
    "LabelFont:XYZ": 62, "TitleFont:XYZ": 62, "TitleFont:": 62, "TitleSize:": 0.0,
    "LabelSize:XYZ": 0.04, "TitleSize:XYZ": 0.04,
}  # fmt: skip

#: What each of ROOT's named styles changes of ``Modern``.
STYLES: dict[str, dict[str, Any]] = {
    "Modern": {},
    "Classic": _CLASSIC,
    "Default": _CLASSIC,
    "Plain": {
        **_CLASSIC, "CanvasColor": 0, "CanvasBorderMode": 0, "PadColor": 0,
        "PadBorderMode": 0, "FrameFillColor": 0, "FrameBorderMode": 0, "TitleFillColor": 0,
        "TitleBorderSize": 1, "StatColor": 0, "StatBorderSize": 1, "FillColor": 0,
        "LegendFillColor": 0,
    },
    "Bold": {
        **_CLASSIC, "PadColor": 10, "CanvasColor": 10, "FrameFillColor": 10, "StatColor": 10,
        "TitleFillColor": 10, "HistLineWidth": 3, "FrameLineWidth": 3, "FuncWidth": 3,
        "LineWidth": 3, "LabelSize:XYZ": 0.05, "TitleSize:XYZ": 0.05,
    },
    "Video": {
        **_CLASSIC, "PadColor": 10, "CanvasColor": 10, "FrameFillColor": 10, "StatColor": 10,
        "TitleFillColor": 10, "HistLineWidth": 8, "FrameLineWidth": 7, "FuncWidth": 8,
        "LineWidth": 3, "LabelSize:XYZ": 0.06, "TitleSize:XYZ": 0.06, "TextSize": 0.08,
    },
    "Pub": {
        **_CLASSIC, "CanvasColor": 0, "CanvasBorderMode": 0, "PadColor": 0,
        "PadBorderMode": 0, "FrameFillColor": 0, "FrameBorderMode": 0, "StatColor": 0,
        "TitleFillColor": 0, "OptTitle": 0, "OptStat": 0, "HistLineWidth": 2,
        "FrameLineWidth": 2, "FuncWidth": 2, "LineWidth": 2, "FillColor": 0,
        "PadLeftMargin": 0.16, "PadBottomMargin": 0.16,
    },
    "ATLAS": {
        "PadTopMargin": 0.05, "PadRightMargin": 0.05, "PadBottomMargin": 0.16,
        "PadLeftMargin": 0.16, "OptTitle": 0, "OptStat": 0, "OptFit": 0,
        "MarkerStyle": 20, "MarkerSize": 1.2, "HistLineWidth": 2, "LineStyle": 1,
        "EndErrorSize": 0.0, "PadTickX": 1, "PadTickY": 1, "TextSize": 0.05,
        "LabelSize:XYZ": 0.05, "TitleSize:XYZ": 0.05, "TitleOffset:X": 1.4,
        "TitleOffset:Y": 1.4, "LegendBorderSize": 0,
    },
    "BELLE2": {
        "PadTopMargin": 0.05, "PadRightMargin": 0.05, "PadBottomMargin": 0.16,
        "PadLeftMargin": 0.18, "OptTitle": 0, "OptStat": 0, "OptFit": 0,
        "MarkerStyle": 20, "MarkerSize": 1.2, "HistLineWidth": 2, "EndErrorSize": 0.0,
        "PadTickX": 1, "PadTickY": 1, "TextSize": 0.05, "LabelSize:XYZ": 0.05,
        "TitleSize:XYZ": 0.05, "TitleOffset:X": 1.4, "TitleOffset:Y": 1.6,
        "LegendBorderSize": 0,
    },
}  # fmt: skip

#: The letters ``SetOptStat("nemr")`` takes, and the digit each is, lowest first.
LETTERS = "nemruoisk"


def opt_stat(mode: Any) -> int:
    """``fOptStat`` from digits or letters: ``"nemr"`` is 1111, a capital a 2 for its error."""
    if not isinstance(mode, str):
        return int(mode)
    total = 0
    for letter in mode:
        place = LETTERS.find(letter.lower())
        if place < 0:
            raise ValueError(f"SetOptStat takes the letters {LETTERS!r}, and {letter!r} is not one")
        total += (2 if letter.isupper() else 1) * 10**place
    return total
