"""Canvases made byte by byte, for the drawing classes no file here holds.

The one canvas in ``tests/data`` that ROOT wrote, ``tcanvas.root``, is a
single pad drawing one graph, saved without being drawn. Everything else a
canvas can hold - nested pads, histograms by option, text in ROOT's
mathematics, lines, arrows, boxes, ellipses, markers, paves, stats boxes
and legends - is written here by :mod:`crafted`: ``TPad`` as
``tcanvas.root`` describes it, ``TCanvas`` in the order its own streamer
writes it (and, as ROOT does, not described), and the drawing classes from
the members their C++ headers declare. That tests reading those layouts
faithfully and drawing what they hold, not that the layouts are ROOT's.
"""

from __future__ import annotations

import pathlib
from typing import Any

from crafted import DATA, UNDESCRIBED, base, craft, layouts, member

from xrdroot import open_root
from xrdroot.streamers import Member

#: ``TObject``'s bits for an object that is on the heap, as every drawn one is.
HEAP = 0x03000000
#: ``SetNDC``'s bit, for text, lines and markers placed in the pad's fractions.
NDC = 1 << 14

#: The drawing classes, as their headers declare their persistent members.
DRAWING: dict[str, list[Member]] = {
    "TAttText": [
        member("fTextAngle", 5, "float"),
        member("fTextSize", 5, "float"),
        member("fTextAlign", 2, "short"),
        member("fTextColor", 2, "short"),
        member("fTextFont", 2, "short"),
    ],
    "TText": [
        base("TNamed", 67),
        base("TAttText"),
        base("TAttBBox2D"),
        member("fX", 8, "double"),
        member("fY", 8, "double"),
    ],
    "TLatex": [
        base("TText"),
        base("TAttLine"),
        member("fFactorSize", 8, "double"),
        member("fFactorPos", 8, "double"),
        member("fLimitFactorSize", 3, "int"),
        member("fOriginSize", 8, "double"),
    ],
    "TLine": [
        base("TObject", 66),
        base("TAttLine"),
        base("TAttBBox2D"),
        *(member(name, 8, "double") for name in ("fX1", "fY1", "fX2", "fY2")),
    ],
    "TArrow": [
        base("TLine"),
        base("TAttFill"),
        member("fAngle", 5, "float"),
        member("fArrowSize", 5, "float"),
        member("fOption", 65, "TString"),
    ],
    "TWbox": [base("TBox"), member("fBorderSize", 2, "short"), member("fBorderMode", 2, "short")],
    "TFrame": [base("TWbox")],
    "TEllipse": [
        base("TObject", 66),
        base("TAttLine"),
        base("TAttFill"),
        base("TAttBBox2D"),
        *(
            member(name, 8, "double")
            for name in ("fX1", "fY1", "fR1", "fR2", "fPhimin", "fPhimax", "fTheta")
        ),
    ],
    "TMarker": [
        base("TObject", 66),
        base("TAttMarker"),
        base("TAttBBox2D"),
        member("fX", 8, "double"),
        member("fY", 8, "double"),
    ],
    "TPave": [
        base("TBox"),
        *(member(name, 8, "double") for name in ("fX1NDC", "fY1NDC", "fX2NDC", "fY2NDC")),
        member("fBorderSize", 3, "int"),
        member("fInit", 3, "int"),
        member("fShadowColor", 3, "int"),
        member("fCornerRadius", 8, "double"),
        member("fOption", 65, "TString"),
        member("fName", 65, "TString"),
    ],
    "TPaveText": [
        base("TPave"),
        base("TAttText"),
        member("fLabel", 65, "TString"),
        member("fLongest", 3, "int"),
        member("fMargin", 5, "float"),
        member("fLines", 64, "TList*"),
    ],
    "TPaveStats": [
        base("TPaveText"),
        member("fOptFit", 3, "int"),
        member("fOptStat", 3, "int"),
        member("fFitFormat", 65, "TString"),
        member("fStatFormat", 65, "TString"),
        member("fParent", 64, "TObject*"),
    ],
    "TPaveLabel": [base("TPave"), base("TAttText"), member("fLabel", 65, "TString")],
    "TPaletteAxis": [base("TPave")],
    "TLegend": [
        base("TPave"),
        base("TAttText"),
        member("fPrimitives", 64, "TList*"),
        member("fEntrySeparation", 5, "float"),
        member("fMargin", 5, "float"),
        member("fNColumns", 3, "int"),
        member("fColumnSeparation", 5, "float"),
    ],
    "TLegendEntry": [
        base("TObject", 66),
        base("TAttText"),
        base("TAttLine"),
        base("TAttFill"),
        base("TAttMarker"),
        member("fObject", 64, "TObject*"),
        member("fLabel", 65, "TString"),
        member("fOption", 65, "TString"),
    ],
    "TColor": [
        base("TNamed", 67),
        member("fNumber", 3, "int"),
        *(
            member(name, 5, "float")
            for name in ("fRed", "fGreen", "fBlue", "fHue", "fLight", "fSaturation", "fAlpha")
        ),
    ],
    "TGaxis": [base("TLine"), base("TAttText"), member("fWmin", 8, "double")],
    "THStack": [
        base("TNamed", 67),
        member("fHists", 64, "TList*"),
        member("fHistogram", 64, "TH1*"),
        member("fMaximum", 8, "double"),
        member("fMinimum", 8, "double"),
    ],
    # Written but never described, to stand for a class a file holds and this cannot read.
    "TUndescribed": [member("fWhatever", 3, "int")],
    # Streams itself as nothing at all.
    "TQObject": [],
    # The order TCanvas::Streamer writes its members in, which ROOT does not describe.
    "TCanvas": [
        base("TPad"),
        member("fDISPLAY", 65, "TString"),
        member("fDoubleBuffer", 3, "int"),
        member("fRetained", 18, "bool"),
        *(
            member(name, 5, "float")
            for name in ("fXsizeUser", "fYsizeUser", "fXsizeReal", "fYsizeReal")
        ),
        member("fWindowTopX", 3, "int"),
        member("fWindowTopY", 3, "int"),
        *(
            member(name, 13, "unsigned int")
            for name in ("fWindowWidth", "fWindowHeight", "fCw", "fCh")
        ),
        member("fCatt", 61, "TAttCanvas"),
        member("kMoveOpaque", 18, "bool"),
        member("kResizeOpaque", 18, "bool"),
        member("fHighLightColor", 2, "short"),
        *(
            member(name, 18, "bool")
            for name in ("fBatch", "kShowEventStatus", "kAutoExec", "kMenuBar")
        ),
    ],
}

#: What an attribute is when a test does not say: what ROOT's constructors give.
DEFAULTS: dict[str, Any] = {
    "fLineColor": 1,
    "fLineStyle": 1,
    "fLineWidth": 1,
    "fFillColor": 0,
    "fFillStyle": 1001,
    "fMarkerColor": 1,
    "fMarkerStyle": 1,
    "fMarkerSize": 1.0,
    "fTextAlign": 11,
    "fTextColor": 1,
    "fTextFont": 42,
    "fTextSize": 0.05,
    "fLeftMargin": 0.1,
    "fRightMargin": 0.1,
    "fBottomMargin": 0.1,
    "fTopMargin": 0.1,
    "fFrameLineColor": 1,
    "fFrameLineWidth": 1,
    "fFrameFillStyle": 1001,
    "fFrameBorderSize": 1,
    "fX2": 1.0,
    "fY2": 1.0,
    "fUxmax": 1.0,
    "fUymax": 1.0,
    "fWNDC": 1.0,
    "fHNDC": 1.0,
    "fAbsWNDC": 1.0,
    "fAbsHNDC": 1.0,
    "fBorderSize": 2,
    "fPhimax": 360.0,
    "fMaximum": -1111.0,
    "fMinimum": -1111.0,
}


def known() -> dict[str, dict[str, Member]]:
    """Every layout a crafted canvas is written with, described or not."""
    made = layouts()
    for donor in ("tcanvas.root", "tgme.root", "embedded-tbox.root"):
        with open_root(str(DATA / donor)) as f:
            for name, members in f._source.streamers().items():
                made.setdefault(name, members)
    for name, members in DRAWING.items():
        made[name] = {one.name: one for one in members}
    return made


#: Every layout, read once.
KNOWN = known()


def nest(classname: str, **flat: Any) -> dict[str, Any]:
    """The members of ``classname``, each base under its own name, from one flat list.

    A member named anywhere in the class or its bases takes the value given
    for it, then :data:`DEFAULTS`; ``fBits`` goes to the ``TObject`` or
    ``TNamed`` at the bottom, and ``fName`` and ``fTitle`` to the ``TNamed``.
    """
    row: dict[str, Any] = {}
    for one in KNOWN[classname].values():
        if one.stype == 0:
            row[one.name] = nest(one.name, **flat)
        elif one.stype == 66:
            row[one.name] = {"fUniqueID": 0, "fBits": flat.get("fBits", HEAP)}
        elif one.stype == 67:
            row[one.name] = {
                "fName": flat.get("fName", ""),
                "fTitle": flat.get("fTitle", ""),
                "fBits": flat.get("fBits", HEAP),
            }
        elif one.name in flat:
            row[one.name] = flat[one.name]
        elif one.name in DEFAULTS:
            row[one.name] = DEFAULTS[one.name]
    return row


def pad(name: str, primitives: list[tuple[Any, ...]], **flat: Any) -> dict[str, Any]:
    """A ``TPad``'s members: what it draws, as ``(class, members, option)``."""
    flat.setdefault("fTitle", name)
    return nest("TPad", fName=name, fPrimitives=primitives, **flat)


def canvas(
    name: str,
    primitives: list[tuple[Any, ...]],
    width: int = 700,
    height: int = 500,
    **flat: Any,
) -> dict[str, Any]:
    """A ``TCanvas``'s members: its pad, and its size in pixels."""
    return {
        "TPad": pad(name, primitives, **flat),
        "fDISPLAY": "$DISPLAY",
        "fCw": width,
        "fCh": height,
        "fWindowWidth": width + 4,
        "fWindowHeight": height + 28,
        "fCatt": {},
        "fBatch": True,
    }


def write(path: pathlib.Path, *canvases: tuple[str, dict[str, Any]]) -> pathlib.Path:
    """A file of canvases, each ``(name, members)``, with the layouts they need."""
    objects = [("TCanvas", name, members) for name, members in canvases]
    return craft(path, objects, KNOWN, UNDESCRIBED | {"TUndescribed"})
