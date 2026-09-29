"""ROOT's enumerations: colours, line, fill and marker styles, palettes and truth.

Each is a plain ``int``, as a C++ enumerator is, so ``kRed + 2`` is colour
634 and ``kBlue - 9`` is 591 exactly as in ROOT; the tables are ROOT's
headers - ``Rtypes.h``, ``TAttLine.h``, ``TAttFill.h``, ``TAttMarker.h``,
``TColor.h`` - copied, and the names are made module attributes from them.
"""

from __future__ import annotations

from typing import Any

#: ``EColor``, from ``Rtypes.h``.
COLORS = {
    "kWhite": 0, "kBlack": 1, "kGray": 920, "kRed": 632, "kGreen": 416, "kBlue": 600,
    "kYellow": 400, "kMagenta": 616, "kCyan": 432, "kOrange": 800, "kSpring": 820,
    "kTeal": 840, "kAzure": 860, "kViolet": 880, "kPink": 900, "kGrape": 100,
    "kBrown": 101, "kAsh": 102, "kP6Blue": 103, "kP6Yellow": 104, "kP6Red": 105,
    "kP6Grape": 106, "kP6Gray": 107, "kP6Violet": 108, "kP8Blue": 109, "kP8Orange": 110,
    "kP8Red": 111, "kP8Pink": 112, "kP8Green": 113, "kP8Cyan": 114, "kP8Azure": 115,
    "kP8Gray": 116, "kP10Blue": 117, "kP10Yellow": 118, "kP10Red": 119, "kP10Gray": 120,
    "kP10Violet": 121, "kP10Brown": 122, "kP10Orange": 123, "kP10Green": 124,
    "kP10Ash": 125, "kP10Cyan": 126,
}  # fmt: skip

#: ``ELineStyle``, from ``TAttLine.h``.
LINE_STYLES = {"kSolid": 1, "kDashed": 2, "kDotted": 3, "kDashDotted": 4}

#: ``EFillStyle``, from ``TAttFill.h``.
FILL_STYLES = {
    "kFDotted1": 3001, "kFDotted2": 3002, "kFDotted3": 3003, "kFHatched1": 3004,
    "kHatched2": 3005, "kFHatched3": 3006, "kFHatched4": 3007, "kFWicker": 3008,
    "kFScales": 3009, "kFBricks": 3010, "kFSnowflakes": 3011, "kFCircles": 3012,
    "kFTiles": 3013, "kFMondrian": 3014, "kFDiamonds": 3015, "kFWaves1": 3016,
    "kFDashed1": 3017, "kFDashed2": 3018, "kFAlhambra": 3019, "kFWaves2": 3020,
    "kFStars1": 3021, "kFStars2": 3022, "kFPyramids": 3023, "kFFrieze": 3024,
    "kFMetopes": 3025, "kFEmpty": 0, "kFSolid": 1,
}  # fmt: skip

#: ``EMarkerStyle``, from ``TAttMarker.h``.
MARKER_STYLES = {
    "kDot": 1, "kPlus": 2, "kStar": 3, "kCircle": 4, "kMultiply": 5, "kFullDotSmall": 6,
    "kFullDotMedium": 7, "kFullDotLarge": 8, "kFullCircle": 20, "kFullSquare": 21,
    "kFullTriangleUp": 22, "kFullTriangleDown": 23, "kOpenCircle": 24, "kOpenSquare": 25,
    "kOpenTriangleUp": 26, "kOpenDiamond": 27, "kOpenCross": 28, "kFullStar": 29,
    "kOpenStar": 30, "kStar2": 31, "kOpenTriangleDown": 32, "kFullDiamond": 33,
    "kFullCross": 34, "kOpenDiamondCross": 35, "kOpenSquareDiagonal": 36,
    "kOpenThreeTriangles": 37, "kOctagonCross": 38, "kFullThreeTriangles": 39,
    "kOpenFourTrianglesX": 40, "kFullFourTrianglesX": 41, "kOpenDoubleDiamond": 42,
    "kFullDoubleDiamond": 43, "kOpenFourTrianglesPlus": 44, "kFullFourTrianglesPlus": 45,
    "kOpenCrossX": 46, "kFullCrossX": 47, "kFourSquaresX": 48, "kFourSquaresPlus": 49,
}  # fmt: skip

#: ``EColorPalette``, from ``TColor.h``, in its order from 51.
_PALETTE_NAMES = (
    "kDeepSea kGreyScale kDarkBodyRadiator kBlueYellow kRainBow kInvertedDarkBodyRadiator "
    "kBird kCubehelix kGreenRedViolet kBlueRedYellow kOcean kColorPrintableOnGrey kAlpine "
    "kAquamarine kArmy kAtlantic kAurora kAvocado kBeach kBlackBody kBlueGreenYellow "
    "kBrownCyan kCMYK kCandy kCherry kCoffee kDarkRainBow kDarkTerrain kFall kFruitPunch "
    "kFuchsia kGreyYellow kGreenBrownTerrain kGreenPink kIsland kLake kLightTemperature "
    "kLightTerrain kMint kNeon kPastel kPearl kPigeon kPlum kRedBlue kRose kRust "
    "kSandyTerrain kSienna kSolar kSouthWest kStarryNight kSunset kTemperatureMap "
    "kThermometer kValentine kVisibleSpectrum kWaterMelon kCool kCopper kGistEarth "
    "kViridis kCividis"
).split()
PALETTES = {name: 51 + index for index, name in enumerate(_PALETTE_NAMES)}
PALETTES.update(kRainbow=PALETTES["kRainBow"], kDarkRainbow=PALETTES["kDarkRainBow"])

#: ``TObject``'s bits, and the options ``Write`` takes.
BITS = {
    "kCanDelete": 1 << 0, "kMustCleanup": 1 << 3, "kIsReferenced": 1 << 4,
    "kHasUUID": 1 << 5, "kCannotPick": 1 << 6, "kNoContextMenu": 1 << 8,
    "kInvalidObject": 1 << 13, "kSingleKey": 1 << 0, "kOverwrite": 1 << 1,
    "kWriteDelete": 1 << 2,
}  # fmt: skip

#: ``kTRUE`` and ``kFALSE``, and the rest of ``RtypesCore.h`` a script meets - and
#: ``GuiTypes.h``'s ``kNone``, the null handle, which scripts take for "nothing yet".
TRUTH: dict[str, Any] = {"kTRUE": True, "kFALSE": False, "kMaxInt": 2**31 - 1, "kNone": 0}

#: Every enumerator here, by name.
ENUMS: dict[str, Any] = {
    **COLORS, **LINE_STYLES, **FILL_STYLES, **MARKER_STYLES, **PALETTES, **BITS, **TRUTH,
}  # fmt: skip

globals().update(ENUMS)

__all__ = sorted(ENUMS)


def named(text: str) -> int:
    """``"kBlue"``, ``"kRed+2"``: what PyROOT makes of an enumerator's name given for a number."""
    import re

    found = re.match(r"^\s*(k\w+)\s*(?:([+-])\s*(\d+))?\s*$", text)
    tables = (COLORS, LINE_STYLES, FILL_STYLES, MARKER_STYLES)
    name = found.group(1) if found else ""
    base = next((table[name] for table in tables if name in table), None)
    if found is None or base is None:
        raise ValueError(f"'{text}' is not the name of one of ROOT's colours or styles.")
    offset = int(found.group(3) or 0)
    return base - offset if found.group(2) == "-" else base + offset
