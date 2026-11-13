"""ROOT's typedefs, as the C++ types they stand for: data the dialect and the runtime share.

The dialect translates ``Int_t`` as ``int``; the runtime makes ``Long64_t(x)``, one called
like a function, the cast it is. Kept apart from both, so each reads it without the other.
"""

from __future__ import annotations

__all__ = ["TYPEDEFS"]

#: ROOT's typedefs (RtypesCore.h, GuiTypes.h), as the types they stand for.
TYPEDEFS = {
    "Char_t": "char",
    "UChar_t": "unsigned char",
    "Short_t": "short",
    "UShort_t": "unsigned short",
    "Int_t": "int",
    "UInt_t": "unsigned int",
    "Seek_t": "int",
    "Long_t": "long",
    "ULong_t": "unsigned long",
    "Float_t": "float",
    "Float16_t": "float",
    "Double_t": "double",
    "Double32_t": "double",
    "LongDouble_t": "long double",
    "Text_t": "char",
    "Bool_t": "bool",
    "Byte_t": "unsigned char",
    "Version_t": "short",
    "Option_t": "char",
    "Ssiz_t": "int",
    "Real_t": "float",
    "Long64_t": "long long",
    "ULong64_t": "unsigned long long",
    "Axis_t": "double",
    "Stat_t": "double",
    "Font_t": "short",
    "Style_t": "short",
    "Marker_t": "short",
    "Width_t": "short",
    "Color_t": "short",
    "SCoord_t": "short",
    "Coord_t": "double",
    "Angle_t": "float",
    "Size_t": "float",
    "Int8_t": "signed char",
    "UInt8_t": "unsigned char",
    "Int16_t": "short",
    "UInt16_t": "unsigned short",
    "Int32_t": "int",
    "UInt32_t": "unsigned int",
    "Int64_t": "long",
    "UInt64_t": "unsigned long",
    "Handle_t": "unsigned long",
    "Window_t": "unsigned long",
    "Pixel_t": "unsigned long",
    "Atom_t": "unsigned long",
    "Longptr_t": "long",
    "ULongptr_t": "unsigned long",
}
