"""The C++ types a macro declares, and what each means for the Python it becomes.

A :class:`CType` is what the translator knows about a declaration: the
base name, canonical (``Int_t`` is ``int``, ``unsigned`` is ``unsigned int``,
``Double32_t`` is ``double``), any template arguments, how many ``*`` and
whether a ``&``, whether ``const``, and array dimensions. Python has no
declarations, so these are what decide the C semantics the translation has
to write out: an ``int`` divided by an ``int`` truncates, a store to a
``float`` rounds to 32 bits, an ``unsigned`` wraps, a pointer compared
with ``0`` is compared with ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

__all__ = [
    "CType",
    "ROOT_TYPEDEFS",
    "canonical",
    "INTEGRAL",
    "FLOATING",
    "SIZES",
    "builtin_name",
]

#: The integral types, canonically spelled, and the width in bits each has on Linux x86-64.
INTEGRAL = {
    "bool": 8,
    "char": 8,
    "signed char": 8,
    "unsigned char": 8,
    "short": 16,
    "unsigned short": 16,
    "int": 32,
    "unsigned int": 32,
    "long": 64,
    "unsigned long": 64,
    "long long": 64,
    "unsigned long long": 64,
    "wchar_t": 32,
    "char16_t": 16,
    "char32_t": 32,
}

#: The floating types, canonically spelled, and their width in bits.
FLOATING = {"float": 32, "double": 64, "long double": 64}

#: ``sizeof`` of every arithmetic type, in bytes.
SIZES = {name: bits // 8 for name, bits in {**INTEGRAL, **FLOATING}.items()}

#: ROOT's typedefs (RtypesCore.h) and the fixed-width ones, as the types they stand for.
ROOT_TYPEDEFS = {
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
    "size_t": "unsigned long",
    "ssize_t": "long",
    "ptrdiff_t": "long",
    "int8_t": "signed char",
    "uint8_t": "unsigned char",
    "int16_t": "short",
    "uint16_t": "unsigned short",
    "int32_t": "int",
    "uint32_t": "unsigned int",
    "int64_t": "long",
    "uint64_t": "unsigned long",
    "Int8_t": "signed char",
    "UInt8_t": "unsigned char",
    "Int16_t": "short",
    "UInt16_t": "unsigned short",
    "Int32_t": "int",
    "UInt32_t": "unsigned int",
    "Int64_t": "long",
    "UInt64_t": "unsigned long",
    "std::size_t": "unsigned long",
    "std::int32_t": "int",
    "std::uint32_t": "unsigned int",
    "std::int64_t": "long",
    "std::uint64_t": "unsigned long",
    "std::uint8_t": "unsigned char",
    "std::ptrdiff_t": "long",
    "Handle_t": "unsigned long",
    "Window_t": "unsigned long",
    "Pixel_t": "unsigned long",
    "Atom_t": "unsigned long",
    "Longptr_t": "long",
    "ULongptr_t": "unsigned long",
}

#: The keywords a built-in type is spelled with, in any order C++ allows.
BUILTIN_WORDS = frozenset(
    "void bool char short int long float double signed unsigned wchar_t char16_t char32_t "
    "auto".split()
)

#: The names of the string types, which become Python ``str``.
STRINGS = frozenset({"std::string", "string", "std::string_view", "string_view"})

#: The smart pointers, which the translation treats as the pointer they hold.
SMART = frozenset(
    {"std::unique_ptr", "unique_ptr", "std::shared_ptr", "shared_ptr", "std::auto_ptr"}
)


#: The words that only modify an integer type's width or sign.
MODIFIERS = frozenset({"signed", "unsigned", "int", "long", "short"})

#: The integer types a sign can be put on.
SIGNABLE = frozenset({"char", "short", "int", "long", "long long"})


def builtin_name(words: list[str]) -> str:
    """The canonical spelling of a built-in type written as ``words``: ``long int`` is ``long``."""
    longs = words.count("long")
    others = [word for word in words if word not in MODIFIERS]
    base = others[0] if others else _integer(words, longs)
    if base == "double" and longs:
        return "long double"
    return _signed(base, words)


def _signed(base: str, words: list[str]) -> str:
    """``base`` with the sign the words give it: ``unsigned int``, ``signed char``."""
    if "unsigned" in words and base in SIGNABLE:
        return f"unsigned {base}"
    if base == "char" and "signed" in words:
        return "signed char"
    return base


def _integer(words: list[str], longs: int) -> str:
    """The integer type ``short``, ``long``, ``long long`` or ``int`` the words spell."""
    if "short" in words:
        return "short"
    return {0: "int", 1: "long"}.get(longs, "long long")


def canonical(name: str) -> str:
    """``name`` with ROOT's and the standard library's typedefs looked through."""
    return ROOT_TYPEDEFS.get(name, name)


@dataclass(eq=False)
class CType:
    """A C++ type as written in a declaration, with its base name looked through typedefs."""

    name: str
    args: list[Any] = field(default_factory=list)
    pointer: int = 0
    reference: bool = False
    const: bool = False
    dims: list[Any] = field(default_factory=list)
    #: A pointer to a function or a ``std::function``: something to call.
    callable: bool = False

    # -- what kind of thing it is ------------------------------------------

    @property
    def is_array(self) -> bool:
        return bool(self.dims)

    @property
    def is_pointer(self) -> bool:
        return self.pointer > 0 and not self.dims

    @property
    def scalar(self) -> bool:
        """A plain value: arithmetic, not a pointer, not an array."""
        return not self.pointer and not self.dims and self.arithmetic

    @property
    def arithmetic(self) -> bool:
        return self.name in INTEGRAL or self.name in FLOATING

    @property
    def integral(self) -> bool:
        return self.scalar and self.name in INTEGRAL

    @property
    def floating(self) -> bool:
        return self.scalar and self.name in FLOATING

    @property
    def is_char(self) -> bool:
        return self.scalar and self.name in ("char", "signed char", "unsigned char")

    @property
    def is_bool(self) -> bool:
        return self.scalar and self.name == "bool"

    @property
    def is_string(self) -> bool:
        """A C string, a character array or a ``std::string``: all Python ``str`` here."""
        chars = self.name in ("char", "signed char", "unsigned char")
        if chars and self.pointer + len(self.dims) == 1:
            return True
        return self.name in STRINGS and not self.pointer and not self.dims

    @property
    def is_smart(self) -> bool:
        return self.name in SMART and not self.pointer

    @property
    def is_void(self) -> bool:
        return self.name == "void" and not self.pointer

    @property
    def is_auto(self) -> bool:
        return self.name in ("auto", "decltype")

    @property
    def is_class(self) -> bool:
        """An object of some class - held by value, pointer or smart pointer alike."""
        return not self.arithmetic and not self.is_string and not self.is_void

    @property
    def is_object_pointer(self) -> bool:
        """A pointer, or smart pointer, to an object of a class: a Python reference or ``None``."""
        return (self.is_pointer and self.is_class) or self.is_smart

    # -- derived types -----------------------------------------------------

    def element(self) -> CType:
        """What indexing this gives: an array's element, or what a pointer points at."""
        if self.dims:
            return replace(self, dims=self.dims[1:], reference=False)
        return replace(self, pointer=max(self.pointer - 1, 0), reference=False)

    def value(self) -> CType:
        """This type with its reference and ``const`` taken off."""
        return replace(self, reference=False, const=False)

    def pointed(self) -> CType:
        """The type ``&x`` has when ``x`` has this one."""
        return replace(self, pointer=self.pointer + 1, reference=False)

    @property
    def size(self) -> int | None:
        """``sizeof`` this type, when it is one this translator knows the size of."""
        if self.pointer and not self.dims:
            return 8
        base = SIZES.get(self.name)
        if base is None:
            return None
        count = 1
        for dim in self.dims:
            if not isinstance(dim, int):
                return None
            count *= dim
        return base * count
