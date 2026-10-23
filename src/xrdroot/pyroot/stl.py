"""ROOT adapters register aliases and opaque types with the generic STL implementation."""

from __future__ import annotations

import re as re
import sys as _sys
from collections.abc import Iterable as Iterable
from collections.abc import Iterator as Iterator
from typing import Any as Any

import numpy

from ..cppstd import bitset, random, stdlib, vectorassign, vectorvalue
from ..cppstd import containers as _generic
from ..cppstd.containers import _ARRAY_TYPE as _ARRAY_TYPE
from ..cppstd.containers import _DTYPE_CLASSES as _DTYPE_CLASSES
from ..cppstd.containers import _DTYPE_META as _DTYPE_META
from ..cppstd.containers import _DTYPE_NAMES as _DTYPE_NAMES
from ..cppstd.containers import _NUMERIC_ASSIGN_READERS as _NUMERIC_ASSIGN_READERS
from ..cppstd.containers import _NUMERIC_READERS as _NUMERIC_READERS
from ..cppstd.containers import _OPAQUE as _OPAQUE
from ..cppstd.containers import CPP_NAMES as CPP_NAMES
from ..cppstd.containers import NPOS as NPOS
from ..cppstd.containers import NUMBERS as NUMBERS
from ..cppstd.containers import PYTHON_TYPES as PYTHON_TYPES
from ..cppstd.containers import SMART_POINTERS as SMART_POINTERS
from ..cppstd.containers import TEMPLATES as TEMPLATES
from ..cppstd.containers import __all__ as __all__
from ..cppstd.containers import _ArrayTemplate as _ArrayTemplate
from ..cppstd.containers import _class_type as _class_type
from ..cppstd.containers import _Container as _Container
from ..cppstd.containers import _converted as _converted
from ..cppstd.containers import _default as _default
from ..cppstd.containers import _element as _element
from ..cppstd.containers import _emplaced as _emplaced
from ..cppstd.containers import _Map as _Map
from ..cppstd.containers import _map_class as _map_class
from ..cppstd.containers import _MapIterator as _MapIterator
from ..cppstd.containers import _Namespace as _Namespace
from ..cppstd.containers import _numeric_dtype as _numeric_dtype
from ..cppstd.containers import _ObjectVector as _ObjectVector
from ..cppstd.containers import _opaque as _opaque
from ..cppstd.containers import _Pair as _Pair
from ..cppstd.containers import _pair_class as _pair_class
from ..cppstd.containers import _resolved_class as _root_class
from ..cppstd.containers import _spelled as _spelled
from ..cppstd.containers import _split_arguments as _split_arguments
from ..cppstd.containers import _Stream as _Stream
from ..cppstd.containers import _Template as _Template
from ..cppstd.containers import _templated as _templated
from ..cppstd.containers import _Vector as _Vector
from ..cppstd.containers import _vector_class as _vector_class
from ..cppstd.containers import _VectorIterator as _VectorIterator
from ..cppstd.containers import cpp_name as cpp_name
from ..cppstd.containers import element_type as element_type
from ..cppstd.containers import is_vector as is_vector
from ..cppstd.containers import std as std
from ..cppstd.containers import string as string
from . import stdlib as _root_limits  # noqa: F401
from . import vectorassign as _root_assignment  # noqa: F401

# Preserve the historical module alias without retaining the temporary name.
np = numpy
del numpy

# Retain the legacy aliases without keeping temporary module bindings.
_bitset = bitset
_random = random
_stdlib = stdlib
_register_vector_assignment = vectorassign.register
_assign_vector_value = vectorvalue.assigned
_vector_value = vectorvalue.converted
_native_vector_value = vectorvalue.native
_numeric_vector_input = vectorvalue.numeric_input
_register_vector_value = vectorvalue.register
del bitset, random, stdlib, vectorassign, vectorvalue

NUMBERS.update({
    "Float_t": "float32",
    "Float16_t": "float32",
    "Double_t": "float64",
    "Double32_t": "float64",
    "Char_t": "int8",
    "UChar_t": "uint8",
    "Short_t": "int16",
    "UShort_t": "uint16",
    "Int_t": "int32",
    "UInt_t": "uint32",
    "Long_t": "int64",
    "Long64_t": "int64",
    "ULong_t": "uint64",
    "ULong64_t": "uint64",
    "Bool_t": "bool",
})
STRINGS = (*_generic.STRINGS, "TString")
_generic.TYPE_ALIASES["TString"] = "string"


def _resolve_root_type(name: str) -> type | None:
    import xrdroot.pyroot as ROOT

    found: Any = ROOT
    for part in name.removeprefix("ROOT::").split("::"):
        found = getattr(found, part, None) if part.isidentifier() else None
    return found if isinstance(found, type) else None


def _rvec_of(kind: Any) -> type:
    from .rdf.rvec import RVec

    return RVec[kind]  # type: ignore[no-any-return]


_generic.set_type_resolver(_resolve_root_type)
TEMPLATES.update({
    name: _Template("RVec", 1, _rvec_of)
    for name in ("RVec", "ROOT::RVec", "ROOT::VecOps::RVec", "VecOps::RVec")
})

# Compatibility imports share the implementation's globals, including rebinding.
# A second alias dictionary would bypass user callbacks in numeric lookup tables.
vars(_generic).update({
    "_root_class": _root_class,
    "_resolve_root_type": _resolve_root_type,
    "_rvec_of": _rvec_of,
})
_generic.STRINGS = STRINGS
_sys.modules[__name__] = _generic
