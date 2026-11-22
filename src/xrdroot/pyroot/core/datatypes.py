"""``TDataType``: the ``EDataType`` code ROOT gives a fundamental type.

ROOT asks a type who it is with ``TDataType::GetType(typeid(T))`` and keeps
the answer as one of ``EDataType``'s numbers, which is how a streamer, a
leaf and a branch say what they hold. The numbers themselves are
:class:`~xrdroot.pyroot.core.collectionproxy.EDataType`'s, because a
collection proxy reports one too; this is the lookup from a type's identity
to that number, and the codes under the bare names a macro writes them by.

The identity is ``type_info``'s mangled name, the Itanium C++ ABI's
(``i`` for ``int``, ``d`` for ``double``, ``Pc`` for ``char*``), which is
the spelling :mod:`xrdroot.pycxx.runtime.typeinfo` mangles a type to here.
A type ROOT's ``GetType`` does not answer for is ``kOther_t`` - it answers
``-1`` for ``const char *`` and for every class, measured against ROOT 6.40.

ROOT has one ``TDataType``, and this merge found it written twice: the class a
dictionary reports - ``GetDataType``, the name, the size - is the one in
:mod:`.reflection`, and the codes a C++ type's ``type_info`` is looked up in are
the table here, which that class's ``GetType`` reads. The class of its own that
this module used to carry was hidden behind that one as soon as the namespace
gathered both, and could answer only half of what ROOT's does, so it is gone.
"""

from __future__ import annotations

from .collectionproxy import EDataType

__all__ = [
    "EDataType",
    "kChar_t", "kShort_t", "kInt_t", "kLong_t", "kFloat_t", "kCounter",
    "kCharStar", "kDouble_t", "kDouble32_t", "kchar",
    "kUChar_t", "kUShort_t", "kUInt_t", "kULong_t", "kBits",
    "kLong64_t", "kULong64_t", "kBool_t", "kFloat16_t",
    "kVoid_t", "kDataTypeAliasUnsigned_t", "kDataTypeAliasSignedChar_t",
    "kOther_t", "kNoType_t", "kNumDataTypes",
]  # fmt: skip

#: ``EDataType``'s codes under the bare names a macro writes them by: ``kInt_t``, ``kBool_t``.
kChar_t = EDataType.kChar_t
kShort_t = EDataType.kShort_t
kInt_t = EDataType.kInt_t
kLong_t = EDataType.kLong_t
kFloat_t = EDataType.kFloat_t
kCounter = EDataType.kCounter
kCharStar = EDataType.kCharStar
kDouble_t = EDataType.kDouble_t
kDouble32_t = EDataType.kDouble32_t
kchar = EDataType.kchar
kUChar_t = EDataType.kUChar_t
kUShort_t = EDataType.kUShort_t
kUInt_t = EDataType.kUInt_t
kULong_t = EDataType.kULong_t
kBits = EDataType.kBits
kLong64_t = EDataType.kLong64_t
kULong64_t = EDataType.kULong64_t
kBool_t = EDataType.kBool_t
kFloat16_t = EDataType.kFloat16_t
kVoid_t = EDataType.kVoid_t
kDataTypeAliasUnsigned_t = EDataType.kDataTypeAliasUnsigned_t
kDataTypeAliasSignedChar_t = EDataType.kDataTypeAliasSignedChar_t
kOther_t = EDataType.kOther_t
kNoType_t = EDataType.kNoType_t
kNumDataTypes = EDataType.kNumDataTypes

#: Each fundamental type's ``EDataType``, by the Itanium name ``typeid`` mangles it to.
BY_MANGLED = {
    "c": kChar_t,
    "h": kUChar_t,
    "s": kShort_t,
    "t": kUShort_t,
    "i": kInt_t,
    "j": kUInt_t,
    "l": kLong_t,
    "m": kULong_t,
    "x": kLong64_t,
    "y": kULong64_t,
    "f": kFloat_t,
    "d": kDouble_t,
    "b": kBool_t,
    "Pc": kCharStar,
}  # fmt: skip
