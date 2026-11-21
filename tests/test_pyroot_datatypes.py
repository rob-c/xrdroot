"""``TDataType::GetType``: the ``EDataType`` code ROOT gives a type's ``typeid``.

ROOT compares a ``type_info`` with each of ``char``, ``int``, ``double`` and their kin and
answers ``kOther_t`` for anything else - ``const char *`` and every class among them. The
codes are measured against ROOT 6.40's own ``TDataType::GetType``.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pycxx.runtime.typeinfo import typeid_of
from xrdroot.pyroot.core.datatypes import EDataType, TDataType


@pytest.mark.parametrize(
    ("spelling", "code"),
    [
        ("char", EDataType.kChar_t),
        ("unsigned char", EDataType.kUChar_t),
        ("short", EDataType.kShort_t),
        ("unsigned short", EDataType.kUShort_t),
        ("int", EDataType.kInt_t),
        ("unsigned int", EDataType.kUInt_t),
        ("long", EDataType.kLong_t),
        ("unsigned long", EDataType.kULong_t),
        ("long long", EDataType.kLong64_t),
        ("unsigned long long", EDataType.kULong64_t),
        ("float", EDataType.kFloat_t),
        ("double", EDataType.kDouble_t),
        ("bool", EDataType.kBool_t),
        ("char*", EDataType.kCharStar),
    ],
)
def test_a_fundamental_type_answers_the_code_root_gives_it(spelling, code):
    expect((TDataType.GetType(typeid_of(spelling)), code))


def test_a_type_root_does_not_name_is_other_as_it_is_for_a_const_char_pointer():
    expect(
        (TDataType.GetType(typeid_of("const char*")), EDataType.kOther_t),
        (TDataType.GetType(typeid_of("TNamed")), EDataType.kOther_t),
    )


def test_the_codes_are_the_namespace_s_own_as_roots_are():
    expect(
        (ROOT.kInt_t, EDataType.kInt_t),
        (ROOT.kBool_t, EDataType.kBool_t),
        (ROOT.kOther_t, EDataType.kOther_t),
        (ROOT.EDataType.kDouble_t, EDataType.kDouble_t),
    )
