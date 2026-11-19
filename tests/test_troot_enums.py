"""``TEnum`` and the interpreter's table of enumerations: ``gROOT.GetListOfEnums`` and the
lookups a class and a macro make of it.

What ROOT answers here was read off the 6.40.04 oracle: ``gROOT->GetListOfEnums()`` is a
``TListOfEnumsWithLock`` holding a ``TEnum`` per enumeration, an enum's constants are a
``THashList`` of ``TEnumConstant`` made once, ``TEnum::GetEnum`` is keyed by the name C++
writes (``nn::EIn`` is found, a bare ``EIn`` is not), and the underlying type of
``enum EOnlyShort : short`` is ``kShort_t``, which is 2.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _empty_table() -> Iterator[None]:
    """Each test starts with no enum declared and leaves the session's table as it found it."""
    kept = dict(troot._ENUMS)
    troot._ENUMS.clear()
    try:
        yield
    finally:
        troot._ENUMS.clear()
        troot._ENUMS.update(kept)


def test_an_enum_is_kept_by_the_name_cpp_writes_and_noted_once() -> None:
    made = troot.note_enum("N::E", [("kA", 1), ("kB", 2)], "the comment")
    assert made is troot.note_enum("N::E", [("kA", 9)])
    assert troot.find_enum("N::E") is made
    assert troot.find_enum("E") is None
    assert made.GetQualifiedName() == "N::E"
    assert made.GetName() == "E"
    assert made.GetTitle() == "the comment"


def test_an_enum_of_the_global_namespace_is_its_own_qualified_name() -> None:
    made = troot.note_enum("ETest", [("kA", 3), ("kB", 4)])
    assert made.GetQualifiedName() == "ETest"
    assert made.GetClass() is None
    assert made.GetUnderlyingType() == troot.K_INT


def test_the_integer_type_an_enum_is_declared_with_is_the_one_it_answers() -> None:
    short = troot.note_enum("EOnlyShort", [("kA", 0)], underlying="short")
    assert short.GetUnderlyingType() == 2
    unknown = troot.note_enum("EOdd", [("kA", 0)], underlying="long double")
    assert unknown.GetUnderlyingType() == troot.K_INT


def test_an_enums_constants_are_made_once_so_a_held_one_stays_the_same_object() -> None:
    made = troot.note_enum("ETest", [("kA", 3), ("kB", 4)])
    constants = made.GetConstants()
    assert constants is made.GetConstants()
    assert [one.GetName() for one in constants] == ["kA", "kB"]
    assert made.GetConstant("kB") is constants.FindObject("kB")
    assert made.GetConstant("kC") is None


def test_a_constant_of_no_enum_has_no_title_and_no_enum() -> None:
    alone = troot.TEnumConstant("kA", 3)
    assert (alone.GetName(), alone.GetTitle(), alone.GetValue()) == ("kA", "", 3)
    assert alone.GetEnum() is None


def test_a_constant_carries_the_title_of_the_enum_it_belongs_to() -> None:
    made = troot.note_enum("ETest", [("kA", 3)], "the comment")
    assert made.GetConstant("kA").GetTitle() == "the comment"
    assert made.GetConstant("kA").GetEnum() is made


def test_the_session_lists_every_enum_and_a_class_lists_the_ones_it_owns() -> None:
    global_enum = troot.note_enum("ETest", [("kA", 3)])
    nested = troot.note_enum("A::B::C::enumC1", [("kClassEnumC1A", 0)])
    namespaced = troot.note_enum("N::namespaceEnum", [("kNamespace", 1)])
    listed = ROOT.gROOT.GetListOfEnums()
    assert listed.ClassName() == "TListOfEnumsWithLock"
    assert [one.GetName() for one in listed] == ["ETest", "enumC1", "namespaceEnum"]
    assert listed.FindObject("enumC1") is nested
    assert list(troot.class_enums("")) == [global_enum]
    assert list(troot.class_enums("N")) == [namespaced]
    assert list(troot.class_enums("Nowhere")) == []


def test_an_enum_of_a_class_answers_with_that_class() -> None:
    made = troot.note_enum("A::B::C::enumC1", [("kA", 0)], scoped=True)
    assert made.IsScoped()
    assert made.GetClass().GetName() == "A::B::C"


def test_t_enum_get_enum_finds_what_was_declared_and_nothing_else() -> None:
    made = troot.note_enum("ETest", [("kA", 3)])
    assert troot.TEnum.GetEnum("ETest") is made
    assert troot.TEnum.GetEnum("ENowhere") is None
    assert troot.TEnum.GetEnum("ETest", troot.TEnum.kAutoload) is made
    assert troot.TEnum.kNoAutoload != troot.TEnum.kAutoload
