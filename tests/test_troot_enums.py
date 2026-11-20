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
    ROOT.gROOT._lists["Globals"].Clear()
    try:
        yield
    finally:
        troot._ENUMS.clear()
        troot._ENUMS.update(kept)
        ROOT.gROOT._lists["Globals"].Clear()


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
    assert made.GetUnderlyingType() == troot.K_UINT


def test_the_integer_type_an_enum_is_declared_with_is_the_one_it_answers() -> None:
    short = troot.note_enum("EOnlyShort", [("kA", 0)], underlying="short")
    assert short.GetUnderlyingType() == 2
    unknown = troot.note_enum("EOdd", [("kA", 0)], underlying="long double")
    assert unknown.GetUnderlyingType() == troot.K_UINT
    negative = troot.note_enum("EOddNeg", [("kA", -1)], underlying="long double")
    assert negative.GetUnderlyingType() == troot.K_INT


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
    assert troot.TEnum.GetEnum("ETest", troot.TEnum.kNoAutoload) is made
    # ROOT 6.40's ``kNoAutoload`` is ``kNone``, not ROOT 5's ``kAutoload``: the oracle prints
    # both as 0, so a lookup by that name asks the table alone.
    assert troot.TEnum.kNoAutoload == troot.TEnum.kNone != troot.TEnum.kAutoload


def test_a_lookup_that_may_autoload_reads_the_library_its_rootmap_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROOT loads the library a rootmap lists an enumeration in and looks again: the table is
    what a dictionary notes its enumerations against, so the second look is what finds it."""
    asked: list[str] = []

    def loader(name: str) -> bool:
        asked.append(name)
        troot.note_enum(name, [("kOne", 1)])
        return True

    monkeypatch.setitem(troot._PROVIDED, "autoload_enum", lambda: loader)
    found = troot.TEnum.GetEnum("myns::EIn")
    assert (asked, found) == (["myns", "myns::EIn"], troot.find_enum("myns::EIn"))
    # ``kInterpLookup`` and ``kNone`` ask the table alone, which holds nothing yet.
    asked.clear()
    troot._ENUMS.clear()
    assert troot.TEnum.GetEnum("myns::EIn", troot.TEnum.kInterpLookup) is None
    assert troot.TEnum.GetEnum("myns::EIn", troot.TEnum.kNone) is None
    assert asked == []


def test_without_a_translator_a_lookup_that_may_autoload_only_asks_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session with no translator loaded - a PyROOT program using no C++ - cannot read a
    library, so the lookup answers what the table holds and no more."""
    monkeypatch.setitem(troot._PROVIDED, "autoload_enum", _no_translator)
    assert troot._autoload_enum("myns::EIn") is None
    assert troot.TEnum.GetEnum("myns::EIn") is None
    troot.note_enum("myns::EIn", [("kOne", 1)])
    assert troot.TEnum.GetEnum("myns::EIn") is troot.find_enum("myns::EIn")


def test_a_lookup_does_not_read_a_library_for_a_scope_the_interpreter_has(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROOT looks an enumeration up where it would be declared, and reads a library only when
    that scope is not here at all: roottest's ``execenumsInNamespaces`` declares an empty
    ``myNamespace`` and then does not find the ``A`` the dictionary on its path declares."""
    asked: list[str] = []

    def loader(name: str) -> bool:
        asked.append(name)
        return False

    monkeypatch.setitem(troot._PROVIDED, "autoload_enum", lambda: loader)
    ROOT.gInterpreter.ProcessLine("namespace myNamespace { }")
    assert troot._owner_known("myNamespace")
    assert troot.TEnum.GetEnum("myNamespace::A") is None
    assert asked == []


def test_a_lookup_finds_the_enumeration_a_library_for_its_scope_declares(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ROOT asks for a library declaring the scope first, which is how a class's
    ``Holder::Mode`` is found before ``Holder`` itself: the library's own load notes the
    enumeration, and the lookup takes it without asking for the enumeration's name too."""
    asked: list[str] = []

    def loader(name: str) -> bool:
        asked.append(name)
        troot.note_enum("Holder::Mode", [("kOff", 0)])
        return True

    monkeypatch.setitem(troot._PROVIDED, "autoload_enum", lambda: loader)
    found = troot.TEnum.GetEnum("Holder::Mode")
    assert (asked, found) == (["Holder"], troot.find_enum("Holder::Mode"))


def test_the_scope_a_name_is_qualified_with_is_every_part_of_it() -> None:
    """A declared ``myns`` does not make ``myns::inner`` a scope the interpreter has, and a
    name in no scope - one with no ``::`` in it - is a scope it knows nothing about."""
    ROOT.gInterpreter.ProcessLine("namespace outer { namespace inner { } }")
    assert (troot._owner_known("outer"), troot._owner_known("outer::inner")) == (True, True)
    assert (troot._owner_known("outer::missing"), troot._owner_known("")) == (False, False)


def _no_translator() -> None:
    """The provider of a ``gInterpreter->AutoLoad`` that this session cannot reach."""
    raise AttributeError("interpreter.autoload")


class _TypeInfo:
    """``typeid(E)`` as a program passes it: ROOT spells it out and looks that up."""

    def __str__(self) -> str:
        return "N::E"


def test_the_integer_type_an_enum_is_stored_as_is_the_one_clang_gives_it() -> None:
    """The oracle's answers: ``enum E {kA, kB}`` is stored ``unsigned int`` (13), one with a
    negative enumerator ``int`` (3), an ``enum class`` ``int``, and ``: short`` is 2."""
    plain = troot.note_enum("EPlain", [("kA", 0), ("kB", 1)])
    assert plain.GetUnderlyingType() == troot.K_UINT == 13
    negative = troot.note_enum("ENeg", [("kC", -1), ("kD", 0)])
    assert negative.GetUnderlyingType() == troot.K_INT == 3
    assert troot.note_enum("EScoped", [("kX", 0)], scoped=True).GetUnderlyingType() == 3
    assert troot.note_enum("EShort", [("kS", 0)], underlying="short").GetUnderlyingType() == 2


def test_every_integer_type_an_enum_may_be_declared_with_is_the_code_root_gives_it() -> None:
    for index, (written, code) in enumerate(troot.UNDERLYING_TYPES.items()):
        made = troot.note_enum(f"EType{index}", [("kA", 0)], underlying=written)
        assert made.GetUnderlyingType() == code, written


def test_a_global_enumerator_is_read_through_the_address_it_is_kept_at() -> None:
    made = troot.TGlobal("kA", "the comment", "int", 7)
    assert (made.GetName(), made.GetTitle()) == ("kA", "the comment")
    assert made.GetTypeName() == "int"
    assert made.GetFullTypeName() == "int"
    assert made.GetAddress().value == 7
    made.GetAddress().value = 9  # a macro writes into the constant it found
    assert made.GetAddress().value == 9
    assert troot.TGlobal("kB").GetAddress().value == 0


def test_the_globals_are_the_enumerators_of_the_enumerations_no_namespace_owns() -> None:
    troot.note_enum("ETest", [("kA", 3), ("kB", 4)])
    troot.note_enum("N::EIn", [("kIn", 1)])
    assert [one.GetName() for one in troot.global_constants()] == ["kA", "kB"]
    assert [one.GetTypeName() for one in troot.global_constants("N")] == ["EIn"]
    assert troot.global_constants("Nowhere") == []


def test_the_sessions_list_of_globals_holds_each_enumerator_of_the_global_scope_once() -> None:
    troot.note_enum("ETest", [("kA", 3)])
    troot.note_enum("N::EIn", [("kIn", 1)])
    listed = ROOT.gROOT.GetListOfGlobals(True)
    assert listed is ROOT.gROOT.GetListOfGlobals()
    assert [one.GetName() for one in listed] == ["kA"]
    assert listed.FindObject("kA").GetAddress().value == 3


def test_a_lookup_by_type_info_is_a_lookup_by_the_name_it_spells() -> None:
    made = troot.note_enum("N::E", [("kA", 0)])
    assert troot.TEnum.GetEnum(_TypeInfo()) is made


def test_a_lookup_that_asks_the_interpreter_alone_finds_only_what_was_noted() -> None:
    """Nothing here reads a library for a lookup, so whichever ``ESearchAction`` a program
    passes it finds exactly the enumerations the interpreter has been told about."""
    made = troot.note_enum("ETest", [("kA", 3)])
    for action in (troot.TEnum.kNone, troot.TEnum.kInterpLookup, troot.TEnum.kAutoload,
                   troot.TEnum.kALoadAndInterpLookup):  # fmt: skip
        assert troot.TEnum.GetEnum("ETest", action) is made
        assert troot.TEnum.GetEnum("ENowhere", action) is None
    assert (troot.TEnum.kNone, troot.TEnum.kInterpLookup) == (0, 2)
    assert (troot.TEnum.kAutoload, troot.TEnum.kALoadAndInterpLookup) == (1, 3)
    assert troot.TEnum.kNoAutoload == troot.TEnum.kNone
