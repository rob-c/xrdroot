"""The interpreter's table of enumerations: what a line, a macro and a dictionary declare.

ROOT's ``TEnum::GetEnum``, ``TClass::GetListOfEnums`` and ``gROOT->GetListOfGlobals()`` all
answer from a table the interpreter fills as it reads declarations - a line run at the prompt
adds to it as a dictionary loaded into the session does. These ask that table what it holds
after a session has run a line, in the terms ROOT 6.40 answers with: a value counted on from
the last, arithmetic and ``sizeof`` worked out, an ``enum class``'s own integer type.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pycint.execute import process_line
from xrdroot.pycint.library.translate import (
    EnumNote,
    _binary_number,
    _number,
    _size,
    _unary_number,
    enums_of,
)
from xrdroot.pycxx import cache
from xrdroot.pycxx.ctype import CType
from xrdroot.pycxx.errors import Where
from xrdroot.pycxx.nodes import (
    Binary,
    Call,
    EnumDecl,
    Literal,
    Name,
    SizeOf,
    Unary,
    Unit,
)
from xrdroot.pyroot.core import troot


@pytest.fixture(autouse=True)
def _empty_table(tmp_path: Path,
                 monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts with an empty table of enumerations - and a cache of its own, so that
    what a line notes is what this test's run of it noted - and leaves the session's as it
    found it."""
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    kept = dict(troot._ENUMS)
    troot._ENUMS.clear()
    try:
        yield from fresh(tmp_path)
    finally:
        troot._ENUMS.clear()
        troot._ENUMS.update(kept)


def number(where: Where, value: int) -> Literal:
    """An ``int`` as C++ writes it, which is what an enumerator's value is a ``Literal`` of."""
    return Literal(where, "int", value, "int")


def constants(name: str) -> list[tuple[str, int]]:
    """Each constant of the enumeration ``name``, and its value, as the table holds them."""
    made = troot.find_enum(name)
    assert made is not None, name
    return [(one.GetName(), one.GetValue()) for one in made.GetConstants()]


def test_a_line_declaring_an_enumeration_puts_it_in_the_interpreter_s_table() -> None:
    process_line("namespace a { namespace b { enum en{kNull}; } }")
    made = troot.find_enum("a::b::en")
    assert made is not None
    assert made.GetQualifiedName() == "a::b::en"
    assert made.GetTitle() == ""  # a line's enumeration has no comment over it
    assert constants("a::b::en") == [("kNull", 0)]


def test_an_enumerator_s_value_is_the_number_cpp_works_out() -> None:
    """ROOT's table keeps the values, so a program reading a constant gets C++'s number:
    counted on from the last, ``1 << 3``, ``sizeof(int)``, ``true``, and another
    enumeration's constant."""
    process_line(
        "enum A8x {kx = 4};\n"
        "enum E8x {ky = A8x::kx, kz};\n"
        "enum EFlags {kFirst = 1 << 3, kSecond, kMask = (kFirst | 1) & ~2, kSigned = ~0};\n"
        "enum ESized {kInt = sizeof(int), kAfter};\n"
        "enum ETruth {kYes = true, kNo = false, kNext};\n"
    )
    assert constants("E8x") == [("ky", 4), ("kz", 5)]
    assert constants("EFlags") == [("kFirst", 8), ("kSecond", 9), ("kMask", 9),
                                   ("kSigned", -1)]  # fmt: skip
    assert constants("ESized") == [("kInt", 4), ("kAfter", 5)]
    assert constants("ETruth") == [("kYes", 1), ("kNo", 0), ("kNext", 1)]


def test_an_enumeration_no_constant_of_which_can_be_worked_out_is_not_noted() -> None:
    """ROOT's table would hold the constants, so one with none of them is left out rather
    than noted as an enumeration that has none - a wrong table is worse than a missing one."""
    process_line("int one() { return 1; }\nenum EUnknown {kA = one(), kB};")
    assert troot.find_enum("EUnknown") is None


def test_an_enumeration_declared_twice_is_the_one_the_table_already_holds() -> None:
    process_line("enum ETwice {kA, kB};")
    process_line("enum ETwice {kA, kB};")
    assert [one.GetName() for one in troot.class_enums("")] == ["ETwice"]


def test_two_enumerations_written_on_one_line_are_both_the_interpreter_s() -> None:
    """A line that declares is cling's top-level declaration wherever it sits, so a line
    holding two enumerations declares both."""
    process_line("enum B9 {ky = 7, kz2}; enum C9 {kz = 1};")
    assert constants("B9") == [("ky", 7), ("kz2", 8)]
    assert constants("C9") == [("kz", 1)]


def test_an_enumeration_of_a_class_is_owned_by_it_with_the_type_it_is_stored_as() -> None:
    process_line("class K { public: enum class ek : short {kOne}; };")
    made = troot.find_enum("K::ek")
    assert made is not None
    assert (made.IsScoped(), made.GetUnderlyingType()) == (True, 2)
    assert [one.GetName() for one in troot.class_enums("K")] == ["ek"]


def test_an_enumerator_written_with_an_expression_no_compiler_is_wanted_for() -> None:
    """C++'s integer division truncates towards zero, which Python's ``//`` does not, and the
    remainder follows the quotient it divides by."""
    process_line("enum EDiv {kQ = -7 / 2, kR = 7 % -2};")
    assert constants("EDiv") == [("kQ", -3), ("kR", 1)]


def test_an_enumerators_value_stops_where_the_declarations_stop_telling_it() -> None:
    """``kTRUE`` is a name ROOT's preprocessor defines, not an enumerator, so what it stands
    for is no number this table knows; the list of values ends there, and the enumeration is
    kept with the values C++ wrote and no guessed ones - a wrong value is worse than none."""
    process_line("enum EMix {kA = 1, kB = kTRUE, kC};\nenum ENone {kA = kTRUE, kB};")
    assert constants("EMix") == [("kA", 1)]
    assert troot.find_enum("ENone") is None  # not a name with values this cannot work out


def test_alignof_is_a_number_cpp_writes_into_an_enumerator() -> None:
    process_line("enum EAlign {kInt = alignof(int)};")
    assert constants("EAlign") == [("kInt", 4)]


def test_an_enumeration_with_no_name_is_no_name_to_look_up() -> None:
    """``enum {kA};`` declares constants and no name, so there is nothing ROOT's table could
    be asked for - and one declared twice over is the ``TEnum`` already there."""
    here = Where("t.C", 1)
    anonymous = EnumDecl(here, None, [("kAnonymous", None)])
    assert enums_of(Unit(here, [anonymous])) == []
    twice = EnumDecl(here, "EDup", [("kA", None)])
    notes = enums_of(Unit(here, [twice, twice]))
    assert [(note.qualified, note.items) for note in notes] == [("EDup", [("kA", 0)])]


def test_an_operation_cpp_leaves_undefined_is_no_number_and_no_exception() -> None:
    """The table's numbers come from the declarations alone, so ``1 / 0`` - which C++ leaves
    undefined - is no value here, and neither is an operation this does not know."""
    here = Where("t.C", 1)
    assert _binary_number(Binary(here, "/", number(here, 1), number(here, 0)), {}) is None
    assert _binary_number(Binary(here, "&", number(here, 1), number(here, 0)), {}) == 0
    assert _binary_number(Binary(here, "&&", number(here, 1), number(here, 0)), {}) is None
    assert _unary_number(Unary(here, "*", number(here, 2)), {}) is None  # not an operator
    assert _unary_number(Unary(here, "-", SizeOf(here, None, number(here, 2))), {}) is None
    assert _number(Literal(here, "float", 1.5, "double"), {}) is None  # an enum's is an int
    assert _number(Call(here, Name(here, ["f"]), []), {}) is None  # a call is no number


def test_the_size_or_alignment_of_a_type_no_table_here_holds_is_no_number() -> None:
    """``sizeof``/``alignof`` are numbers C++ writes into an enumerator's value for the types
    whose size is the same everywhere; a class's size is a compiler's, and is no number."""
    here = Where("t.C", 1)
    assert _size(SizeOf(here, CType("int"), None, True)) == 4  # ``alignof(int)``
    assert _size(SizeOf(here, CType("int"), None)) == 4  # ``sizeof(int)``
    assert _size(SizeOf(here, None, number(here, 2))) is None  # ``sizeof(x)``
    assert _size(SizeOf(here, CType("int", pointer=1), None)) is None  # ``sizeof(int*)``
    assert _size(SizeOf(here, CType("NoSuchClass"), None)) is None  # a class's own size


def test_a_run_from_the_cache_notes_what_the_run_that_translated_it_did() -> None:
    """A cached translation is not parsed again, so what its parse noted is kept with it:
    without that an enumeration would be the interpreter's only the first time its line ran."""
    process_line("enum ECached {kOne = 3};")
    troot._ENUMS.clear()
    process_line("enum ECached {kOne = 3};")  # the same line: its translation is in the cache
    assert constants("ECached") == [("kOne", 3)]


def test_an_enumerations_note_is_plain_json_a_library_can_keep() -> None:
    note = EnumNote("ns::E", [("kA", 0), ("kB", 5)], scoped=True, underlying="short")
    assert note.as_stored() == {"name": "ns::E", "items": [["kA", 0], ["kB", 5]],
                               "scoped": True, "underlying": "short"}  # fmt: skip
    assert EnumNote.of_stored(note.as_stored()) == note
    assert EnumNote.of_stored({"name": "E"}) == EnumNote("E", [], False, "")


def test_an_enumerator_of_a_global_enumeration_is_a_global_the_session_lists() -> None:
    """``gROOT->GetListOfGlobals()`` holds what a line declared outside every namespace, so a
    macro reading a constant through its address reads the number C++ gave it."""
    process_line("enum EGlobal {kFirst = 2, kSecond};")
    listed = {one.GetName(): one.GetAddress().value for one in ROOT.gROOT.GetListOfGlobals()}
    assert (listed["kFirst"], listed["kSecond"]) == (2, 3)
