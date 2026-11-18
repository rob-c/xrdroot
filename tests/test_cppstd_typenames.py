"""C++ type names as ROOT spells them: ``TClassEdit::ShortType`` and ``GetNameForIO``.

The names are ROOT's own: what ``roottest/root/meta/execString.cxx`` and
``roottest/root/io/uniquePointer/checkNameForIO.C`` ask for, which is what a
class is called by and what a member is read and written as.
"""

from __future__ import annotations

import xrdroot.pyroot as ROOT
from xrdroot.cppstd import typenames

#: A ``std::string`` spelled the long way, as a rootmap or a branch title spells it.
LONG_STRING = "basic_string<char,char_traits<char>,allocator<char> >"


def test_a_type_name_is_its_name_and_its_template_arguments() -> None:
    parsed = typenames.parse("std::vector<std::pair<int,double> >")
    assert (parsed.name, len(parsed.arguments)) == ("std::vector", 1)
    pair = parsed.arguments[0]
    assert (pair.name, [argument.name for argument in pair.arguments]) == (
        "std::pair", ["int", "double"])
    assert str(parsed) == "std::vector<std::pair<int,double> >"
    assert typenames.parse("  int  ").name == "int"
    assert typenames.parse("T*").name == "T*"


def test_each_argument_is_split_where_its_own_commas_are_not() -> None:
    assert typenames._arguments("int,vector<int,double>,char") == [
        "int", "vector<int,double>", "char"]
    assert typenames._arguments("int") == ["int"]
    assert typenames._matching("a<b<c>>", 1) == 6  # the outer bracket closes last


def test_a_modifier_after_the_name_is_kept_as_written() -> None:
    parsed = typenames.parse("vector<int>*")
    assert (parsed.name, parsed.modifiers, str(parsed)) == ("vector", "*", "vector<int>*")
    assert typenames.parse("const T*").name == "const T*"


def test_short_type_shortens_the_standards_own_long_spellings() -> None:
    """ROOT shortens a ``basic_string`` to ``string`` and drops the arguments that are its
    defaults, which is how a class of that type is named."""
    assert typenames.short_type(LONG_STRING) == "string"
    assert typenames.short_type(f"list<{LONG_STRING}") == "list<string>"
    assert typenames.short_type("basic_string<char,char_traits<char> >") == "string"
    assert typenames.short_type(
        "std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >"
    ) == "std::string"
    assert typenames.short_type(
        f"std::vector<{LONG_STRING}, std::allocator<{LONG_STRING}> >", typenames.DROP_STD
    ) == "vector<string>"
    assert typenames.short_type(f"vector<{LONG_STRING}>") == "vector<string>"
    assert typenames.short_type("vector<int>") == "vector<int>"
    assert typenames.short_type("int") == "int"


def test_two_closing_brackets_are_written_apart_as_cpp_had_to_write_them() -> None:
    assert typenames.short_type(
        " map<basic_string<char,char_traits<char> >,map<basic_string<char,char_traits<char>,"
        "std::allocator<char> >,  basic_string<char,std::char_traits<char>,allocator<char> > >",
        False) == "map<string,map<string,string> >"  # fmt: skip


def test_the_name_a_unique_ptr_member_is_read_by_is_a_pointer() -> None:
    """``GetNameForIO``: a ``unique_ptr<T>`` inside standard containers reads as a ``T*``, and
    one inside a template ROOT knows nothing about keeps its own name."""
    for written, read in [("unique_ptr<T>", "T*"), ("vector<unique_ptr<T>>", "vector<T*>"),
                          ("list<vector<unique_ptr<T>>>", "list<vector<T*> >"),
                          ("A<unique_ptr<T>>", "A<unique_ptr<T> >"),
                          ("A<vector<unique_ptr<T>>>", "A<vector<unique_ptr<T> > >"),
                          ("B<A<vector<unique_ptr<T>>>>", "B<A<vector<unique_ptr<T> > > >"),
                          ("std::vector<unique_ptr<TrackingRegion> >",
                           "std::vector<TrackingRegion*>"),
                          ("edm::Wrapper<vector<unique_ptr<TrackingRegion> > >",
                           "edm::Wrapper<vector<unique_ptr<TrackingRegion> > >")]:  # fmt: skip
        assert typenames.name_for_io(written) == read


def test_tclassedit_answers_with_the_flags_a_macro_passes_it() -> None:
    assert ROOT.TClassEdit.ShortType(f"list<{LONG_STRING}", 0) == "list<string>"
    assert ROOT.TClassEdit.ShortType("vector<int>") == "vector<int>"
    assert ROOT.TClassEdit.GetNameForIO("vector<unique_ptr<T>>") == "vector<T*>"
    assert ROOT.TClassEdit.CleanType("  int ") == "int"
    assert ROOT.TClassEdit.kDropAllDefault == 0x100 and ROOT.TClassEdit.kDropStd == 0x10
    assert (ROOT.TClassEdit.kNotSTL, ROOT.TClassEdit.kVector) == (1, 2)


def test_a_ta_class_is_named_by_the_short_spelling_of_its_type() -> None:
    """``TClass::GetClass`` finds the one class through any spelling of its name, and calls it
    the short one - as ROOT does."""
    for asked, called in [(LONG_STRING, "string"), ("string", "string"),
                          ("std::string", "string"), ("vector<string>", "vector<string>"),
                          ("std::vector<int>", "vector<int>"), ("TH1F", "TH1F")]:
        assert ROOT.TClass.GetClass(asked).GetName() == called


def test_the_interpreter_echoes_a_type_name_it_is_asked_about() -> None:
    """``GetInterpreterTypeName(name, output)``: a name here is already the one the translator
    resolved, so it goes into ``output`` as it came."""
    from xrdroot.pyroot.stl import std

    held = std.string()
    assert ROOT.gInterpreter.GetInterpreterTypeName("list<string>", held) == "list<string>"
    assert str(held) == "list<string>"
    cell = type("Cell", (), {"value": ""})()
    assert ROOT.gInterpreter.GetInterpreterTypeName("vector<int>", cell) == "vector<int>"
    assert cell.value == "vector<int>"
    assert ROOT.gInterpreter.GetInterpreterTypeName("list<string>", 1) == "list<string>"
    assert ROOT.gInterpreter.GetMapfile() is None
