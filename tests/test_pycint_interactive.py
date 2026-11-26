"""The C++ prompt's core: what typed lines run, what stays, and what cling would print."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from pyrootsupport import fresh
from xrdroot.pycint.declaring import translate_declaring
from xrdroot.pycint.interactive import Interpreter, Said
from xrdroot.pycint.interactive.classify import kind
from xrdroot.pycint.interactive.lexer import pending, pieces
from xrdroot.pycint.interactive.values import shown, spelled_for
from xrdroot.pycxx import cache

#: A pointer as cling prints one: an address, which differs from run to run.
ADDRESS = r"0x[0-9a-f]+"


@pytest.fixture(autouse=True)
def _fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    yield from fresh(tmp_path)


def typed(prompt: Interpreter, *lines: str) -> list[str]:
    """Every line the prompt says, errors included, in order, for ``lines`` typed in turn."""
    return [said.text for line in lines for said in prompt.feed(line)]


# -- multi-line input ------------------------------------------------------------------


def test_an_open_brace_waits_for_the_lines_that_close_it() -> None:
    prompt = Interpreter()
    assert prompt.prompt == "root [0] "
    assert typed(prompt, "int total = 0;", "for (int i = 0; i < 4; i++) {") == []
    assert prompt.prompt == "root (cont'ed, cancel with .@) [2] "
    assert typed(prompt, "  total += i;", "}") == []
    assert prompt.prompt == "root [4] "
    assert typed(prompt, "total") == ["(int) 6"]


def test_a_function_typed_over_several_lines_is_declared_once_it_closes() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int add(int a,", "        int b) {", "  return a + b;", "}") == []
    assert typed(prompt, "add(2, 3)") == ["(int) 5"]


def test_an_expression_spread_over_lines_is_printed_whole() -> None:
    prompt = Interpreter()
    assert typed(prompt, "(1 +  // one", " 2)") == ["(int) 3"]


def test_dot_at_cancels_the_lines_of_an_unfinished_input() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int j = 4;", "if (j > 3) {", "j = 0;", ".@") == []
    assert prompt.prompt == "root [4] "
    assert typed(prompt, "j") == ["(int) 4"]


def test_cancelling_drops_an_unfinished_input_as_ctrl_c_does() -> None:
    prompt = Interpreter()
    typed(prompt, "{")
    prompt.cancel()
    assert prompt.prompt == "root [1] "


def test_unnamed_blocks_and_loops_run_at_once() -> None:
    prompt = Interpreter()
    assert typed(prompt, "{", '  printf("%d\\n", 6 * 7);', "}") == []
    assert typed(prompt, 'do { printf("x\\n"); } while (false);') == []


def test_if_else_and_try_catch_are_one_statement_each() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int k = 1;", "if (k) k = 2; else k = 3;", "k") == ["(int) 2"]
    assert typed(prompt, "if (k > 5) { k = 9; } else { k = 7; }", "k") == ["(int) 7"]


# -- persistence -------------------------------------------------------------------------


def test_variables_functions_and_classes_stay_for_every_line_after() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int i = 3;", "i + 1") == ["(int) 4"]
    assert typed(prompt, "i++;", "i") == ["(int) 4"]
    assert typed(prompt, "int twice(int x) { return 2 * x; }", "twice(i)") == ["(int) 8"]
    assert typed(prompt, "class A { public: int v; A(int q) : v(q) {} };", "A a(5);") == []
    assert typed(prompt, "a.v * i") == ["(int) 20"]


def test_includes_and_using_declarations_stay(tmp_path: Path) -> None:
    (tmp_path / "helper.h").write_text("inline int helper(int a) { return a + 100; }\n")
    prompt = Interpreter(directory=tmp_path)
    assert typed(prompt, '#include "helper.h"', "helper(1)") == ["(int) 101"]
    assert typed(prompt, "helper(2)") == ["(int) 102"]
    assert typed(prompt, "using namespace std", "cout << helper(3) << endl;") == []


def test_a_declaration_runs_its_initializer_once(capsys: pytest.CaptureFixture[str]) -> None:
    prompt = Interpreter()
    typed(prompt, 'int noisy() { printf("made\\n"); return 1; }', "int made = noisy();")
    typed(prompt, "made + 1", "made + 2")
    assert capsys.readouterr().out == "made\n"


# -- value printing --------------------------------------------------------------------


def test_numbers_are_printed_with_their_type_as_cling_prints_them() -> None:
    prompt = Interpreter()
    assert typed(prompt, "3 / 2", "double d = 3.14", "float f = 2.5", "Long64_t big = 7") == [
        "(int) 1", "(double) 3.1400000", "(float) 2.50000f", "(long long) 7"
    ]  # fmt: skip
    assert typed(prompt, "'c'", "d > 3", "1e20 * 1.0") == [
        "(char) 'c'", "(bool) true", "(double) 1.0000000e+20"
    ]  # fmt: skip


def test_strings_are_printed_quoted_with_their_type() -> None:
    prompt = Interpreter()
    assert typed(prompt, 'const char *s = "abc";', "s") == ['(const char *) "abc"']
    assert typed(prompt, 'std::string text = "x"', "text") == [
        '(std::string &) "x"', '(std::string &) "x"'
    ]  # fmt: skip
    assert typed(prompt, 'TString t("123ABC")') == ['(TString &) "123ABC"[6]']


def test_pointers_are_printed_as_addresses_and_objects_as_cling_describes_them() -> None:
    prompt = Interpreter()
    said = typed(prompt, 'TH1F *p = new TH1F("p", "p", 5, 0, 1)', "p", "&*p")
    assert all(re.fullmatch(rf"\(TH1F \*\) {ADDRESS}", line) for line in said)
    assert typed(prompt, 'TNamed n("a", "b")', "*p") == [
        "(TNamed &) Name: a Title: b", "(TH1F &) Name: p Title: p NbinsX: 5"
    ]  # fmt: skip
    assert typed(prompt, "TH1F *none = nullptr", "TObject o") == [
        "(TH1F *) nullptr", "(TObject &) Name: TObject Title: Basic ROOT object"
    ]  # fmt: skip


def test_a_class_with_no_printer_is_printed_by_its_address() -> None:
    prompt = Interpreter()
    typed(prompt, "struct P { double x, y; };")
    (line,) = typed(prompt, "P pt{1.5, 2.5}")
    assert re.fullmatch(rf"\(P &\) @{ADDRESS}", line)
    assert typed(prompt, "pt.x + pt.y") == ["(double) 4.0000000"]


def test_an_assigned_object_is_printed_from_its_copy_when_its_address_cannot_be_taken() -> None:
    prompt = Interpreter()
    typed(prompt, "struct Q { int v; };", "Q a{1};", "Q b{2};")
    (line,) = typed(prompt, "a = b")
    assert re.fullmatch(rf"\(Q &\) @{ADDRESS}", line)


def test_containers_are_printed_element_by_element() -> None:
    prompt = Interpreter()
    assert typed(prompt, "std::vector<int> v{1, 2, 3}") == ["(std::vector<int> &) { 1, 2, 3 }"]
    assert typed(prompt, "std::vector<double> e") == ["(std::vector<double> &) {}"]
    assert typed(prompt, "std::map<int, int> m; m[1] = 2;", "m") == [
        "(std::map<int, int> &) { 1 => 2 }"
    ]  # fmt: skip


def test_a_value_of_a_type_the_translator_does_not_know_is_typed_by_what_it_is() -> None:
    prompt = Interpreter()
    typed(prompt, 'TH1F h("h", "h", 10, 0, 1); h.Fill(.5);')
    assert typed(prompt, "h.GetEntries()", "h.GetName()", "h.GetNbinsX()") == [
        "(double) 1.0000000", '(const char *) "h"', "(int) 10"
    ]  # fmt: skip
    (axis,) = typed(prompt, "h.GetXaxis()")
    assert re.fullmatch(rf"\(TAxis \*\) {ADDRESS}", axis)


def test_a_void_call_prints_nothing() -> None:
    prompt = Interpreter()
    assert typed(prompt, "void nothing() {}", "nothing()", "gStyle->SetOptStat(0)") == []


@pytest.mark.parametrize(
    ("value", "spelled", "line"),
    [
        (None, None, None),
        (3, "void", None),
        ("abc", "char *", '(char *) "abc"'),
        (65, "char", "(char) 'A'"),
        (True, "bool", "(bool) true"),
        (0, "bool", "(bool) false"),
        ("x", "std::string", '(std::string) "x"'),
        (["a", None, 1.5], "std::vector<std::string>",
         '(std::vector<std::string>) { "a", nullptr, 1.5000000 }'),
    ],
)  # fmt: skip
def test_each_kind_of_value_is_printed_as_cling_prints_it(
    value: object, spelled: str | None, line: str | None
) -> None:
    assert shown(value, spelled) == line


def test_an_element_with_no_printer_is_printed_by_its_address() -> None:
    assert re.fullmatch(
        rf"\(std::vector<Foo>\) \{{ {ADDRESS} \}}", shown([object()], "std::vector<Foo>") or ""
    )


@pytest.mark.parametrize(
    ("value", "spelled"),
    [(None, None), (2.5, "double"), (7, "int"), (np.bool_(True), "bool"), (np.int64(3), "int")],
)
def test_a_value_of_unknown_type_is_named_by_what_it_is(value: object, spelled: str | None) -> None:
    assert spelled_for(value) == spelled


def test_a_tstring_value_is_named_by_its_class() -> None:
    import xrdroot.pyroot as ROOT

    assert spelled_for(ROOT.TString("ab")) == "TString"
    assert shown(ROOT.TString("ab"), None) == '(TString) "ab"[2]'


def test_semicolons_suppress_what_would_otherwise_be_printed() -> None:
    prompt = Interpreter()
    assert typed(prompt, "int i = 3;", "i + 1;", "i;") == []
    assert typed(prompt, "int j = 4") == ["(int) 4"]


# -- errors, then going on ---------------------------------------------------------------


def test_an_undeclared_name_is_an_error_at_its_column_and_the_session_goes_on() -> None:
    prompt = Interpreter()
    said = prompt.feed("int i = 1; nope + i")
    assert said == [Said("ROOT_prompt_0:1:12: error: use of undeclared identifier 'nope'", True)]
    # As cling rolls back the input that failed, the i it declared is not declared either.
    assert typed(prompt, "i") == ["ROOT_prompt_1:1:1: error: use of undeclared identifier 'i'"]
    assert typed(prompt, "int i = 2;", "i") == ["(int) 2"]


def test_an_undeclared_name_from_a_macro_is_placed_where_the_macro_was_used() -> None:
    prompt = Interpreter()
    assert typed(prompt, "#define Q undefined_thing", "  Q + 1") == [
        "ROOT_prompt_1:1:3: error: use of undeclared identifier 'undefined_thing'"
    ]  # fmt: skip


def test_what_the_translator_refuses_is_reported_and_not_kept() -> None:
    prompt = Interpreter()
    (line,) = typed(prompt, "int k = ;")
    assert line.startswith("ROOT_prompt_0:1:1: error: ")
    assert typed(prompt, "int k = 2;", "k") == ["(int) 2"]


def test_a_failure_as_it_runs_is_placed_at_the_line_it_happened_on() -> None:
    prompt = Interpreter()
    (line,) = typed(prompt, "int zero = 0;", "{", "  int q = 1;", "  q = q / zero;", "}")
    assert line.startswith("ROOT_prompt_4:3:3: error: ZeroDivisionError: ")
    assert typed(prompt, "zero") == ["(int) 0"]


def test_a_failure_in_a_function_typed_earlier_is_placed_in_that_function() -> None:
    prompt = Interpreter()
    typed(prompt, "int boom(int d) {", "  return 1 / d;", "}")
    (line,) = typed(prompt, "boom(0)")
    assert line.startswith("ROOT_prompt_2:2:3: error: ZeroDivisionError")


def test_a_refused_line_run_through_process_line_is_reported_where_it_was_refused() -> None:
    prompt = Interpreter()
    (line,) = typed(prompt, 'gROOT->ProcessLine("int x = ;");')
    assert line.startswith("<ProcessLine>:2:1: error: ")


def test_a_failure_with_no_line_of_its_own_is_placed_at_the_line_that_ran_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xrdroot.pycxx.errors import MacroError

    prompt = Interpreter()

    def fail(line: str) -> None:
        raise MacroError("it failed somewhere unknown")

    monkeypatch.setattr("xrdroot.pyroot.gROOT.ProcessLine", fail, raising=False)
    assert typed(prompt, 'gROOT->ProcessLine("x");') == [
        "ROOT_prompt_0:1:1: error: it failed somewhere unknown"
    ]  # fmt: skip


def test_a_refusal_in_an_included_header_is_placed_in_the_header(tmp_path: Path) -> None:
    (tmp_path / "bad.h").write_text("int broken = ;\n")
    prompt = Interpreter(directory=tmp_path)
    (line,) = typed(prompt, '#include "bad.h"')
    assert line.startswith("bad.h:1:1: error: ")


def test_a_refusal_with_nowhere_is_placed_at_the_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    from xrdroot.pycint.interactive import unit
    from xrdroot.pycxx.errors import Refusal

    def refuse(*args: object) -> None:
        raise Refusal("no place to say")

    prompt = Interpreter()
    monkeypatch.setattr(unit, "translate_declaring", refuse)
    assert typed(prompt, "1") == ["ROOT_prompt_0:1:1: error: no place to say"]


def test_exit_ends_the_session_with_its_status() -> None:
    prompt = Interpreter()
    assert typed(prompt, "exit(3);") == []
    assert (prompt.finished, prompt.status) == (True, 3)


def test_a_failure_as_it_runs_ends_the_session_when_asked_to_as_root_x_does() -> None:
    """ROOT's ``-x`` exits on an exception: code that did not compile goes on."""
    prompt = Interpreter(exit_on_error=True)
    typed(prompt, "nope")
    assert (prompt.finished, prompt.status) == (False, 0)
    typed(prompt, "throw 3;")
    assert (prompt.finished, prompt.status) == (True, 1)


def test_an_input_of_nothing_but_a_comment_or_a_semicolon_does_nothing() -> None:
    prompt = Interpreter()
    assert typed(prompt, "// nothing", ";", "") == []


# -- the pieces of an input -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "waits"),
    [
        ("{", True),
        ('f("abc', True),
        ("/* still", True),
        ("#define LONG 1 \\", True),
        ("x)", False),
        ("int x;", False),
    ],
)
def test_an_input_waits_while_a_bracket_a_string_or_a_comment_is_open(
    text: str, waits: bool
) -> None:
    assert pending(text) is waits


def test_an_input_is_cut_where_its_statements_end() -> None:
    cut = pieces('TH1F h("h","h",10,0,1); h.Fill(.5); h.GetEntries() // done')
    assert [(piece.text, piece.column, piece.end) for piece in cut] == [
        ('TH1F h("h","h",10,0,1);', 1, "semicolon"),
        ("h.Fill(.5);", 25, "semicolon"),
        ("h.GetEntries()", 37, "open"),
    ]


def test_classes_initializers_and_blocks_end_where_cpp_ends_them() -> None:
    def texts(text: str) -> list[str]:
        return [piece.text for piece in pieces(text)]

    assert texts("struct S {} s; int y") == ["struct S {} s;", "int y"]
    assert texts("std::vector<int> v{1,2}; v") == ["std::vector<int> v{1,2};", "v"]
    assert texts("void f() { }\nf()") == ["void f() { }", "f()"]
    assert texts("namespace N { int a; } { int b; }") == ["namespace N { int a; }", "{ int b; }"]
    assert texts("auto l = [](int q) { return q; }; l(2)") == [
        "auto l = [](int q) { return q; };", "l(2)"
    ]  # fmt: skip
    assert texts("do { i++; } while (i < 3); i") == ["do { i++; } while (i < 3);", "i"]


def test_preprocessor_lines_are_pieces_of_their_own() -> None:
    cut = pieces("#include <vector>\n  #define X 3\nint v = X")
    assert [(piece.text, piece.line, piece.column, piece.end) for piece in cut] == [
        ("#include <vector>", 1, 1, "directive"),
        ("#define X 3", 2, 3, "directive"),
        ("int v = X", 3, 1, "open"),
    ]
    inside = pieces("f(1,\n#if 1\n2)\n")
    assert [piece.end for piece in inside] == ["open"]


@pytest.mark.parametrize(
    ("text", "what", "name"),
    [
        ('TH1F h("h", "h", 10, 0, 1)', "declaration", "h"),
        ("h.Fill(.5)", "statement", None),
        ("f(3)", "statement", None),
        ("a * b", "declaration", "b"),
        ("unsigned i = 3", "declaration", "i"),
        ("unsigned int u", "declaration", "u"),
        ("std::map<int, std::vector<int>> m", "declaration", "m"),
        ("std::vector<int>::iterator it", "declaration", "it"),
        ("cout << x", "statement", None),
        ("x < 3", "statement", None),
        ("a < b; c", "statement", None),
        ("new TH1F()", "statement", None),
        ("bool operator<(int a, int b) { return a; }", "declaration", None),
        ("using namespace std", "declaration", None),
        ("int(3.5)", "statement", None),
        ("*p", "statement", None),
        ("#include <x>", "directive", None),
        ("TH1F *h", "declaration", "h"),
        ("::data *a = new ::data", "declaration", "a"),
        ("::f(3)", "statement", None),
        ("const TH1F &r = h", "declaration", "r"),
        ("x", "statement", None),
    ],
)
def test_a_piece_is_a_declaration_when_cpp_reads_it_as_one(
    text: str, what: str, name: str | None
) -> None:
    found = kind(pieces(text)[0])
    assert (found.what, found.name) == (what, name)


# -- the translator's part -----------------------------------------------------------------


def test_the_translator_says_the_type_of_each_variable_declared_as_cling_spells_it() -> None:
    source = (
        'const char *s = "a"; std::vector<double> v; extern int ex[]; std::array<int, 3> ar;'
        " auto q = gRandom->Gaus(); std::string str; TH1F **pp = nullptr;"
    )
    _, declared = translate_declaring(source, "ROOT_prompt")
    assert declared == {
        "s": ("s", "const char *"),
        "v": ("v", "std::vector<double>"),
        "ex": ("ex", "int[]"),
        "ar": ("ar", "std::array<int, 3>"),
        "q": ("q", None),
        "str": ("str_", "std::string"),
        "pp": ("pp", "TH1F **"),
    }


def test_only_a_macro_the_session_read_is_remembered_for_dot_typedef(tmp_path):
    """``.typedef`` lists what the sources of the session declare, so ``.L``/``.x`` remember
    theirs - and a library or a path that is not there is not one of them."""
    from xrdroot.pycint.interactive.commands import SOURCES, remember_source

    kept = dict(SOURCES)
    macro = tmp_path / "t.C"
    macro.write_text("typedef int word;\n")
    try:
        remember_source(macro)
        assert str(macro.resolve()) in SOURCES
        library = tmp_path / "libt.so"
        library.write_text("")
        remember_source(library)  # a library's types are ROOT's compiled ones
        remember_source(tmp_path / "gone.C")  # a macro that is not there
        assert list(SOURCES) == [*kept, str(macro.resolve())]
    finally:
        SOURCES.clear()
        SOURCES.update(kept)


def test_a_byte_that_is_no_character_is_echoed_as_its_number() -> None:
    """``char`` values outside printable ASCII have no character to quote - ``-128`` would be
    ``chr()`` of a negative number - so they are written as the number a compiler assigns to a
    ``char``, which is what the line is re-parsed as; printable ones and the control escapes keep
    their quotes."""
    from xrdroot.pycint.interactive.values import _character

    assert _character(65) == "'A'"
    assert _character(-128) == "-128" and _character(200) == "200"
    assert [_character(one) for one in (10, 9, 0, 92, 39)] == [
        "'\\n'", "'\\t'", "'\\0'", "'\\\\'", "'\\''"
    ]
    assert _character("x") == "'x'"
