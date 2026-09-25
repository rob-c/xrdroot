"""Cutting a macro into tokens and preprocessing it as Cling does."""

from __future__ import annotations

from pathlib import Path

import pytest

from xrdroot.cint import Refusal, Where
from xrdroot.cint.literals import number
from xrdroot.cint.preprocessor import preprocess
from xrdroot.cint.tokens import tokenize, unescape


def words(text: str, file: str = "<macro>") -> str:
    tokens, _ = preprocess(text, file)
    return " ".join(token.text for token in tokens)


def test_tokens_know_their_kind_line_and_whether_a_line_starts_with_them() -> None:
    tokens = tokenize('int x = 0x1F; // comment\n/* a\nb */ s = R"(raw\n)" "a\\n";\n#if')
    assert [(t.kind, t.text, t.where.line, t.bol) for t in tokens[:3]] == [
        ("id", "int", 1, True),
        ("id", "x", 1, False),
        ("op", "=", 1, False),
    ]
    assert tokens[3].kind == "num"
    assert [t.where.line for t in tokens if t.kind == "str"] == [3, 4]
    assert tokens[-3].bol and not tokens[-2].bol and tokens[-2].text == "if"
    assert tokens[-1].kind == "eof"
    assert repr(tokens[0]) == "<id 'int' <macro>:1>"


def test_a_backslash_newline_joins_lines_and_keeps_the_later_line_numbers() -> None:
    tokens = tokenize("#define A \\\n  1\nB\\\n")
    assert [(t.text, t.where.line) for t in tokens[:-1]] == [
        ("#", 1),
        ("define", 1),
        ("A", 1),
        ("1", 1),
        ("B", 3),
    ]
    assert tokenize("")[0].kind == "eof"


def test_a_character_no_cpp_is_written_with_is_refused() -> None:
    with pytest.raises(Refusal, match=r"<macro>:2: '`' is not a character"):
        tokenize("int a;\n`")


@pytest.mark.parametrize(
    ("literal", "text"),
    [
        (r'"a\n\t\\\"b"', 'a\n\t\\"b'),
        (r"'\0'", "\0"),
        (r"'\x41'", "A"),
        (r'"é\101"', "éA"),
        ('R"x(a\\nb)x"', "a\\nb"),
        (r'u8"\q"', "q"),
        (r"'\e'", "\x1b"),
    ],
)
def test_escapes_are_what_cpp_makes_of_them(literal: str, text: str) -> None:
    assert unescape(literal) == text


@pytest.mark.parametrize(
    ("text", "value", "ctype"),
    [
        ("10", 10, "int"),
        ("10u", 10, "unsigned int"),
        ("10UL", 10, "unsigned long"),
        ("7LL", 7, "long long"),
        ("0x1F", 31, "int"),
        ("0b101", 5, "int"),
        ("017", 15, "int"),
        ("0", 0, "int"),
        ("1'000", 1000, "int"),
        ("1.5f", 1.5, "float"),
        (".5", 0.5, "double"),
        ("2.", 2.0, "double"),
        ("1e-3", 0.001, "double"),
        ("0x1p4", 16.0, "double"),
    ],
)
def test_number_literals_are_worth_what_cpp_says(text: str, value: float, ctype: str) -> None:
    assert number(text, Where("m.C", 1)) == (value, ctype)


def test_what_is_no_number_is_refused_and_a_user_literal_named() -> None:
    with pytest.raises(Refusal, match="1_GeV is a user-defined literal"):
        number("1_GeV", Where("m.C", 1))
    with pytest.raises(Refusal, match="1.2.3 is not a number literal"):
        number("1.2.3", Where("m.C", 1))


def test_macros_expand_as_the_preprocessor_expands_them() -> None:
    source = """
    #define N 10
    #define SQ(x) ((x) * (x))
    #define CAT(a, b) a##b
    #define STR(a) #a
    #define LOG(fmt, ...) printf(fmt, __VA_ARGS__)
    #define NONE() 0
    #define SELF SELF + 1
    int CAT(x, 1) = SQ(N + 1); const char *s = STR(a  "q" b); LOG("%d %d", 1, 2); NONE();
    SELF; SQ; CAT(x, );
    #undef N
    N __LINE__ __FILE__
    """
    assert words(source, "m.C") == (
        "int x1 = ( ( 10 + 1 ) * ( 10 + 1 ) ) ; const char * s = \"a \\\"q\\\" b\" ; "
        'printf ( "%d %d" , 1 , 2 ) ; 0 ; SELF + 1 ; SQ ; x ; N 12 "m.C"'
    )


@pytest.mark.parametrize(
    ("condition", "kept"),
    [
        ("defined(__CLING__) && !defined __CINT__", True),
        ("defined(__ROOTCLING__)", False),
        ("ROOT_VERSION_CODE >= ROOT_VERSION(6, 40, 0)", True),
        ("1 ? 0 : 1", False),
        ("(2 + 3) * 4 == 20 && 7 / 2 == 3 && -7 % 2 == -1", True),
        ("1 << 3 == 8 && (5 >> 1) == 2 && (6 & 3) == 2 && (4 | 1) == 5 && (6 ^ 3) == 5", True),
        ("~0 == -1 && +1 && 1 != 2 && 2 > 1 && 1 < 2 && 1 <= 1 && 1 >= 1 || 0", True),
        ("UNDEFINED_NAME || false", False),
        ("true", True),
        ("'A' == 65 && 1 / 0 == 0", True),
        ('__has_include("missing.h")', False),
        ("__has_include(<vector>)", True),
    ],
)
def test_conditions_are_decided_as_cling_decides_them(condition: str, kept: bool) -> None:
    assert words(f"#if {condition}\nyes\n#else\nno\n#endif\n") == ("yes" if kept else "no")


def test_conditional_sections_nest_and_chain() -> None:
    source = """
    #ifdef __CLING__
    #  if 0
    a
    #  elif 1
    b
    #  elif 1
    c
    #  else
    d
    #  endif
    #else
    #  if 1
    e
    #  endif
    #endif
    #ifndef __CLING__
    f
    #elif 1
    g
    #endif
    #pragma once
    #line 7
    #warning ignored
    #undef
    #
    """
    assert words(source) == "b g"


@pytest.mark.parametrize(
    ("source", "why"),
    [
        ("#if 1\nx\n", "an #if is never closed"),
        ("#endif\n", "#endif has no #if to belong to"),
        ("#else\n", "#else has no #if"),
        ("#elif 1\n", "#elif has no #if"),
        ("#if\n#endif\n", "an #if has no condition"),
        ("#if 1 +\n#endif\n", "ends before its expression"),
        ("#if (1\n#endif\n", "opens a bracket it does not close"),
        ("#if 1 ? 2\n#endif\n", "has a \\? without its :"),
        ("#if 1 2\n#endif\n", "has '2' left over"),
        ('#if "s"\n#endif\n', "cannot be part of an #if condition"),
        ("#if defined(\n#endif\n", r"defined\( \) must hold one name"),
        ("#if defined\n#endif\n", "defined must be followed by a name"),
        ("#define\n", "#define names no macro"),
        ("#define F(a, b\n", "parameter list is never closed"),
        ("#define F(a) a\nF(1, 2)\n", r"F\(\) is a macro of 1 arguments, used with 2"),
        ("#define F(a) a\nF(1\n", "never closed by a \\)"),
        ("#define P(a, b) a ## b\nP(+, /)\n", "which is not one C\\+\\+ token"),
        ("#error stop here\n", "stops itself with #error stop here"),
    ],
)
def test_what_the_preprocessor_cannot_make_sense_of_is_refused(source: str, why: str) -> None:
    with pytest.raises(Refusal, match=why):
        preprocess(source)


def test_a_local_header_is_read_in_and_a_system_one_left_to_root(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.h").write_text("int inner;\n")
    (tmp_path / "local.h").write_text('#include "sub/inner.h"\nint local;\n#include <cmath>\n')
    macro = tmp_path / "m.C"
    tokens, included = preprocess(
        '#include "local.h"\n#include "TH1.h"\n#include LATER\nint main;', str(macro)
    )
    assert " ".join(t.text for t in tokens) == "int inner ; int local ; int main ;"
    assert [path.name for path in included] == ["local.h", "inner.h"]


def test_an_include_that_includes_itself_forever_is_refused(tmp_path: Path) -> None:
    (tmp_path / "loop.h").write_text('#include "loop.h"\n')
    with pytest.raises(Refusal, match="nests deeper than 40 files"):
        preprocess('#include "loop.h"\n', str(tmp_path / "m.C"))


def test_a_header_in_latin_1_is_still_read(tmp_path: Path) -> None:
    (tmp_path / "old.h").write_bytes(b'const char *s = "\xe9";\n')
    tokens, _ = preprocess('#include "old.h"\n', str(tmp_path / "m.C"))
    assert tokens[-2].text == '"é"'
