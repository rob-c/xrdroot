"""The expression language itself: its words, its grammar, its names and its refusals.

The arithmetic cases are go-hep's ``rexpr`` scalar tests, evaluated here over
a batch of one entry; the refusals are ROOT's own cases of text that is not an
expression, with the name of what was wrong in each message.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from xrdroot import Formula, FormulaError, UnsupportedFeatureError, compile_formula
from xrdroot.formula.lexer import tokenize
from xrdroot.formula.names import Names

SCALARS = {"x": np.array([3.0]), "y": np.array([4.0]), "eta": np.array([-2.7])}


def one(text: str, columns=SCALARS) -> float:
    return compile_formula(text, columns).evaluate(columns).tolist()[0]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("1", 1),
        ("1 + 2*3", 7),
        ("(1+2)*3", 9),
        ("x", 3),
        ("x + y", 7),
        ("x*x + y*y", 25),
        ("sqrt(x*x + y*y)", 5),
        ("hypot(x, y)", 5),
        ("-x", -3),
        ("x > y", 0),
        ("x < y", 1),
        ("x == 3 && y == 4", 1),
        ("x == 3 && y == 5", 0),
        ("x == 9 || y == 4", 1),
        ("!(x > y)", 1),
        ("abs(eta) < 2.5", 0),
        ("TMath::Abs(eta) < 3", 1),
        ("pi > 3", 1),
        ("max(x, y)", 4),
        ("min(x, y, 1)", 1),
        ("pow(x, 2)", 9),
        ("int(2.7)", 2),
        ("fmod(7, 4)", 3),
        ("x > y ? 1 : 2", 2),
        ("x < y ? x : y", 3),
        ("x^2 + y^2", 25),
        ("-x^2", -9),
        ("2^-1", 0.5),
        ("3/2", 1.5),
        ("7 % 4", 3),
        ("-7 % 4", -3),
        ("7.9 % 4", 3),
        ("x << 2", 12),
        ("16 >> 2", 4),
        ("1 << 64", 0),
        ("6 & 3", 2),
        ("6 | 3", 7),
        ("~0", -1),
        ("(int)x/2", 1.5),
        ("(unsigned int)(-1) > 0", 1),
        ("(bool)x + (bool)0", 1),
        ("(float)0.1 == 0.1", 0),
        ("(double)x", 3),
        ("Long64_t(2.5)", 2),
        ("0x1F", 31),
        ("0x1fUL", 31),
        ("1e3", 1000),
        ("2.5f", 2.5),
        (".5", 0.5),
        ("5.", 5),
        ("10u", 10),
        ("true + kTRUE + false", 2),
        ("x > 1 && y > 1 || 0", 1),
        ("1 + 2 == 3", 1),
        ("1 & 3 == 3", 1),
        ("+x", 3),
    ],
)
def test_arithmetic_is_what_root_s_formulas_make_of_it(text, want):
    assert one(text) == pytest.approx(want)


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("TMath::Sqrt(16)", 4),
        ("TMath::Power(2, 10)", 1024),
        ("TMath::Exp(0) + TMath::Log(1) + TMath::Log10(100)", 3),
        ("TMath::Log2(8)", 3),
        ("TMath::Sin(0) + TMath::Cos(0) + TMath::Tan(0)", 1),
        ("TMath::ASin(1) * 2 - TMath::Pi()", 0),
        ("TMath::ACos(1) + TMath::ATan(0)", 0),
        ("TMath::ATan2(1, 1) * 4 - TMath::Pi()", 0),
        ("TMath::SinH(0) + TMath::CosH(0) + TMath::TanH(0)", 1),
        ("TMath::ASinH(0) + TMath::ACosH(1) + TMath::ATanH(0)", 0),
        ("TMath::Min(3, 1, 2) + TMath::Max(3, 1)", 4),
        ("TMath::Floor(-1.5) + TMath::Ceil(1.2)", 0),
        ("TMath::Nint(2.5) + TMath::Nint(3.5)", 6),
        ("round(2.5) + round(-2.5)", 0),
        ("round(2.4)", 2),
        ("TMath::Sign(3, -1) + TMath::Sign(-2, 0)", -1),
        ("TMath::Hypot(3, 4)", 5),
        ("TMath::Erf(0) + TMath::Erfc(0)", 1),
        ("TMath::Gamma(5)", 24),
        ("TMath::LnGamma(1)", 0),
        ("TMath::Gaus(0)", 1),
        ("TMath::Gaus(1, 1, 2)", 1),
        ("TMath::Gaus(0, 0, 1, 1)", 1 / math.sqrt(2 * math.pi)),
        ("TMath::Gaus(0, 0, 0)", 1e30),
        ("TMath::BreitWigner(0)", 2 / math.pi),
        ("TMath::BreitWigner(5, 5, 2)", 1 / math.pi),
        ("TMath::IsNaN(0/0.) + TMath::Finite(1/0.)", 1),
        ("TMath::Even(4) + TMath::Odd(4) + TMath::Odd(3)", 2),
        ("TMath::TwoPi() - 2*TMath::Pi()", 0),
        ("TMath::PiOver2() + TMath::PiOver4() - 0.75*TMath::Pi()", 0),
        ("TMath::InvPi() * TMath::Pi()", 1),
        ("TMath::E() - exp(1)", 0),
        ("TMath::Ln10() - log(10) + TMath::LogE() - log10(TMath::E())", 0),
        ("TMath::Sqrt2() ^ 2", 2),
        ("TMath::DegToRad() * TMath::RadToDeg()", 1),
        ("TMath::C()", 299792458),
        ("TMath::Sq(3) + sq(2)", 13),
        ("fabs(-2) + std::abs(-1) + cbrt(8)", 5),
        ("exp2(3) + log2(4)", 10),
        ("asin(0) + acos(1) + atan(0) + atan2(0, 1)", 0),
        ("sinh(0) + cosh(0) + tanh(0) + asinh(0) + acosh(1) + atanh(0)", 1),
        ("sin(0) + cos(0) + tan(0)", 1),
        ("floor(1.5) + ceil(1.5) + trunc(-1.5) + rint(2.5)", 4),
        ("fmin(1, 2) + fmax(1, 2)", 3),
        ("erf(0) + erfc(0) + tgamma(4) + lgamma(1)", 7),
        ("isnan(0/0.) + isinf(1/0.) + isfinite(1)", 3),
        ("std::sqrt(9) + std::pow(2, 2)", 7),
    ],
)
def test_the_functions_are_tmath_s_and_c_s(text, want):
    assert one(text) == pytest.approx(want)


def test_words_are_cut_as_a_c_plus_plus_compiler_cuts_them():
    tokens = tokenize("evt.P3.Px>=TMath::Abs(x[0])&&Sum$(y)")
    assert [(t.kind, t.text) for t in tokens] == [
        ("name", "evt.P3.Px"),
        ("op", ">="),
        ("name", "TMath::Abs"),
        ("op", "("),
        ("name", "x"),
        ("op", "["),
        ("number", "0"),
        ("op", "]"),
        ("op", ")"),
        ("op", "&&"),
        ("name", "Sum$"),
        ("op", "("),
        ("name", "y"),
        ("op", ")"),
        ("end", ""),
    ]
    assert repr(tokens[0]) == "<Token name 'evt.P3.Px' at 0>"


def test_a_character_no_expression_has_is_refused_where_it_is():
    with pytest.raises(FormulaError, match=r"'x # 2' has '#' at character 2"):
        compile_formula("x # 2", SCALARS)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "an empty expression"),
        ("x +", "could not be parsed: a number, a name or '\\(' was expected at character 3"),
        ("(x", "a '\\)' was expected"),
        ("x y", "an operator or the end was expected"),
        ("x ? 1", "a ':' was expected"),
        ("x[0", "a '\\]' was expected"),
        (".size", "a number, a name"),
        (")", "a number, a name"),
        ("nosuchfunc(x)", "'nosuchfunc' in 'nosuchfunc\\(x\\)' is not a function"),
        ("sqrt(1, 2)", "sqrt takes 1 argument, and 'sqrt\\(1, 2\\)' gives it 2"),
        ("Sum$()", "Sum\\$ takes 1 argument"),
        ("Alt$(x)", "Alt\\$ takes 2 arguments"),
        ("TMath::Gaus()", "TMath::Gaus takes 1 to 4 arguments"),
        ("int(1, 2)", "int takes 1 argument"),
        ("Nope$(x)", "'Nope\\$' in 'Nope\\$\\(x\\)' is not a function"),
        ("Nope$", "'Nope\\$' in 'Nope\\$' is not one of ROOT's special names"),
        ("nope + 1", "'nope' in 'nope \\+ 1' is not a branch, an alias"),
        ("xx", "the nearest are 'x'"),
    ],
)
def test_text_that_is_not_an_expression_is_refused_saying_why(text, message):
    with pytest.raises(FormulaError, match=message):
        compile_formula(text, SCALARS)


def test_an_unknown_name_is_refused_with_the_branches_there_are_when_none_is_near():
    names = [f"branch{i}" for i in range(10)]
    with pytest.raises(FormulaError, match=r"there is 'branch0', .* and 2 more"):
        compile_formula("zzz", names)
    with pytest.raises(FormulaError, match="there are no branches to read"):
        compile_formula("zzz", [])


def test_cplusplus_this_does_not_evaluate_is_refused_by_name():
    names = {"pt": 1, "evt": 0, "P3": 0}
    with pytest.raises(UnsupportedFeatureError, match="calls GetX\\(\\) on 'pt'"):
        compile_formula("pt.GetX()", names)
    with pytest.raises(UnsupportedFeatureError, match="calls at\\(\\) on 'pt'"):
        compile_formula("pt[0].at()", names)
    with pytest.raises(UnsupportedFeatureError, match="asks for 'Px' inside 'P3'"):
        compile_formula("P3.Px", names)
    with pytest.raises(UnsupportedFeatureError, match="puts '@' in front of something"):
        compile_formula("@pt", names)
    with pytest.raises(FormulaError, match=r"pt.size\(\) takes 0 arguments"):
        compile_formula("pt.size(1)", names)


def test_names_resolve_to_the_longest_branch_they_spell():
    names = Names(["evt", "P3", "P3.Px", "ArrayI16[10]", "friend.x", "x"])
    assert names.resolve("evt.P3.Px") == ("P3.Px", "")
    assert names.resolve("P3.Px") == ("P3.Px", "")
    assert names.resolve("P3.Py") == ("P3", "Py")
    assert names.resolve("ArrayI16") == ("ArrayI16[10]", "")
    assert names.resolve("friend.x") == ("friend.x", "")
    assert names.resolve("other.x") is None
    assert names.resolve("evt.nothing") == ("evt", "nothing")


def test_a_formula_says_which_branches_it_needs_once_each_in_order():
    names = {"pt": 1, "eta": 1, "x": 0, "mu.pt": 1}
    for text, want in [
        ("pt", ("pt",)),
        ("pt + eta", ("pt", "eta")),
        ("pt + pt", ("pt",)),
        ("Sum$(pt) > 10", ("pt",)),
        ("pt[x]", ("pt", "x")),
        ("mu.pt", ("mu.pt",)),
        ("Entry$ > 5", ()),
        ("sqrt(x)", ("x",)),
        ("@pt.size() + eta.size()", ("pt", "eta")),
    ]:
        formula = compile_formula(text, names)
        assert formula.branches == want
        assert formula.text == text
    assert repr(compile_formula("pt + eta", names)) == "<Formula 'pt + eta' of pt, eta>"
    assert repr(compile_formula("1", names)) == "<Formula '1' of no branches>"


def test_aliases_are_expanded_wherever_they_are_written():
    names = {"px": 0, "py": 0, "pt": 1}
    aliases = {"ptsq": "px*px + py*py", "mag": "sqrt(ptsq)", "jets": "pt"}
    formula = compile_formula("mag + jets[0]", names, aliases=aliases)
    assert formula.branches == ("px", "py", "pt")
    columns = {"px": np.array([3.0]), "py": np.array([4.0]), "pt": np.array([[1.0, 2.0]])}
    assert formula.evaluate(columns).tolist() == [6.0]
    with pytest.raises(UnsupportedFeatureError, match="indexes an alias that stands for"):
        compile_formula("mag[0]", names, aliases=aliases)


def test_an_alias_that_stands_for_itself_is_refused_with_the_way_round():
    aliases = {"a": "b + 1", "b": "2 * a"}
    with pytest.raises(FormulaError, match="the alias 'a' stands for itself, through a -> b -> a"):
        compile_formula("a", {}, aliases=aliases)


def test_a_cast_in_brackets_is_a_branch_when_a_branch_is_called_that():
    columns = {"int": np.array([2.5])}
    assert compile_formula("(int) * 2", columns).evaluate(columns).tolist() == [5.0]
    with pytest.raises(FormulaError, match="an operator or the end was expected"):
        compile_formula("(int)2.5", columns)
    assert compile_formula("(int)2.5", []).evaluate({}, rows=1).tolist() == [2]


def test_strings_escape_as_c_writes_them():
    columns = {"s": ["a\tb", 'say "hi"', "c"]}
    formula = compile_formula('s == "a\\tb" || s == \'say "hi"\'', columns)
    assert formula.evaluate(columns).tolist() == [True, True, False]


def test_whether_it_loops_is_known_only_when_the_names_say_how_deep_they_go():
    assert compile_formula("pt * 2", {"pt": 1}).per_element
    assert not compile_formula("pt[0] * 2", {"pt": 1}).per_element
    assert not compile_formula("Sum$(pt)", {"pt": 1}).per_element
    assert compile_formula("pt", {"pt": np.zeros((3, 2))}).per_element
    assert not compile_formula("Iteration$", {"pt": 1}).per_element
    with pytest.raises(ValueError, match="compiled with names alone"):
        compile_formula("pt", ["pt"]).per_element  # noqa: B018


def test_a_bad_index_is_refused_when_the_dimensions_are_known():
    with pytest.raises(FormulaError, match="'pt' has 1 dimension and is given 2 indices"):
        compile_formula("pt[0][1]", {"pt": 1})
    with pytest.raises(FormulaError, match="'x' has 0 dimensions and is given 1 indices"):
        compile_formula("x[0]", {"x": 0})
    with pytest.raises(FormulaError, match="'pt' indexed 1 times is one value"):
        compile_formula("pt[0].size()", {"pt": 1})
    with pytest.raises(UnsupportedFeatureError, match="ROOT nests those loops"):
        compile_formula("m[][idx]", {"m": 2, "idx": 1})


def test_a_formula_is_exported_from_the_package():
    assert isinstance(compile_formula("1", []), Formula)
