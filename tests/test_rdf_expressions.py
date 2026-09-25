"""The C++ expressions ``Define`` and ``Filter`` take as strings, evaluated a batch at a time.

The arithmetic is C++'s - integers divide as C divides them, ``float`` stays
``float``, ``^`` is exclusive or - and collections are ``RVec``\\ s, compared
into ``RVec<int>`` and indexed by masks. What C++ evaluates conditionally is
evaluated conditionally: the side of ``&&`` that is skipped, and the branch of
``? :`` that is not taken, are never computed for the entries that skip them.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import FormulaError, Jagged, UnsupportedFeatureError
from xrdroot.rdf.expression import Expression, Scope
from xrdroot.rdf.values import from_column, to_column

COLUMNS = {
    "pt": Jagged(np.array([40.0, 12.5, 33.0, 5.0, 50.0], np.float32), [0, 2, 2, 3, 5]),
    "eta": Jagged(np.array([0.5, -1.0, 2.0, 0.1, -0.3], np.float32), [0, 2, 2, 3, 5]),
    "q": Jagged(np.array([1, -1, 1, -1, -1], np.int32), [0, 2, 2, 3, 5]),
    "other": Jagged(np.array([1.0, 2.0, 3.0, 4.0]), [0, 1, 2, 3, 4]),
    "n": np.array([2, 0, 1, 2], np.int32),
    "x": np.array([1.5, -2.5, np.nan, 4.0]),
    "f": np.array([1.0, 2.0, 3.0, 4.0], np.float32),
    "small": np.array([100, 100, -100, 1], np.int8),
    "u": np.array([1, 2, 3, 4], np.uint32),
    "big": np.array([1, 2, 3, 4], np.uint64),
    "flag": np.array([True, False, True, False]),
    "s": ["a", "bb", "a", ""],
    "grid": np.arange(8.0).reshape(4, 2),
    "double": np.array([7.0, 8.0, 9.0, 10.0]),
}


def value(text: str, columns: dict | None = None):
    given = COLUMNS if columns is None else columns
    count = len(next(iter(given.values())))
    expression = Expression(text, list(given))
    scope = Scope(lambda name: from_column(name, given[name]), count, np.arange(10, 10 + count))
    return to_column(expression.evaluate(scope), count)


def listed(text: str):
    found = value(text)
    return found.tolist() if hasattr(found, "tolist") else found


@pytest.mark.parametrize(
    ("text", "dtype"),
    [
        ("n * 0 + 1", np.int32),
        ("n * 0 + 1u", np.uint32),
        ("n * 0 + 1l", np.int64),
        ("n * 0 + 1ull", np.uint64),
        ("n * 0 + 3000000000", np.int64),
        ("n * 0 + 0x1fu", np.uint32),
        ("f * 2", np.float32),
        ("f * 2.f", np.float32),
        ("f * 2.", np.float64),
        ("small + small", np.int32),
        ("u + n", np.uint32),
        ("big - n", np.uint64),
        ("u + 1l", np.int64),
        ("flag + flag", np.int32),
        ("-small", np.int32),
        ("+flag", np.int32),
    ],
)
def test_arithmetic_takes_the_type_cpp_would_give_it(text, dtype):
    assert value(text).dtype == dtype


def test_literals_are_read_as_cpp_spells_them():
    assert listed("n * 0 + 0x1f")[0] == 31
    assert listed("n * 0 + 010")[0] == 8
    assert listed("n * 0 + 1e3")[0] == 1000.0
    assert listed("n * 0 + M_PI")[0] == pytest.approx(np.pi)
    assert listed("n > 0 == true") == [True, False, True, True]
    assert listed('s == "a\\x62"'.replace("\\x62", "")) == [True, False, True, False]


def test_integers_divide_and_take_remainders_as_c_does():
    columns = {"a": np.array([-7, 7, -7, 7]), "b": np.array([2, -2, -2, 3])}
    assert value("a / b", columns).tolist() == [-3, -3, 3, 2]
    assert value("a % b", columns).tolist() == [-1, 1, -1, 1]
    assert value("a / 2.", columns).tolist() == [-3.5, 3.5, -3.5, 3.5]
    assert np.isinf(value("x / 0", COLUMNS)[0])


def test_an_integer_divided_by_zero_is_refused_naming_the_entry():
    with pytest.raises(ValueError, match="by zero in entry 11"):
        value("a / b", {"a": np.array([1, 1]), "b": np.array([1, 0])})
    with pytest.raises(ValueError, match=r"integer % by zero in entry 10"):
        value("n % 0", {"n": np.array([1, 0])})


def test_the_bitwise_operators_are_for_integers():
    assert listed("n ^ 1") == [3, 1, 0, 3]
    assert listed("n & 1") == [0, 0, 1, 0]
    assert listed("n | 4") == [6, 4, 5, 6]
    assert listed("n << 2") == [8, 0, 4, 8]
    assert listed("n >> 1") == [1, 0, 0, 1]
    assert listed("n << 40") == [0, 0, 0, 0]  # past the width: undefined in C++, zero here
    assert listed("~n") == [-3, -1, -2, -3]
    with pytest.raises(UnsupportedFeatureError, match="for integers"):
        value("x ^ 1")
    with pytest.raises(UnsupportedFeatureError, match="fmod"):
        value("x % 2")
    with pytest.raises(UnsupportedFeatureError, match="~ is for integers"):
        value("~x")


def test_strings_compare_with_equals_and_nothing_else():
    assert listed('s == "a"') == [True, False, True, False]
    assert listed('"a" != s') == [False, True, False, True]
    assert listed("s.size()") == [1, 2, 1, 0]
    assert listed("s.length() + s.empty()") == [1, 2, 1, 1]
    for text in ('s < "b"', "s + 1", "-s", "s == 1", "s ? 1 : 0"):
        with pytest.raises(UnsupportedFeatureError, match="string"):
            value(text)
    with pytest.raises(UnsupportedFeatureError, match="size\\(\\), length\\(\\) and empty"):
        value('s.find("a")')
    with pytest.raises(UnsupportedFeatureError, match="rather than 0"):
        value("s.size(1)")


def test_comparing_collections_gives_rvec_int_and_masks_select_with_it():
    assert value("pt > 30").content.dtype == np.int32
    assert value("!q").content.tolist() == [0, 0, 0, 0, 0]
    assert value("pt[pt > 30]").tolist() == [[40.0], [], [33.0], [50.0]]
    assert value("pt[q > 0 && pt > 35]").tolist() == [[40.0], [], [], []]
    assert listed("Sum(pt > 30)") == [1, 0, 1, 1]


def test_a_skipped_side_of_and_and_or_is_never_evaluated():
    assert listed("n > 0 && pt[0] > 30") == [True, False, True, False]
    assert listed("n == 0 || pt[0] > 30") == [True, True, True, False]
    assert listed("n > 5 && pt[7] > 0") == [False, False, False, False]
    with pytest.raises(ValueError, match="entry 11"):
        value("pt[0] > 30")


def test_and_with_a_collection_on_either_side_is_element_by_element():
    assert value("n > 0 && pt > 30").tolist() == [[1, 0], [], [1], [0, 1]]
    assert value("pt > 30 || n > 1").tolist() == [[1, 1], [], [1], [1, 1]]


def test_the_ternary_evaluates_each_branch_only_where_it_is_taken():
    assert listed("n > 0 ? pt[0] : -1.f") == [40.0, -1.0, 33.0, 5.0]
    assert value("n > 0 ? pt[0] : -1.f").dtype == np.float32
    assert listed('n > 0 ? "some" : "none"') == ["some", "none", "some", "some"]
    assert value("n > 1 ? pt : Reverse(pt)").tolist() == [[40.0, 12.5], [], [33.0], [5.0, 50.0]]
    assert listed("n > 0 ? n : 1.5") == [2.0, 1.5, 1.0, 2.0]


def test_the_ternary_refuses_what_cpp_refuses():
    with pytest.raises(UnsupportedFeatureError, match="Where"):
        value("pt > 1 ? 1 : 0")
    with pytest.raises(UnsupportedFeatureError, match="same kind"):
        value("n > 1 ? pt : 0")
    with pytest.raises(UnsupportedFeatureError, match="same kind"):
        value("n > 1 ? Combinations(pt, 2) : Combinations(pt, 2)")


def test_casts_come_in_all_three_spellings():
    assert listed("(int)f") == [1, 2, 3, 4]
    assert value("int(f * 1.5)").tolist() == [1, 3, 4, 6]
    assert value("static_cast<float>(n)").dtype == np.float32
    assert value("(unsigned int)n").dtype == np.uint32
    assert listed("(bool)n") == [True, False, True, True]
    assert value("(int)pt").content.tolist() == [40, 12, 33, 5, 50]
    assert value("(double) + 1").tolist() == [8.0, 9.0, 10.0, 11.0]  # a column called double


def test_a_nan_made_an_integer_is_refused():
    with pytest.raises(ValueError, match="NaN cast to an integer in entry 12"):
        value("(int)x")


def test_casts_that_are_not_casts_are_refused():
    with pytest.raises(FormulaError, match="not a type"):
        value("static_cast<nothing>(x)")
    with pytest.raises(FormulaError, match="takes 1 argument"):
        value("int(x, x)")


def test_an_index_is_a_number_per_entry_and_must_be_in_range():
    assert listed("n > 0 ? pt[n - 1] : 0.f") == [12.5, 0.0, 33.0, 50.0]
    assert value("grid[1]").tolist() == [1.0, 3.0, 5.0, 7.0]
    with pytest.raises(ValueError, match="index past the end of a collection in entry 11"):
        value("pt[n - 1]")
    with pytest.raises(UnsupportedFeatureError, match="an index is an integer"):
        value("pt[1.5]")
    with pytest.raises(UnsupportedFeatureError, match="indexes a number per entry"):
        value("x[0]")


def test_a_mask_must_be_as_long_as_what_it_masks():
    with pytest.raises(ValueError, match="different sizes in entry 10, 2 and 1"):
        value("pt[other > 0]")
    with pytest.raises(ValueError, match="operator \\+ was given collections"):
        value("pt + other")


def test_the_methods_of_an_rvec_are_the_ones_expressions_use():
    assert listed("pt.size()") == [2, 0, 1, 2]
    assert value("pt.size()").dtype == np.int64
    assert listed("pt.empty()") == [False, True, False, False]
    assert listed("n > 0 ? pt.front() + pt.back() : 0.f") == [52.5, 0.0, 66.0, 55.0]
    assert listed("n > 1 ? pt.at(1) : 0.f") == [12.5, 0.0, 0.0, 50.0]
    assert listed("Take(pt, 1, 0.f)[0]") == [40.0, 0.0, 33.0, 5.0]
    assert listed("pt[pt > 0].size()") == [2, 0, 1, 2]


def test_methods_nobody_has_are_refused():
    with pytest.raises(ValueError, match="front\\(\\) of an empty collection"):
        value("pt.front()")
    with pytest.raises(UnsupportedFeatureError, match="size\\(\\), empty\\(\\), front"):
        value("pt.push_back(1)")
    with pytest.raises(UnsupportedFeatureError, match="has no methods"):
        value("x.size()")
    with pytest.raises(UnsupportedFeatureError, match="with 1 arguments"):
        value("pt.size(1)")


def test_mathematical_functions_apply_to_every_element():
    assert np.allclose(value("sqrt(pt)").content, np.sqrt(COLUMNS["pt"].content))
    assert value("sqrt(pt)").content.dtype == np.float32
    assert listed("std::abs(n - 1)") == [1, 1, 0, 1]
    assert value("pow(pt, 2)").tolist()[3] == [25.0, 2500.0]
    assert listed("n * 0 + TMath::Pi()")[0] == pytest.approx(np.pi)
    assert listed("ROOT::VecOps::Sum(pt) + VecOps::Sum(pt)")[0] == 105.0


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("n > 0 ? Max(pt) : 0", [40.0, 0.0, 33.0, 50.0]),
        ("n > 0 ? Min(pt) : 0", [12.5, 0.0, 33.0, 5.0]),
        ("ArgMax(pt)", [0, 0, 0, 1]),
        ("ArgMin(pt)", [1, 0, 0, 0]),
        ("Mean(pt)", [26.25, 0.0, 33.0, 27.5]),
        ("Product(q)", [-1, 1, 1, 1]),
        ("Var(pt)", [np.var([40, 12.5], ddof=1), 0, 0, np.var([5, 50], ddof=1)]),
        ("StdDev(pt)", [np.std([40, 12.5], ddof=1), 0, 0, np.std([5, 50], ddof=1)]),
        ("Any(q > 0)", [True, False, True, False]),
        ("All(q > 0)", [False, True, True, False]),
        ("Sum(pt, 0.)", [52.5, 0.0, 33.0, 55.0]),
        ("Dot(pt, pt * 0 + 2)", [105.0, 0.0, 66.0, 110.0]),
    ],
)
def test_vecops_reductions_by_name(text, wanted):
    assert np.allclose(listed(text), wanted)


@pytest.mark.parametrize(
    ("text", "wanted"),
    [
        ("Take(pt, Argsort(pt))", [[12.5, 40.0], [], [33.0], [5.0, 50.0]]),
        ("Sort(-pt)", [[-40.0, -12.5], [], [-33.0], [-50.0, -5.0]]),
        ("Nonzero(q > 0)", [[0], [], [0], []]),
        ("Where(q > 0, pt, 0.f)", [[40.0, 0.0], [], [33.0], [0.0, 0.0]]),
        (
            "Concatenate(pt, pt)",
            [[40.0, 12.5, 40.0, 12.5], [], [33.0, 33.0], [5.0, 50.0, 5.0, 50.0]],
        ),
        ("Drop(pt, Nonzero(q > 0))", [[12.5], [], [], [5.0, 50.0]]),
        ("Enumerate(pt)", [[0, 1], [], [0], [0, 1]]),
        ("Range(n)", [[0, 1], [], [0], [0, 1]]),
        ("Range(1, n + 1)", [[1, 2], [], [1], [1, 2]]),
        ("Range(0, 6, n + 2)", [[0, 4], [0, 2, 4], [0, 3], [0, 4]]),
    ],
)
def test_vecops_that_make_collections_by_name(text, wanted):
    assert value(text).tolist() == wanted


def test_take_of_a_count_is_of_the_first_elements():
    assert value("Take(pt, 1)", {"pt": COLUMNS["pt"][2:]}).tolist() == [[33.0], [5.0]]


def test_combinations_are_members_to_take_one_at_a_time():
    assert value("Combinations(pt, 2)[1]").tolist() == [[1], [], [], [1]]
    assert value("Take(pt, Combinations(pt, eta)[0])").tolist()[0] == [40.0, 40.0, 12.5, 12.5]
    whole = value("Combinations(pt, 2)")
    assert isinstance(whole, tuple) and [each.tolist() for each in whole] == [
        [[0], [], [], [0]],
        [[1], [], [], [1]],
    ]
    with pytest.raises(UnsupportedFeatureError, match="\\[0\\] or \\[1\\]"):
        value("Combinations(pt, 2)[n]")
    with pytest.raises(IndexError, match="member 2 of 2"):
        value("Combinations(pt, 2)[2]")
    with pytest.raises(UnsupportedFeatureError, match="take one of them"):
        value("Combinations(pt, 2) + 1")
    with pytest.raises(UnsupportedFeatureError, match="take one of them"):
        value("-Combinations(pt, 2)")


def test_kinematics_by_name():
    assert np.allclose(value("DeltaPhi(pt * 0, eta)").content, COLUMNS["eta"].content)
    assert np.allclose(value("DeltaPhi(0.f, eta, 1.)").content, [0.5, -1.0, 0.0, 0.1, -0.3])
    assert np.allclose(
        value("DeltaR2(eta, 0.f, eta, 0.f)").content, 2 * COLUMNS["eta"].content ** 2
    )
    assert np.allclose(value("DeltaR2(eta, 0.f, eta, 0.f, 1.)").content[2], 4.0)
    assert np.allclose(
        value("DeltaR(eta, 0.f, eta, 0.f)").content, np.sqrt(2) * np.abs(COLUMNS["eta"].content)
    )
    masses = value("InvariantMasses(pt, eta, eta, pt * 0 + 10, pt, eta, eta, pt * 0 + 10)")
    assert np.allclose(masses.content, 20.0, rtol=1e-3)  # two alike: twice the mass
    assert value("InvariantMass(pt, eta, eta, pt * 0)").dtype == np.float32


def test_vecops_given_the_wrong_things_are_refused():
    cases = {
        "Sum(n)": "takes a collection per entry",
        "Sum(pt, n)": "single number to start from",
        "Take(pt, 1.5)": "takes an integer",
        "Range(n, n, n * 0)": "stride of zero",
        "Combinations(pt, n)": "one size for every entry",
    }
    for text, message in cases.items():
        with pytest.raises((UnsupportedFeatureError, FormulaError), match=message):
            value(text)
    with pytest.raises(ValueError, match="Max of an empty collection in entry 11"):
        value("Max(pt)")
    with pytest.raises(ValueError, match="Take of elements past the end"):
        value("Take(pt, 1)")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty expression"),
        ("x +", "a number, a name or '\\('"),
        ("x x", "an operator or the end"),
        ("(x", "a '\\)' was expected"),
        ("nosuch > 1", "not a column of this frame.*there is 'pt'"),
        ("smal > 1", "the nearest are 'small'"),
        ("nosuch(x)", "not a function this knows"),
        ("sqrt(x, x)", "takes 1 argument, and"),
        ("Take(pt)", "takes 2 to 3 arguments"),
        ("x; y", "not part of the C\\+\\+ expressions"),
        ("x.size", "not a column of this frame"),
    ],
)
def test_what_is_not_an_expression_is_refused_when_it_is_compiled(text, message):
    with pytest.raises(FormulaError, match=message):
        value(text)


def test_a_lambda_or_a_non_string_is_refused_with_what_to_do_instead():
    with pytest.raises(UnsupportedFeatureError, match=r"starts a lambda.*Python callable"):
        value("[&x](x)")
    with pytest.raises(FormulaError, match="for a lambda or statements, give a Python callable"):
        value("[](double x) { return x; }")
    with pytest.raises(TypeError, match="is a string"):
        Expression(3, ["x"])  # type: ignore[arg-type]
    assert repr(Expression("x + 1", ["x"])) == "<Expression 'x + 1'>"
    assert Expression("pt[0] + x * pt.size()", ["pt", "x"]).columns == ("pt", "x")


def test_columns_that_are_python_objects_are_refused_by_name():
    with pytest.raises(UnsupportedFeatureError, match="'m' holds Python objects - nested"):
        value("m + 1", {"m": [[1, 2], [3]]})
    with pytest.raises(UnsupportedFeatureError, match="'o' holds Python objects"):
        value("o + 1", {"o": np.array([{}, None], dtype=object)})


def test_fixed_size_arrays_and_collections_of_them_are_rvecs():
    assert value("grid * 2").tolist() == [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0], [12.0, 14.0]]
    wide = Jagged(np.arange(12.0).reshape(6, 2), [0, 2, 2, 5, 6])
    assert value("Sum(w)", {"w": wide}).tolist() == [
        1.0 + 2 + 3 + 0,
        0.0,
        45.0 - 1 - 2 - 3 + 0,
        21.0,
    ]
    assert value("n * 0 + 7", {"n": np.zeros(3)}).tolist() == [7.0, 7.0, 7.0]
    assert value("7", {"n": np.zeros(3)}).tolist() == [7, 7, 7]
