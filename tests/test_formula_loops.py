"""The implicit loop: collections evaluated element by element, as ``TTree::Draw`` loops.

The collection cases are go-hep's ``rexpr`` tests, with one deliberate
difference: two collections of different lengths are paired up to the
shorter, as ROOT's documentation says ``TTreeFormula`` does, where go-hep
refuses them. The multi-dimensional cases are the table in ROOT's own
``TTree::Draw`` documentation, of a ``fMatrix[3][3]`` and a ``fResults[5][2]``.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import FormulaError, Jagged, UnsupportedFeatureError, compile_formula


def jagged(rows):
    lengths = [len(row) for row in rows]
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return Jagged(np.asarray([x for row in rows for x in row], dtype=np.float64), offsets)


def run(text, columns, **kwargs):
    result = compile_formula(text, columns).evaluate(columns, **kwargs)
    return result.tolist()


GOHEP = {
    "pt": jagged([[10, 20, 30]]),
    "eta": jagged([[0.5, -1.5, 2.5]]),
    "w": np.array([2.0]),
}


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("pt", [10, 20, 30]),
        ("pt*2", [20, 40, 60]),
        ("pt*w", [20, 40, 60]),
        ("pt + eta", [10.5, 18.5, 32.5]),
        ("pt > 15", [0, 1, 1]),
        ("pt - pt[0]", [0, 10, 20]),
        ("pt > Sum$(pt)/Length$(pt)", [0, 0, 1]),
        ("Iteration$ + pt*0", [0, 1, 2]),
        ("pt[Iteration$] + eta", [10.5, 18.5, 32.5]),
        ("Length$ + pt*0", [3, 3, 3]),
        ("Entry$ + pt*0", [7, 7, 7]),
        ("Alt$(pt, 0)", [10, 20, 30]),
    ],
)
def test_a_collection_is_evaluated_once_per_element(text, want):
    assert run(text, GOHEP, entry_start=7) == [want]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("pt[0]", 10),
        ("pt[2] - pt[0]", 20),
        ("Length$(pt)", 3),
        ("Sum$(pt)", 60),
        ("Min$(pt)", 10),
        ("Max$(pt)", 30),
        ("Sum$(pt) / Length$(pt)", 20),
        ("MaxIf$(pt, eta > 0)", 30),
        ("MinIf$(pt, eta > 0)", 10),
        ("Entry$", 7),
        ("Entries$", 100),
        ("LocalEntry$", 7),
        ("w + 1", 3),
        ("Iteration$", 0),
        ("Length$", 1),
        ("pt.size()", 3),
        ("@pt.size()", 3),
    ],
)
def test_an_index_or_a_reducer_makes_one_value_per_entry(text, want):
    assert run(text, GOHEP, entry_start=7, entries=100) == [want]


def test_alt_supplies_the_value_a_shorter_collection_does_not_have():
    columns = {"a": jagged([[1, 2, 3]]), "b": jagged([[10]])}
    assert run("a + Alt$(b, 0)", columns) == [[11, 2, 3]]
    assert run("Alt$(b[2], -1) + Alt$(a[2], -1)", columns) == [2]


def test_an_empty_collection_gives_nothing_but_its_reducers_still_answer():
    columns = {"pt": jagged([[]]), "w": np.array([1.0])}
    assert run("pt*w", columns) == [[]]
    assert run("Length$(pt)", columns) == [0]
    assert run("Sum$(pt)", columns) == [0]
    assert run("Min$(pt)", columns) == [0]
    assert run("MaxIf$(pt, pt > 0)", columns) == [0]


def test_collections_of_different_lengths_run_to_the_shorter_as_root_runs_them():
    columns = {
        "a": jagged([[1, 2, 3], [1], [], [5, 6]]),
        "b": jagged([[10, 20], [10, 20, 30], [7], [1, 1]]),
    }
    assert run("a + b", columns) == [[11, 22], [11], [], [6, 7]]
    one = {"a": jagged([[1]]), "b": jagged([[10, 20, 30]]), "w": np.array([1.0])}
    assert run("a", one) == [[1]]
    assert run("b*w", one) == [[10, 20, 30]]
    assert run("a + b", one) == [[11]]


def test_an_element_that_is_not_there_is_left_out_and_masked():
    columns = {"pt": jagged([[10, 20], [5], []]), "n": np.array([2, 1, 0])}
    formula = compile_formula("pt[1]", columns)
    values, valid = formula.evaluate_masked(columns)
    assert valid.tolist() == [True, False, False]
    assert values[0] == 20
    assert np.isnan(formula.evaluate(columns)[1:]).all()
    assert run("pt[n-1]", columns)[:2] == [20, 5]
    values, valid = compile_formula("pt - pt[1]", columns).evaluate_masked(columns)
    assert valid.tolist() == [[True, True], [False], []]
    assert run("pt - pt[1]", columns) == [[-10, 0], [], []]
    assert run("pt[5] + Sum$(pt)", columns)[0] != run("pt[5] + Sum$(pt)", columns)[0]  # NaN


def test_a_ternary_is_missing_only_where_the_side_it_takes_is():
    columns = {"pt": jagged([[10, 20], [5]]), "w": np.array([1.0, 3.0])}
    taken = run("w > 2 ? pt[1] : -1", columns)
    assert taken[0] == -1 and np.isnan(taken[1])
    assert run("w > 2 ? pt : 0", columns) == [[0, 0], [5]]
    assert run("w > 2 ? 1 : pt[3]", columns)[1] == 1


def test_a_cast_of_nan_to_an_integer_has_no_value():
    columns = {"x": np.array([np.nan, 2.7, -2.7])}
    values, valid = compile_formula("(int)x", columns).evaluate_masked(columns)
    assert valid.tolist() == [False, True, True]
    assert values.tolist()[1:] == [2, -2]


def test_an_index_that_is_nan_or_past_either_end_has_no_value():
    columns = {"pt": jagged([[10, 20]] * 3), "i": np.array([np.nan, -1.0, 1.9])}
    _, valid = compile_formula("pt[i]", columns).evaluate_masked(columns)
    assert valid.tolist() == [False, False, True]
    assert compile_formula("pt[i]", columns).evaluate(columns)[2] == 20


MATRIX = np.arange(3 * 3, dtype=np.float64).reshape(1, 3, 3)
RESULTS = 100 + np.arange(5 * 2, dtype=np.float64).reshape(1, 5, 2)


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("fMatrix", 9),
        ("fMatrix[][]", 9),
        ("fMatrix[1]", 3),
        ("fMatrix[1][]", 3),
        ("fMatrix[][0]", 3),
        ("fMatrix[2][1] - fResults[4][1]", 1),
        ("fMatrix[2][] - fResults[4][1]", 3),
        ("fMatrix[2][] - fResults[4][]", 2),
        ("fMatrix[][2] - fResults[][1]", 3),
        ("fMatrix[][2] - fResults[][]", 6),
        ("fMatrix[][2] - fResults[3][]", 2),
        ("fMatrix[][] - fResults[][]", 6),
    ],
)
def test_multi_dimensional_arrays_loop_as_root_s_documentation_tabulates(text, count):
    columns = {"fMatrix": MATRIX, "fResults": RESULTS}
    formula = compile_formula(text, columns)
    result = formula.evaluate(columns)
    got = len(result.flat) if isinstance(result, Jagged) else len(result)
    assert got == count
    assert formula.per_element == (count > 1)


def test_multi_dimensional_values_are_the_ones_the_loop_pairs():
    columns = {"m": MATRIX, "r": RESULTS}
    m, r = MATRIX[0], RESULTS[0]
    assert run("m[][2] - r[3][]", columns) == [[m[0, 2] - r[3, 0], m[1, 2] - r[3, 1]]]
    expected = [m[i, 2] - r[i, j] for i in range(3) for j in range(2)]
    assert run("m[][2] - r[][]", columns) == [expected]
    expected = [m[i, j] - r[i, j] for i in range(3) for j in range(2)]
    assert run("m - r", columns) == [expected]
    assert run("m[1][2]", columns) == [5]
    assert run("m[][0]", columns) == [[0, 3, 6]]


def test_an_index_after_a_loop_is_worked_out_for_each_entry():
    columns = {
        "m": np.arange(18.0).reshape(2, 3, 3),
        "n": np.array([0, 2]),
        "f": np.array([1.0, np.nan]),
    }
    assert run("m[][n]", columns) == [[0, 3, 6], [11, 14, 17]]
    values, valid = compile_formula("m[][f]", columns).evaluate_masked(columns)
    assert values[0].tolist() == [1, 4, 7]
    assert valid.tolist() == [[True] * 3, [False] * 3]


def test_rows_of_fixed_arrays_loop_over_both_dimensions():
    columns = {"x": Jagged(np.arange(12.0).reshape(6, 2), [0, 2, 2, 6])}
    assert run("x", columns) == [[0, 1, 2, 3], [], [4, 5, 6, 7, 8, 9, 10, 11]]
    assert run("x[][1]", columns) == [[1, 3], [], [5, 7, 9, 11]]
    assert run("Sum$(x[0])", columns) == [1, 0, 9]


def test_vectors_of_vectors_are_indexed_and_sized_level_by_level():
    columns = {"vv": [[[1, 2], [3]], [], [[4, 5, 6]]], "n": np.array([1, 0, 0])}
    assert run("vv", columns) == [[1, 2, 3], [], [4, 5, 6]]
    assert run("vv[0]", columns) == [[1, 2], [], [4, 5, 6]]
    assert run("vv[][0]", columns) == [[1, 3], [], [4]]
    assert run("vv.size()", columns) == [[2, 1], [], [3]]
    assert run("@vv.size()", columns) == [2, 0, 1]
    assert run("vv[n].size()", columns)[::2] == [1, 3]  # the middle entry has no vv[0]
    assert run("Length$(vv)", columns) == [3, 0, 3]
    assert run("Sum$(vv[0])", columns) == [3, 0, 15]
    values = compile_formula("vv[0][1]", columns).evaluate(columns)
    assert values[0] == 2 and np.isnan(values[1]) and values[2] == 5


def test_a_batch_of_empty_rows_is_as_deep_as_the_names_say():
    columns = {"vv": [[], []]}
    formula = compile_formula("vv[0][0]", {"vv": 2})
    assert np.isnan(formula.evaluate(columns)).all()
    with pytest.raises(FormulaError, match="has 1 dimension and is given 2"):
        compile_formula("vv[0][0]", columns).evaluate(columns)


def test_a_collection_indexed_by_a_collection_follows_the_index():
    columns = {
        "pt": jagged([[10, 20, 30], [], [1, 2]]),
        "idx": [[2, 0], [], [5, 1]],
        "w": np.array([1.0, 2.0, 3.0]),
    }
    assert run("pt[idx]", columns) == [[30, 10], [], [2]]
    assert run("pt[idx] * w", columns) == [[30, 10], [], [6]]
    assert run("Sum$(pt[idx])", columns) == [40, 0, 2]
    assert compile_formula("pt[idx]", {"pt": 1, "idx": 1}).per_element
    with pytest.raises(UnsupportedFeatureError, match="ROOT runs those as two loops"):
        run("pt - pt[idx]", columns)


def test_strings_compare_with_strings_and_do_nothing_else():
    columns = {"s": ["ab", "cd", "abc"], "b": np.array([b"x", b"y", b"z"]), "x": np.ones(3)}
    assert run('s == "ab"', columns) == [True, False, False]
    assert run('s != "ab"', columns) == [False, True, True]
    assert run('b == "y"', columns) == [False, True, False]
    assert run('strstr(s, "b")', columns) == [True, False, True]
    assert run("s", columns) == ["ab", "cd", "abc"]
    assert run('x > 0 ? "yes" : "no"', columns) == ["yes"] * 3
    words = {"words": [["a", "bb"], [], ["bb"]]}
    assert run('words == "bb"', words) == [[False, True], [], [True]]
    for text, message in [
        ("s + 1", "\\+ was given strings"),
        ("sqrt(s)", "sqrt was given strings"),
        ('s < "b"', "< was given a string"),
        ("s == 1", "== was given a string"),
        ("!s", "! was given strings"),
        ("s % 2", "% was given strings"),
        ("Sum$(s)", "Sum\\$ was given strings"),
        ("x[s]", "an index was given strings"),
        ("strstr(x, x)", "strstr compares strings, and was given numbers"),
        ("x > 0 ? s : 1", "a ternary gives a string on one side"),
        ("(int)s", "the cast to int was given strings"),
    ]:
        with pytest.raises(UnsupportedFeatureError, match=message):
            run(text, {**columns, "x": np.ones((3, 2))} if "x[" in text else columns)


def test_columns_of_python_objects_are_taken_apart_or_refused_by_name():
    assert run("x * 2", {"x": np.array([1, 2], dtype=object)}) == [2, 4]
    assert run("Sum$(x)", {"x": [(1, 2), np.array([3.0])]}) == [3, 3]
    with pytest.raises(UnsupportedFeatureError, match="'m' holds whole objects or maps"):
        run("m", {"m": [{"a": 1}]})
    with pytest.raises(UnsupportedFeatureError, match="'m' mixes rows with single values"):
        run("m", {"m": [[1], 2]})


def test_the_columns_have_to_be_there_and_line_up():
    formula = compile_formula("a + b", ["a", "b"])
    with pytest.raises(KeyError, match="reads 'b', which is not among the columns given"):
        formula.evaluate({"a": np.ones(2)})
    with pytest.raises(ValueError, match="a has 2, b has 3"):
        formula.evaluate({"a": np.ones(2), "b": np.ones(3)})
    with pytest.raises(ValueError, match="given no columns"):
        compile_formula("Entry$", []).evaluate({})
    assert compile_formula("Entry$", []).evaluate({}, rows=2).tolist() == [0, 1]
    assert compile_formula("Entry$", []).evaluate({"x": [0, 0, 0]}).tolist() == [0, 1, 2]


def test_entry_numbers_are_where_the_rows_came_from():
    formula = compile_formula("Entry$ * 10 + LocalEntry$", [])
    result = formula.evaluate({}, entry_numbers=[5, 9], local_entries=[0, 4])
    assert result.tolist() == [50, 94]
    assert compile_formula("Entries$", []).evaluate({}, rows=2, entries=40).tolist() == [40, 40]


def test_a_constant_is_one_value_per_entry():
    assert (
        compile_formula("2 * TMath::Pi()", []).evaluate({}, rows=3).tolist()
        == [pytest.approx(2 * np.pi)] * 3
    )
    values, valid = compile_formula("1", []).evaluate_masked({}, rows=2)
    assert values.tolist() == [1, 1] and valid.tolist() == [True, True]
