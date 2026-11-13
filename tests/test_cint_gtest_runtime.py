"""GoogleTest's runtime, piece by piece: printing values, the checks, matchers, diagnostics."""

from __future__ import annotations

import io
import sys
from collections.abc import Iterator
from typing import Any

import numpy
import pytest

from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.runtime import ROOT, diags
from xrdroot.pycint.runtime.diags import TestSupport
from xrdroot.pycxx.runtime.gtest import checks, runner, testing
from xrdroot.pycxx.runtime.gtest.exceptions import caught, internal
from xrdroot.pycxx.runtime.gtest.fixtures import PARAMETER
from xrdroot.pycxx.runtime.gtest.state import STATE, Result, pending, skip
from xrdroot.pycxx.runtime.gtest.values import almost_equal, equal, show, ulps


@pytest.fixture(autouse=True)
def fresh() -> Iterator[None]:
    """Every test starts with no tests registered, no test running and nothing pending."""
    runner.reset()
    yield
    runner.reset()


@pytest.fixture
def running() -> Result:
    """A test running, for the checks to record their failures against."""
    result = STATE.result = Result("Suite", "Name")
    return result


def said(capsys: pytest.CaptureFixture[str]) -> str:
    """What has been printed, the failure being reported included."""
    runner.flush()
    return capsys.readouterr().out


def test_values_are_printed_as_googletest_prints_them() -> None:
    assert [show(v) for v in (True, numpy.bool_(False), None, 'a"b\n', 3, numpy.int8(-2))] == [
        "true", "false", "NULL", '"a\\"b\\n"', "3", "-2",
    ]  # fmt: skip
    assert [show(v) for v in (0.5, 0.1 + 0.2, float("inf"), numpy.float32(0.1234567))] == [
        "0.5", "0.30000000000000004", "inf", "0.123456702",
    ]  # fmt: skip
    assert show(numpy.float32(0.1)) == "0.1"
    assert [show(v) for v in ((1, "a"), [], [1, 2], numpy.arange(40), object)] == [
        '(1, "a")', "{}", "{ 1, 2 }",
        "{ " + ", ".join(map(str, range(32))) + ", ... }", "<class 'object'>",
    ]  # fmt: skip


def test_equality_is_cs_for_characters_and_for_arrays_element_by_element() -> None:
    marker = object()
    assert equal(marker, marker)
    assert equal("a", 97) and equal(97, "a") and not equal("ab", 97)
    assert equal(numpy.array([1, 2]), numpy.array([1, 2]))
    assert not equal(numpy.array([1, 2]), numpy.array([1, 3]))


def test_float_and_double_equality_hold_within_four_units_in_the_last_place() -> None:
    one = 1.0
    assert ulps(one, numpy.nextafter(one, 2), False) == 1
    assert ulps(-0.0, 0.0, False) == 0
    assert ulps(-1.0, -numpy.nextafter(1.0, 2), False) == 1
    assert almost_equal(one, one + 4 * numpy.spacing(one), False)
    assert not almost_equal(one, one + 5 * numpy.spacing(one), False)
    assert almost_equal(1e300, float("inf"), True)  # both past a float's range
    assert not almost_equal(float("nan"), float("nan"), False)


def test_each_failed_comparison_is_worded_as_googletest_words_it(
    running: Result, capsys: pytest.CaptureFixture[str]
) -> None:
    check = checks.Compare
    assert not check("STREQ", "a", None, "s", "t", "f.cxx", 1, False)
    assert not check("STRCASEEQ", "a", "B", "s", "t", "f.cxx", 2, False)
    assert not check("STRCASEEQ", None, "B", "s", "t", "f.cxx", 3, False)
    assert check("STRCASEEQ", None, None, "s", "t", "f.cxx", 3, False)
    assert not check("STRNE", "a", "a", "s", "t", "f.cxx", 4, False)
    assert not check("DOUBLE_EQ", 1.0, 1.5, "d", "e", "f.cxx", 5, True)
    assert running.failures == 5 and running.fatal
    assert said(capsys) == (
        'f.cxx:1: Failure\nExpected equality of these values:\n  s\n    Which is: "a"\n'
        "  t\n    Which is: NULL\n"
        'f.cxx:2: Failure\nExpected equality of these values:\n  s\n    Which is: "a"\n'
        '  t\n    Which is: "B"\nIgnoring case\n'
        'f.cxx:3: Failure\nExpected equality of these values:\n  s\n    Which is: NULL\n'
        '  t\n    Which is: "B"\nIgnoring case\n'
        'f.cxx:4: Failure\nExpected: (s) != (t), actual: "a" vs "a"\n'
        "f.cxx:5: Failure\nExpected equality of these values:\n  d\n    Which is: 1\n"
        "  e\n    Which is: 1.5\n"
    )


def test_a_failed_near_says_how_far_apart_the_values_are(
    running: Result, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not checks.Near(1.0, 1.5, 0.1, "a", "b", "e", "f.cxx", 7, False)
    assert said(capsys) == (
        "f.cxx:7: Failure\nThe difference between a and b is 0.5, which exceeds e, where\n"
        "a evaluates to 1,\nb evaluates to 1.5, and\ne evaluates to 0.1.\n"
    )


def test_a_failed_matcher_says_what_it_expected(
    running: Result, capsys: pytest.CaptureFixture[str]
) -> None:
    assert not checks.That("abc", testing.HasSubstr("x"), "s", "m", "f.cxx", 8, False)
    assert not checks.That(3, 4, "n", "4", "f.cxx", 9, False)
    assert said(capsys) == (
        'f.cxx:8: Failure\nValue of: s\nExpected: has substring "x"\n  Actual: "abc"\n'
        "f.cxx:9: Failure\nValue of: n\nExpected: is equal to 4\n  Actual: 3\n"
    )


def test_the_matchers_match_as_gmocks_do() -> None:
    cases: list[tuple[Any, Any, bool, str]] = [
        (testing.StartsWith("ab"), "abc", True, 'starts with "ab"'),
        (testing.EndsWith("bc"), "abc", True, 'ends with "bc"'),
        (testing.ContainsRegex("b+"), "abc", True, 'contains regular expression "b+"'),
        (testing.MatchesRegex("b+"), "abc", False, 'matches regular expression "b+"'),
        (testing.StrEq("a"), "a", True, 'is equal to "a"'),
        (testing.StrNe("a"), "a", False, 'isn\'t equal to "a"'),
        (testing.Ne(2), 3, True, "isn't equal to 2"),
        (testing.Lt(2), 3, False, "is < 2"),
        (testing.Le(3), 3, True, "is <= 3"),
        (testing.Gt(2), 3, True, "is > 2"),
        (testing.Ge(4), 3, False, "is >= 4"),
        (testing.IsEmpty(), [], True, "is empty"),
        (testing.Not(testing.Eq(1)), 1, False, "not (is equal to 1)"),
        (testing.AllOf(testing.Gt(1), testing.Lt(5)), 3, True, "(is > 1) and (is < 5)"),
        (testing.AnyOf(testing.Gt(5), testing.Lt(1)), 3, False, "(is > 5) or (is < 1)"),
    ]
    for matcher, value, matches, description in cases:
        assert (matcher.matches(value), matcher.describe()) == (matches, description)


def test_only_what_a_cpp_throw_becomes_counts_as_the_exception_a_test_expects() -> None:
    class Custom(Exception):
        pass

    assert internal(UnsupportedFeatureError("gap")) and internal(KeyboardInterrupt())
    assert not internal(ValueError("v"))
    assert caught(ValueError("v"), "const std::invalid_argument &")
    assert caught(Custom(), "::mine::Custom") and not caught(Custom(), "Other")
    assert caught(KeyError("k"), "std::out_of_range")


def test_a_missing_piece_of_xrdroot_is_never_the_exception_a_test_expects(
    running: Result,
) -> None:
    def gap() -> None:
        raise UnsupportedFeatureError("the thing is missing")

    with pytest.raises(UnsupportedFeatureError):
        checks.Throws(gap, "std::exception", "gap()", "f.cxx", 1, False)
    assert running.failures == 0


def test_a_skip_or_a_failure_outside_any_test_is_reported_and_counted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not skip("f.cxx", 1, "why")
    assert not checks.Fail("f.cxx", 2, False)
    assert STATE.stray == 1
    assert said(capsys) == "f.cxx:1: Skipped\nwhy\nf.cxx:2: Failure\nFailed\n"
    note = pending() << "lost"  # nothing failed: a message streamed after it goes nowhere
    assert note.render() == ": \nlost\n" and STATE.pending is None


def test_a_fixtures_inherited_hooks_and_questions_answer_as_googletests() -> None:
    test = testing.Test()
    test.SetUp(), test.TearDown(), test.RecordProperty("k", 1)
    for hook in ("SetUpTestSuite", "TearDownTestSuite", "SetUpTestCase", "TearDownTestCase"):
        getattr(testing.Test, hook)()
    assert not testing.Test.IsSkipped() and not testing.Test.HasNonfatalFailure()
    result = STATE.result = Result("S", "N", failures=1)
    assert testing.Test.HasNonfatalFailure()
    result.skipped = True
    assert testing.Test.IsSkipped()
    info = testing.UnitTest.GetInstance().current_test_info()
    assert (info.name(), info.test_suite_name(), info.test_case_name()) == ("N", "S", "S")
    STATE.result = None
    assert testing.UnitTest.GetInstance().current_test_info() is None


def test_the_parameter_and_environment_pieces_do_what_googletests_do() -> None:
    environment = testing.Environment()
    environment.SetUp(), environment.TearDown()
    PARAMETER.value = 7
    assert isinstance(testing.WithParamInterface(), testing.WithParamInterface)
    assert testing.WithParamInterface["int"].GetParam() == 7
    assert testing.TestParamInfo["int"](1, 0).index == 0
    assert testing.StaticAssertTypeEq["int", "int"]()


def test_googletests_namespaces_refuse_by_name_what_they_do_not_have() -> None:
    with pytest.raises(UnsupportedFeatureError, match=r"^testing::Mock is a part of GoogleTest"):
        testing.Mock  # noqa: B018
    with pytest.raises(AttributeError):
        testing.__wrapped__  # noqa: B018
    assert ROOT.TestSupport is TestSupport and ROOT.testing is testing


def test_standard_output_and_error_can_be_captured_as_googletest_captures_them() -> None:
    testing.internal.CaptureStdout()
    testing.internal.CaptureStderr()
    print("out")
    print("err", file=sys.stderr)
    assert testing.internal.GetCapturedStderr() == "err\n"
    assert testing.internal.GetCapturedStdout() == "out\n"


def test_a_check_of_diagnostics_ends_once_and_lets_other_lines_through(
    running: Result, capsys: pytest.CaptureFixture[str]
) -> None:
    outer = TestSupport.CheckDiagsRAII(3000, "Outer", "said")
    inner = TestSupport.CheckDiagsRAII()
    sys.stderr.write("Error in <Outer>: said\nplain\npart")
    inner.close()
    outer.close()
    outer.close()
    assert capsys.readouterr().err == "plain\npart"
    assert running.failures == 0


def test_a_diagnostic_read_after_its_checks_ended_is_let_through(
    capsys: pytest.CaptureFixture[str],
) -> None:
    check = TestSupport.CheckDiagsRAII()
    reader = sys.stderr
    sys.stderr = io.StringIO()  # someone else's stream in place when the check ends
    check.close()
    assert isinstance(reader, io.TextIOBase)
    sys.stderr = reader
    reader.write("Error in <Late>: said\n")
    reader.flush()
    sys.stderr = reader.finish()  # type: ignore[attr-defined]
    assert capsys.readouterr().err == "Error in <Late>: said\n"
    assert not diags.ACTIVE


def test_googletests_flags_are_taken_from_any_argument_list() -> None:
    runner.InitGoogleTest()
    runner.InitGoogleTest(1, ("prog", "--gtest_filter=A.*", "--gtest_color=no"))
    assert runner.REGISTRY.filter == "A.*"


ENVIRONMENT_SAID: list[str] = []


class Loud(testing.Environment):
    def SetUp(self) -> None:
        ENVIRONMENT_SAID.append("up")

    def TearDown(self) -> None:
        ENVIRONMENT_SAID.append("down")


class Broken(testing.Test):
    def SetUp(self) -> None:
        checks.Fail("f.cxx", 1, True)

    def TestBody(self) -> None:
        raise AssertionError("the body runs after a fatal failure in SetUp")


def test_a_test_whose_fixture_cannot_be_made_or_set_up_fails_without_its_body(
    capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = {"Broken_A_Test": Broken}
    runner.REGISTRY.cases += [
        runner.Case("Broken", "A", "Broken_A_Test", namespace),
        runner.Case("Gone", "B", "Gone_B_Test", namespace),
    ]
    assert testing.AddGlobalTestEnvironment(Loud()) is not None
    assert runner.RunAllTests() == 1
    out = capsys.readouterr().out
    assert "f.cxx:1: Failure\nFailed\n[  FAILED  ] Broken.A" in out
    assert (
        "unknown file: Failure\nC++ exception with description \"'Gone_B_Test'\" thrown in "
        "the test "
        "fixture's constructor.\n"
    ) in out
    assert ENVIRONMENT_SAID == ["up", "down"]
