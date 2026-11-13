"""GoogleTest programs translated and run: ``TEST`` blocks, assertions, the runner's output."""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest

from cintfake import fake
from xrdroot.cli import main
from xrdroot.pycint import Refusal, translate
from xrdroot.pycxx import cache

#: ROOT's diagnostic levels, which ``ROOT_EXPECT_ERROR`` and ``CheckDiagsRAII`` name.
LEVELS = {"kInfo": 1000, "kWarning": 2000, "kError": 3000, "kSysError": 5000}


@pytest.fixture(autouse=True)
def fake_pyroot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """``xrdroot.pyroot`` as the fake ROOT, with ROOT's diagnostic levels."""
    module = types.ModuleType("xrdroot.pyroot")
    module.__dict__.update(vars(fake()), **LEVELS)
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot", module)
    monkeypatch.setenv(cache.ENVIRONMENT, str(tmp_path / "cache"))
    return module


def program(tmp_path: Path, body: str, name: str = "unit.cxx") -> Path:
    path = tmp_path / name
    path.write_text('#include "gtest/gtest.h"\n#include <ROOT/TestSupport.hxx>\n' + body)
    return path


def ran(path: Path, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str]:
    """The exit status of running ``path`` with ``argv``, and its output, timings made 0."""
    status = main(["run", "-n", str(path), "--", *argv])
    out = capsys.readouterr().out.replace(str(path), "FILE")
    return status, re.sub(r"\(\d+ ms", "(0 ms", out)


PASSING = """
TEST(Sums, Add) {
   EXPECT_EQ(1 + 1, 2);
   ASSERT_NE(1, 2) << "never said";
   EXPECT_TRUE(3 > 2);
   EXPECT_FALSE(3 < 2);
   EXPECT_LT(1, 2); EXPECT_LE(2, 2); EXPECT_GT(3, 2); EXPECT_GE(3, 3);
   EXPECT_STREQ("a", "a"); EXPECT_STRNE("a", "b");
   EXPECT_STRCASEEQ("Ab", "aB"); EXPECT_STRCASENE("a", "b");
   EXPECT_FLOAT_EQ(0.1f + 0.2f, 0.3f); EXPECT_DOUBLE_EQ(0.1 + 0.2, 0.3);
   EXPECT_NEAR(1.0, 1.05, 0.1);
   EXPECT_THROW(throw std::invalid_argument("x"), std::invalid_argument);
   EXPECT_NO_THROW(int a = 1);
   EXPECT_ANY_THROW(throw std::runtime_error("x"));
   EXPECT_THAT(std::string("a fit"), ::testing::HasSubstr("fit"));
   SUCCEED() << "fine";
}
"""


def test_a_passing_program_prints_googletests_report_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, PASSING), capsys)
    assert status == 0
    assert out == (
        "[==========] Running 1 test from 1 test suite.\n"
        "[----------] Global test environment set-up.\n"
        "[----------] 1 test from Sums\n"
        "[ RUN      ] Sums.Add\n"
        "[       OK ] Sums.Add (0 ms)\n"
        "[----------] 1 test from Sums (0 ms total)\n\n"
        "[----------] Global test environment tear-down\n"
        "[==========] 1 test from 1 test suite ran. (0 ms total)\n"
        "[  PASSED  ] 1 test.\n"
    )


FAILING = """
TEST(Bad, Expect) {
   int x = 3;
   EXPECT_EQ(x, 4) << "x was " << x;
   EXPECT_TRUE(x == 4);
   printf("after\\n");
}
TEST(Bad, Assert) {
   ASSERT_LT(3, 2);
   printf("never\\n");
}
"""


def test_an_expect_eq_failure_is_reported_and_the_test_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, FAILING), capsys)
    assert status == 1
    assert (
        "[ RUN      ] Bad.Expect\n"
        "FILE:6: Failure\n"
        "Expected equality of these values:\n  x\n    Which is: 3\n  4\nx was 3\n"
        "FILE:7: Failure\nValue of: x == 4\n  Actual: false\nExpected: true\n"
        "after\n"
        "[  FAILED  ] Bad.Expect (0 ms)\n"
    ) in out


def test_an_assert_failure_ends_the_test_and_the_summary_lists_every_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, FAILING), capsys)
    assert status == 1
    assert "FILE:11: Failure\nExpected: (3) < (2), actual: 3 vs 2\n[  FAILED  ] Bad.Assert" in out
    assert "never" not in out
    assert out.endswith(
        "[  PASSED  ] 0 tests.\n"
        "[  FAILED  ] 2 tests, listed below:\n"
        "[  FAILED  ] Bad.Expect\n"
        "[  FAILED  ] Bad.Assert\n\n"
        " 2 FAILED TESTS\n"
    )


FIXTURE = """
class Store : public ::testing::Test {
protected:
   int fN = 1;
   static int fSuites;
   static void SetUpTestSuite() { fSuites = 10; printf("suite up\\n"); }
   static void TearDownTestCase() { printf("suite down\\n"); }
   void SetUp() override { fN = fSuites + 1; }
   void TearDown() override { printf("down %d\\n", fN); }
   ~Store() { printf("gone\\n"); }
};
int Store::fSuites = 0;
TEST_F(Store, SeesSetUp) {
   EXPECT_EQ(fN, 11);
   SCOPED_TRACE("first");
   EXPECT_GT(fN, 20);
   EXPECT_FALSE(HasFatalFailure());
   EXPECT_TRUE(HasFailure());
}
TEST_F(Store, DISABLED_Later) { FAIL(); }
"""


def test_a_fixture_is_set_up_and_torn_down_around_each_test_and_its_suite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, FIXTURE), capsys)
    assert status == 1
    assert (
        "[----------] 1 test from Store\n"
        "suite up\n"
        "[ RUN      ] Store.SeesSetUp\n"
        "FILE:18: Failure\nExpected: (fN) > (20), actual: 11 vs 20\n"
        "Google Test trace:\nFILE:17: first\n"
        "down 11\ngone\n"
        "[  FAILED  ] Store.SeesSetUp (0 ms)\n"
        "suite down\n"
    ) in out
    assert out.endswith(" 1 FAILED TEST\n\n  YOU HAVE 1 DISABLED TEST\n\n")


def test_the_disabled_tests_run_when_asked_and_the_filter_selects_by_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = program(tmp_path, FIXTURE)
    status, out = ran(path, capsys, "--gtest_filter=*Later*", "--gtest_also_run_disabled_tests")
    assert status == 1
    assert out.startswith("Note: Google Test filter = *Later*\n[==========] Running 1 test")
    assert "FILE:22: Failure\nFailed\ndown 11\ngone\n[  FAILED  ] Store.DISABLED_Later" in out
    status, out = ran(path, capsys, "--gtest_filter=Store.*-*Sees*:*Later*")
    assert status == 0
    assert "Running 0 tests from 0 test suites." in out


PARAMETERISED = """
struct Case { int n; const char *name; };
class Even : public ::testing::TestWithParam<int> {};
TEST_P(Even, IsEven) { EXPECT_EQ(GetParam() % 2, 0); }
INSTANTIATE_TEST_SUITE_P(Few, Even, ::testing::Values(2, 3));
INSTANTIATE_TEST_SUITE_P(, Even, ::testing::Range(4, 8, 2));
class Named : public ::testing::TestWithParam<Case> {};
TEST_P(Named, Positive) { EXPECT_GT(GetParam().n, 0); }
INSTANTIATE_TEST_SUITE_P(All, Named, ::testing::Values(Case{1, "one"}),
   [](const ::testing::TestParamInfo<Case> &info) { return std::string(info.param.name); });
std::vector<int> gSizes{1, 5};
class Pairs : public ::testing::TestWithParam<std::tuple<int, bool>> {};
TEST_P(Pairs, Ok) { SUCCEED(); }
INSTANTIATE_TEST_CASE_P(Grid, Pairs, ::testing::Combine(::testing::ValuesIn(gSizes),
   ::testing::Bool()), ::testing::PrintToStringParamName());
class Unused : public ::testing::TestWithParam<int> {};
TEST_P(Unused, Never) { FAIL(); }
"""


def test_a_parameterised_test_runs_once_for_each_value_of_each_instantiation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, PARAMETERISED), capsys)
    assert status == 1
    names = re.findall(r"\[ RUN      \] (.+)", out)
    assert names == [
        "Few/Even.IsEven/0",
        "Few/Even.IsEven/1",
        "Even.IsEven/0",
        "Even.IsEven/1",
        "All/Named.Positive/one",
        "Grid/Pairs.Ok/(1, false)",
        "Grid/Pairs.Ok/(1, true)",
        "Grid/Pairs.Ok/(5, false)",
        "Grid/Pairs.Ok/(5, true)",
    ]
    assert "[  FAILED  ] Few/Even.IsEven/1, where GetParam() = 3 (0 ms)\n" in out
    assert "[  FAILED  ] Few/Even.IsEven/1, where GetParam() = 3\n" in out


def test_the_tests_are_listed_rather_than_run_when_asked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, PARAMETERISED + FAILING), capsys, "--gtest_list_tests")
    assert status == 0
    assert out.startswith("Bad.\n  Expect\n  Assert\nFew/Even.\n  IsEven/0  # GetParam() = 2\n")


SKIPS_AND_THROWS = """
TEST(Odd, Skips) { GTEST_SKIP() << "not today"; FAIL(); }
TEST(Odd, Dies) { EXPECT_DEATH(abort(), ""); }
TEST(Odd, Throws) { throw std::runtime_error("boom"); }
TEST(Odd, Missing) { ROOT::NoSuchThing(); }
TEST(Odd, Refused) { ::testing::Mock(); }
TEST(Odd, Wrong) {
   EXPECT_THROW(throw std::runtime_error("r"), std::invalid_argument);
   EXPECT_THROW(int a = 1, std::exception);
   EXPECT_NO_THROW(throw std::runtime_error("nt"));
   EXPECT_ANY_THROW(int b = 2);
   ADD_FAILURE() << "added";
   ADD_FAILURE_AT("other.cxx", 9);
}
"""


def test_a_skipped_test_and_a_death_test_are_reported_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = ran(program(tmp_path, SKIPS_AND_THROWS), capsys)
    assert "FILE:4: Skipped\nnot today\n[  SKIPPED ] Odd.Skips (0 ms)\n" in out
    assert (
        "FILE:5: Skipped\nThe death test of abort() is not run: xrdroot runs every test in one "
        "process, which a statement meant to end its process would end.\n"
        "[  SKIPPED ] Odd.Dies (0 ms)\n"
    ) in out
    assert "[  SKIPPED ] 2 tests, listed below:\n[  SKIPPED ] Odd.Skips\n" in out


def test_an_exception_or_a_refusal_in_a_test_fails_that_test_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, SKIPS_AND_THROWS), capsys)
    assert status == 1
    assert (
        'FILE:6: Failure\nC++ exception with description "boom" thrown in the test body.\n'
    ) in out
    assert "FILE:7: Failure\nAttributeError with description" in out
    assert (
        'FILE:8: Failure\nUnsupportedFeatureError with description "testing::Mock is a part of '
        'GoogleTest that xrdroot does not provide" thrown in the test body.\n'
    ) in out


def test_the_exception_assertions_say_what_was_thrown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = ran(program(tmp_path, SKIPS_AND_THROWS), capsys)
    assert (
        'Expected: throw std::runtime_error("r") throws an exception of type '
        'std::invalid_argument.\n  Actual: it throws RuntimeError with description "r".\n'
    ) in out
    assert "  Actual: it throws nothing.\n" in out
    assert (
        "Expected: throw std::runtime_error(\"nt\") doesn't throw an exception.\n"
        '  Actual: it throws RuntimeError with description "nt".\n'
    ) in out
    assert "Expected: int b = 2 throws an exception.\n  Actual: it doesn't.\n" in out
    assert "FILE:14: Failure\nFailed\nadded\nother.cxx:9: Failure\nFailed\n" in out


DIAGNOSTICS = """
TEST(Diag, Expected) {
   ROOT_EXPECT_ERROR(Error("Where", "what %d", 1), "Where", "what 1");
   ROOT_EXPECT_WARNING_PARTIAL(Warning("Here", "a long story"), "Here", "long");
   ROOT_EXPECT_INFO(Info("I", "said"), "I", "said");
   ROOT_EXPECT_NODIAG(printf("plain\\n"));
}
TEST(Diag, Wrong) {
   ROOT_EXPECT_SYSERROR(Info("Where", "only info"), "Where", "gone");
   ROOT_EXPECT_NODIAG(Warning("W", "loud"));
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.optionalDiag(kWarning, "Maybe", "perhaps");
   diags.requiredDiag(kError, "Must", "happen", false);
}
TEST(Diag, Declared) {
   ROOT::TestSupport::CheckDiagsRAII diags{kError, "Here", "said"};
   Error("Here", "said");
}
"""


def test_root_expect_error_takes_the_diagnostic_it_expects_and_fails_without_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["run", "-n", str(program(tmp_path, DIAGNOSTICS))])
    captured = capsys.readouterr()
    assert status == 1
    assert "[       OK ] Diag.Expected" in captured.out
    assert "[       OK ] Diag.Declared" in captured.out
    assert captured.err == "Info in <Where>: only info\nWarning in <W>: loud\n"
    assert (
        "ROOT::TestSupport::CheckDiagsRAII: Expected diagnostic message missing:\n"
        "SysError in <Where>: gone\n"
    ) in captured.out
    assert (
        "ROOT::TestSupport::CheckDiagsRAII: Unexpected diagnostic message received:\n"
        "Warning in <W>: loud\n"
    ) in captured.out
    assert (
        "unknown file: Failure\nROOT::TestSupport::CheckDiagsRAII: Expected diagnostic message "
        "missing:\nError in <Must>: happen\n[  FAILED  ] Diag.Wrong"
    ) in captured.out


OWN_MAIN = """
TEST(Own, Runs) { EXPECT_EQ(1, 1); }
int main(int argc, char **argv) {
   ::testing::InitGoogleTest(&argc, argv);
   printf("argc %d %s\\n", argc, argv[1]);
   return RUN_ALL_TESTS() + 3;
}
"""


def test_a_program_with_its_own_main_initialises_googletest_and_runs_the_tests(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, OWN_MAIN), capsys, "--gtest_filter=Own.*", "extra")
    assert status == 3
    assert out.startswith("argc 2 extra\nNote: Google Test filter = Own.*\n")


def test_a_failure_outside_any_test_fails_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = "TEST(A, B) {}\nint main() { EXPECT_EQ(1, 2); return RUN_ALL_TESTS(); }\n"
    status, out = ran(program(tmp_path, body), capsys)
    assert status == 1
    assert out.startswith("FILE:4: Failure\n") and out.endswith("[  PASSED  ] 1 test.\n")


def test_googletests_header_is_read_once_however_often_it_is_included() -> None:
    source = '#include <gtest/gtest.h>\n#include "gmock/gmock.h"\nTEST(A, B) {}\n'
    assert "A_B_registered" in translate(source)


def test_a_typed_test_is_refused_by_name() -> None:
    with pytest.raises(Refusal, match=r"^<macro>:2: TYPED_TEST, a GoogleTest typed test, "):
        translate("#include <gtest/gtest.h>\nTYPED_TEST(Suite, Name) {}\n")


def test_a_fixture_calls_getparam_as_its_own_method() -> None:
    python = translate(
        "#include <gtest/gtest.h>\nclass F : public ::testing::TestWithParam<int> {};\n"
        "TEST_P(F, A) { int x = GetParam(); }\n"
    )
    assert "x = int(self.GetParam())" in python


MATCHERS = """
MATCHER_P2(RelativeNear, expected, tolerance, "is near " + ::testing::PrintToString(expected))
{
   const double diff = std::fabs(arg - expected);
   if (diff <= tolerance * std::fabs(expected))
      return true;
   *result_listener << "off by " << diff;
   return false;
}
MATCHER_P(IsMultipleOf, n, "") { return arg % n == 0; }
MATCHER(IsPositive, "") { return arg > 0; }
TEST(Match, Defined) {
   EXPECT_THAT(1.01, RelativeNear(1.0, 0.1));
   EXPECT_THAT(2.0, RelativeNear(1.0, 0.1));
   EXPECT_THAT(7, IsMultipleOf(3));
   EXPECT_THAT(-1, IsPositive());
}
"""


def test_a_matcher_the_test_defines_matches_describes_and_explains_as_gmocks_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, out = ran(program(tmp_path, MATCHERS), capsys)
    assert status == 1
    assert (
        "Value of: 2.0\nExpected: is near 1\n  Actual: 2, off by 1\n"
        "FILE:17: Failure\nValue of: 7\nExpected: is multiple of 3\n  Actual: 7\n"
        "FILE:18: Failure\nValue of: -1\nExpected: is positive\n  Actual: -1\n"
    ) in out
