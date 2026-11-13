"""Judging runs as ctest and RootTestDriver.cmake do, on synthetic results and .ref files."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools.rootsuite import cases, refdiff
from tools.rootsuite.judge import judge, refusal
from tools.rootsuite.model import Test
from tools.rootsuite.process import Result


def result(stdout: str = "", stderr: str = "", code: int | None = 0, **fields) -> Result:
    base = {"command": ["x"], "exit_code": code, "timed_out": False, "runtime": 0.5}
    return Result(**{**base, "stdout": stdout, "stderr": stderr, **fields})


def roottest(**fields) -> Test:
    base = {"id": "roottest/a/t", "ctest": "roottest-a-t", "kind": "roottest", "directory": "a",
            "runner": "macro", "custom_diff": True}
    return Test(**{**base, **fields})


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "a").mkdir()
    (tmp_path / "a/t.ref").write_text("Processing t.C...\nvalue = 1\n\nptr @0x1234 ok\n")
    (tmp_path / "a/t.err").write_text("Warning: careful\n")
    (tmp_path / "a/cnv.sh").write_text("sed -e 's/[0-9]/N/g'\n")
    return tmp_path


def status(test: Test, run: Result, root: Path) -> str:
    return judge(test, run, root, root).status


# --- the reference diff ----------------------------------------------------


def test_custom_diff_ignores_processing_lines_blank_lines_pointers_and_spaces():
    assert refdiff.custom_diff("Processing x.C...\na = 1\n\nptr @0xdead\n", "a=1\nptr\n") == ""


def test_custom_diff_shows_a_unified_diff_when_the_outputs_differ():
    shown = refdiff.custom_diff("a = 1\n", "a = 2\n")
    assert "-a = 1" in shown and "+a = 2" in shown


def test_custom_diff_keeps_its_space_class_quirk_dropping_lines_of_those_letters_only():
    assert refdiff.filtered(["pace\n", "  \n", "x\n"], ignore_spaces=False) == ["  \n", "x\n"]


def test_custom_diff_canonicalises_input_lines_and_libcxx_namespaces():
    ref, out = "input_line_12 std::__1::vector\n", "input_line_9 std::vector\n"
    assert refdiff.custom_diff(ref, out) == ""


def test_exact_diff_counts_any_difference():
    assert refdiff.exact("a\n", "a\n") == "" and refdiff.exact("a\n", "a \n")


def test_unexpected_errors_forgive_aclic_and_auto_registration_lines():
    errors = "Info in <TUnixSystem::ACLiC>: creating shared library x.so\nError: bad\n"
    assert refdiff.unexpected_errors(errors) == ["Error: bad"]


# --- the driver and ctest ----------------------------------------------------


def test_a_roottest_with_an_outref_passes_when_stdout_matches_after_filtering(root):
    assert status(roottest(outref="a/t.ref"), result("value=1\nptr ok\n"), root) == "PASS"


def test_a_roottest_with_an_outref_is_a_diff_when_stdout_differs(root):
    verdict = judge(roottest(outref="a/t.ref"), result("value = 2\nptr ok\n"), root, root)
    assert verdict.status == "DIFF" and "+value = 2" in verdict.diff


def test_an_errref_compares_the_error_stream_separately(root):
    test = roottest(errref="a/t.err")
    assert status(test, result("", "Warning: careful\n"), root) == "PASS"
    assert status(test, result("", "Warning: other\n"), root) == "DIFF"


def test_an_outcnv_script_converts_the_output_before_the_diff(root):
    (root / "a/n.ref").write_text("value = N\n")
    test = roottest(outref="a/n.ref", outcnv="a/cnv.sh")
    assert status(test, result("value = 7\n"), root) == "PASS"


def test_an_outcnvcmd_runs_on_the_file_holding_the_output(root):
    (root / "a/n.ref").write_text("kept\n")
    test = roottest(outref="a/n.ref", outcnvcmd=("sed", "-e", "/(int) 0/d"))
    assert status(test, result("kept\n(int) 0\n"), root) == "PASS"


def test_a_wrong_exit_code_fails_before_any_diff_and_passrc_sets_the_expected_one(root):
    assert status(roottest(outref="a/t.ref"), result("value = 1\n", code=1), root) == "FAIL"
    assert status(roottest(passrc=3), result(code=3), root) == "PASS"


def test_an_assert_macro_fails_on_unexpected_standard_error(root):
    test = roottest(check_out=False, check_err=True)
    assert status(test, result("anything", "Error in <X>: bad\n"), root) == "FAIL"
    assert status(test, result("anything", ""), root) == "PASS"


def test_willfail_turns_a_failure_into_a_pass_and_a_pass_into_a_failure(root):
    assert status(roottest(will_fail=True), result(code=1), root) == "PASS"
    assert status(roottest(will_fail=True), result(), root) == "FAIL"


def test_a_failregex_match_fails_even_a_clean_exit(root):
    test = roottest(failregex=("FAILED|Error in",))
    assert status(test, result("Test 3 ... FAILED\n"), root) == "FAIL"
    assert status(test, result("all OK\n"), root) == "PASS"


def test_a_passregex_decides_the_verdict_instead_of_the_exit_code(root):
    test = roottest(passregex=("dereference null pointer",))
    assert status(test, result("", "dereference null pointer\n", code=1), root) == "PASS"
    assert status(test, result("fine\n"), root) == "FAIL"


def test_a_signal_is_a_failure_that_says_so(root):
    verdict = judge(roottest(), result("", "Segmentation fault\n", code=-11), root, root)
    assert verdict.status == "FAIL" and "signal 11" in verdict.reason


def test_a_timeout_is_its_own_status(root):
    assert status(roottest(), result(code=None, timed_out=True), root) == "TIMEOUT"


def test_a_translator_refusal_exiting_two_is_unsupported(root):
    run = result("", "xrdroot run: x.C:3: typeid is not something this translator turns\n", 2)
    verdict = judge(roottest(), run, root, root)
    assert verdict.status == "UNSUPPORTED" and "typeid" in verdict.reason


def test_an_unsupported_feature_traceback_is_unsupported_whatever_the_exit(root):
    run = result("", "Traceback...\nxrdroot.errors.UnsupportedFeatureError: no TRef yet\n", 1)
    assert refusal(run) == "no TRef yet"


def test_a_usage_error_or_a_missing_file_is_a_failure_not_a_refusal(root):
    usage = result("", "usage: xrdroot\nxrdroot: error: unrecognized arguments: b.C\n", 2)
    missing = result("", "xrdroot ls: [Errno 2] No such file or directory: '-1'\n", 2)
    assert judge(roottest(), usage, root, root).reason.startswith("exit 2, expected 0: xrdroot")
    assert status(roottest(), missing, root) == "FAIL"


def test_a_gtest_failure_names_the_failing_case(root):
    run = result("[  FAILED  ] TH1.Fill\n[  PASSED  ] 3 tests.\n[  FAILED  ] 1 test, listed\n", "",
                 1)
    verdict = judge(roottest(runner="gtest", custom_diff=False), run, root, root)
    assert "TH1.Fill" in verdict.reason
    assert cases.gtest_counts(run.stdout) == {"passed": 3, "failed": 1}


# --- pyunittests ----------------------------------------------------------------

JUNIT = """<testsuites><testsuite>
<testcase name="test_ok"/>
<testcase name="test_skip"><skipped/></testcase>
<testcase name="test_bad"><failure message="AssertionError: 1 != 2">tb</failure></testcase>
<testcase name="test_refused"><error message="errors.UnsupportedFeatureError: no"/></testcase>
</testsuite></testsuites>"""


def test_junit_xml_gives_case_counts_and_first_lines_of_failures(tmp_path):
    (tmp_path / "j.xml").write_text(JUNIT)
    counts, messages = cases.junit(tmp_path / "j.xml")
    assert counts == {"passed": 1, "failed": 1, "errors": 1, "skipped": 1}
    assert messages[0] == "test_bad: AssertionError: 1 != 2"


def test_a_pyunittest_fails_with_its_case_counts_and_first_failure(tmp_path):
    (tmp_path / "j.xml").write_text(JUNIT)
    test = roottest(runner="pyunittest", kind="pyunittest")
    verdict = cases.judge_pyunittest(test, result(code=1), tmp_path / "j.xml")
    assert verdict.status == "FAIL" and verdict.cases["failed"] == 1
    assert verdict.reason.startswith("2 of 4 cases failed; first test_bad")


def test_a_pyunittest_failing_only_on_refusals_is_unsupported(tmp_path):
    xml = '<testsuite><testcase name="t"><failure message="UnsupportedFeatureError: x"/>'
    (tmp_path / "j.xml").write_text(xml + "</testcase></testsuite>")
    test = roottest(runner="pyunittest", kind="pyunittest")
    assert cases.judge_pyunittest(test, result(code=1), tmp_path / "j.xml").status == "UNSUPPORTED"


def test_a_pyunittest_passes_on_a_clean_exit_and_fails_when_nothing_was_collected(tmp_path):
    test = roottest(runner="pyunittest", kind="pyunittest")
    assert cases.judge_pyunittest(test, result(), tmp_path / "none.xml").status == "PASS"
    verdict = cases.judge_pyunittest(test, result(code=5), tmp_path / "none.xml")
    assert verdict.status == "FAIL" and "no tests" in verdict.reason
