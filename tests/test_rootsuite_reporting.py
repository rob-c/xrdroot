"""Reports, the scope file, the ratchet and the command line, on synthetic results."""

from __future__ import annotations

import json

import pytest
from tools.rootsuite import cli, ratchet, report, scope
from tools.rootsuite.judge import STATUSES
from tools.rootsuite.model import Test


def record(test_id: str, status: str, kind: str = "roottest", **more) -> dict:
    return {"id": test_id, "ctest": test_id, "kind": kind, "area": "root/io", "status": status,
            **more}


def results(*records: dict, **meta) -> dict:
    return report.document(list(records), {"kinds": [], "only": [], **meta})


# --- reports -------------------------------------------------------------------


def test_the_document_counts_statuses_by_kind():
    doc = results(record("a", "PASS"), record("b", "FAIL"), record("c", "PASS", kind="gtest"))
    assert doc["schema"] == report.SCHEMA
    assert doc["counts"]["roottest"]["PASS"] == 1 and doc["counts"]["gtest"]["PASS"] == 1


def test_the_summary_gives_pass_rates_with_and_without_out_of_scope_tests():
    doc = results(record("a", "PASS"), record("b", "FAIL", reason="x.C:3: bad thing"),
                  record("c", "OUT-OF-SCOPE", reason="It is out."), record("d", "PASS"))
    text = report.summary(doc)
    assert "| roottest | 4 | 2 | 1 |" in text
    assert "50.0% | 66.7% |" in text
    assert "1: It is out." in text


def test_the_summary_lists_failure_causes_with_paths_and_numbers_made_generic():
    doc = results(record("a", "FAIL", reason="/w/a.C:3: no 'TRef' yet"),
                  record("b", "UNSUPPORTED", reason="/w/b.C:12: no 'TRef' yet"))
    assert "    2  <path> no 'TRef' yet" in report.summary(doc)
    assert report.cause("exit 2, expected 0") == "exit N, expected N"


def test_the_summary_counts_cases_and_ranks_causes_by_the_programs_each_alone_blocks():
    doc = results(record("a", "FAIL", kind="gtest", cases={"passed": 2, "failed": 2},
                         failures=["A.x: no 'TUUID' yet", "A.y: no 'TUUID' yet"]),
                  record("b", "FAIL", kind="gtest", cases={"passed": 0, "failed": 2},
                         failures=["B.x: no 'TUUID' yet", "B.y: exit 3"]),
                  record("c", "UNSUPPORTED", kind="gtest", reason="no 'TEnv' yet"),
                  record("d", "PASS", kind="gtest", cases={"passed": 1}))
    text = report.summary(doc)
    assert "| gtest | 4 | 1 | 2 | 0 | 1 | 0 | 0 | 0 | 25.0% | 25.0% | 3/7 |" in text
    assert "    1     2  no 'TUUID' yet" in text and "    1     1  no 'TEnv' yet" in text
    assert "    0     1  exit N" in text


def test_report_redraws_the_summary_from_results_json(tmp_path):
    report.write(results(record("a", "PASS")), tmp_path)
    (tmp_path / "summary.md").unlink()
    assert cli.main(["report", "--results", str(tmp_path / "results.json")]) == 0
    assert "ROOT test suite" in (tmp_path / "summary.md").read_text()


def test_report_refuses_results_of_another_schema(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps({"schema": "other"}))
    assert cli.main(["report", "--results", str(tmp_path / "results.json")]) == 2


# --- the scope file ------------------------------------------------------------------


def test_a_scope_label_covers_tests_by_id_glob_by_need_and_by_build_step(tmp_path):
    path = tmp_path / "scope.toml"
    path.write_text('[[label]]\nreason = "Out by id."\nids = ["roottest/cling/dict/*"]\n'
                    '[[label]]\nreason = "Out by need."\nneeds = ["dictionary"]\n'
                    'setup = ["compiled-macro"]\n')
    labels = scope.load(path)
    base = {"ctest": "c", "kind": "roottest", "directory": "d", "runner": "macro"}
    found = scope.label_for(Test(id="roottest/cling/dict/a/b", **base), labels)
    assert found is not None and found.reason == "Out by id."
    assert scope.label_for(Test(id="x", needs=("dictionary",), **base), labels)
    assert scope.label_for(Test(id="y", setup_of="compiled-macro", **base), labels)
    assert scope.label_for(Test(id="z", **base), labels) is None


def test_a_scope_reason_must_be_a_full_sentence(tmp_path):
    path = tmp_path / "scope.toml"
    path.write_text('[[label]]\nreason = "gui stuff"\nids = ["*"]\n')
    with pytest.raises(ValueError, match="full sentence"):
        scope.load(path)


def test_the_committed_scope_file_loads_with_a_sentence_for_every_label():
    labels = scope.load()
    assert labels and all(label.reason.endswith(".") for label in labels)


def test_the_committed_scope_holds_only_what_no_mac_with_apple_silicon_runs():
    labels = scope.load()
    assert all("macOS" in label.reason and not label.needs and not label.setup
               for label in labels)
    base = {"ctest": "c", "kind": "other", "directory": "d", "runner": "program"}
    assert scope.label_for(Test(id="other/tmva/DNN/TMVA-DNN-ArithmeticCuda", **base), labels)
    assert scope.label_for(Test(id="gtest/io/io/RIoUring", **base), labels)
    for kept in ("roottest/root/aclic/misc/x", "roottest/cling/dict/y", "gtest/core/meta/z",
                 "pyunittest/bindings/pyroot/cppyy/w", "roottest/python/distrdf/v"):
        assert scope.label_for(Test(id=kept, **base), labels) is None


def test_the_committed_baseline_holds_known_statuses_for_every_test_it_lists():
    baseline = json.loads(ratchet.BASELINE.read_text())
    assert baseline and set(baseline.values()) <= set(STATUSES)


# --- the ratchet ----------------------------------------------------------------------


def test_a_baseline_pass_that_fails_or_goes_missing_is_a_regression():
    doc = results(record("a", "FAIL"), record("b", "PASS"))
    failed = ratchet.regressions({"a": "PASS", "b": "PASS", "c": "PASS", "d": "FAIL"}, doc)
    assert [(r["id"], r["status"]) for r in failed] == [("a", "FAIL"), ("c", "MISSING")]


def test_a_baseline_pass_of_a_kind_this_run_did_not_run_is_not_a_regression():
    """``--kind roottest`` says nothing about the ``other`` kind, so its baseline PASSes are
    not missing - they were not asked for - while a kind the run covers is still judged."""
    doc = results(record("roottest/a", "FAIL", kind="roottest"),
                  record("roottest/b", "PASS", kind="roottest"), kinds=["roottest"])
    baseline = {"roottest/a": "PASS", "roottest/b": "PASS", "other/c": "PASS",
                "gtest/d": "PASS"}
    assert [r["id"] for r in ratchet.regressions(baseline, doc)] == ["roottest/a"]
    assert ratchet.judged(doc) == {"roottest"}
    assert ratchet.judged(results(record("a", "PASS"))) == set(ratchet.FULL_KINDS)


def test_a_baseline_with_unknown_statuses_is_refused():
    with pytest.raises(ValueError):
        ratchet.regressions({"a": "GREAT"}, results())


def test_a_baseline_needs_a_full_run_of_the_oracle_free_kinds(tmp_path):
    with pytest.raises(ValueError, match="full run"):
        ratchet.write_baseline(results(record("a", "PASS"), only=["x"]), tmp_path / "b.json")
    with pytest.raises(ValueError, match="full run"):
        ratchet.write_baseline(results(record("a", "PASS"), kinds=["gtest"]), tmp_path / "b.json")
    ratchet.write_baseline(results(record("a", "PASS")), tmp_path / "b.json")
    assert json.loads((tmp_path / "b.json").read_text()) == {"a": "PASS"}


def test_the_ratchet_command_fails_on_a_regression_and_passes_without_one(tmp_path, capsys):
    report.write(results(record("a", "PASS"), record("b", "TIMEOUT")), tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"a": "PASS"}))
    args = ["ratchet", str(tmp_path / "results.json"), "--baseline", str(baseline)]
    assert cli.main(args) == 0
    baseline.write_text(json.dumps({"a": "PASS", "b": "PASS"}))
    assert cli.main(args) == 1
    assert "b: PASS -> TIMEOUT" in capsys.readouterr().out


def test_the_baseline_command_writes_every_status_of_a_full_run(tmp_path):
    report.write(results(record("a", "PASS"), record("b", "SKIP")), tmp_path)
    out = tmp_path / "baseline.json"
    assert cli.main(["baseline", str(tmp_path / "results.json"), "--output", str(out)]) == 0
    assert json.loads(out.read_text()) == {"a": "PASS", "b": "SKIP"}


# --- listing ------------------------------------------------------------------------------


def test_list_prints_counts_per_kind_beside_ci_and_one_line_per_test(tmp_path, capsys):
    (tmp_path / "test").mkdir()
    (tmp_path / "test/CMakeLists.txt").write_text(
        "ROOT_EXECUTABLE(stressX stressX.cxx)\nROOT_ADD_TEST(test-stressx COMMAND stressX)\n")
    assert cli.main(["list", "--root", str(tmp_path), "--tests"]) == 0
    out = capsys.readouterr().out
    assert "test/stressx\ttest-stressx\tprogram" in out
    assert "test             1   CI: 64" in out


def test_the_tutorial_kind_is_handed_to_the_tutorial_harness(tmp_path, monkeypatch):
    import tools.tutorials.cli as tutorials

    given = []
    monkeypatch.setattr(tutorials, "main", lambda argv: given.append(argv) or 0)
    args = ["run", "--root", str(tmp_path), "--kind", "tutorial", "--out", str(tmp_path / "o")]
    assert cli.main(args) == 0
    parsed = tutorials.parser().parse_args(given[0])
    assert parsed.tutorials == tmp_path / "tutorials" and parsed.out == tmp_path / "o/tutorials"


def test_list_of_another_kind_is_empty_and_tutorials_alone_skip_discovery(tmp_path, capsys):
    assert cli.main(["list", "--root", str(tmp_path), "--kind", "gtest"]) == 0
    assert "gtest            0" in capsys.readouterr().out
    assert cli.main(["list", "--root", str(tmp_path), "--kind", "tutorial"]) == 0


def test_the_tutorial_count_is_the_tutorial_harness_catalogs(tmp_path, monkeypatch, capsys):
    import tools.tutorials.catalog as catalog

    (tmp_path / "tutorials").mkdir()
    seen = {"a": type("T", (), {"test": True, "vetoed": False})(),
            "b": type("T", (), {"test": True, "vetoed": True})()}
    monkeypatch.setattr(catalog, "catalog", lambda *args: seen)
    assert cli.main(["list", "--root", str(tmp_path), "--kind", "tutorial",
                     "--features", "x"]) == 0
    assert "tutorial         1" in capsys.readouterr().out


# --- running ---------------------------------------------------------------------------


def test_run_writes_the_results_and_summary_with_the_runs_settings(tmp_path, monkeypatch, capsys):
    (tmp_path / "test").mkdir()
    (tmp_path / "test/CMakeLists.txt").write_text("ROOT_ADD_TEST(test-true COMMAND true)\n")
    out = tmp_path / "out"
    assert cli.main(["run", "--root", str(tmp_path), "--out", str(out), "--features", "x",
                     "--xrdroot-src", str(tmp_path), "--jobs", "1"]) == 0
    doc = json.loads((out / "results.json").read_text())
    assert doc["tests"][0]["status"] == "PASS" and doc["meta"]["features"] == ["x"]
    assert doc["meta"]["xrdroot_commit"] == "" and (out / "summary.md").is_file()
    assert '"PASS": 1' in capsys.readouterr().out
    assert cli._commit(None) == ""
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert cli._commit(tmp_path) == ""


def test_timeouts_rerun_alone_are_excused_when_they_pass(tmp_path, monkeypatch, capsys):
    doc = results(record("a", "TIMEOUT"), record("b", "TIMEOUT"),
                  root=str(tmp_path), python="py", xrdroot_src=str(tmp_path), features=["x"])
    report.write(doc, tmp_path)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"a": "PASS", "b": "PASS"}))
    monkeypatch.setattr(cli, "discover", lambda root, features: [])

    class Rerun:
        def __init__(self, settings):
            assert settings.only == ("a", "b") and settings.jobs == 1

        def run(self, tests):
            return [record("a", "PASS"), record("b", "TIMEOUT")]

    monkeypatch.setattr(cli, "Runner", Rerun)
    args = ["ratchet", str(tmp_path / "results.json"), "--baseline", str(baseline),
            "--rerun-timeouts"]
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert "a: timed out, but passed when rerun alone" in out and "1 regressions" in out
    assert cli._recheck(doc, [], tmp_path) == set()
