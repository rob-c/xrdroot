"""The refusal report is the worklist for getting the refusals to zero, so it needs a guard.

A wrong answer and a refusal are different things: only the second is a name xrdroot lacks,
and the report has to group those by the reason rather than by the frame it was raised in.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.tutorials import unsupported


def _run(*records: dict[str, object]) -> dict[str, object]:
    return {"tutorials": list(records)}


def test_the_reason_is_the_exception_line_rather_than_a_traceback_frame():
    record = {
        "path": "a.py",
        "status": "UNSUPPORTED",
        "details": [
            '  File "xrdroot/cli/run.py", line 180, in run',
            "AttributeError: ROOT has TMacro; xrdroot.pyroot does not yet",
        ],
    }
    assert unsupported.cause(record) == (
        "AttributeError: ROOT has TMacro; xrdroot.pyroot does not yet"
    )


def test_two_refusals_of_one_thing_group_though_their_numbers_differ():
    said = "AttributeError: ROOT has TFoam; xrdroot.pyroot does not yet"
    first = {"path": "a.C", "status": "UNSUPPORTED", "details": [f"x.C:81: {said}"]}
    second = {"path": "b.C", "status": "UNSUPPORTED", "details": [f"x.C:19: {said}"]}
    found = unsupported.causes(_run(first, second))
    assert len(found) == 1
    assert sorted(found[0][1]) == ["a.C", "b.C"]


def test_a_wrong_answer_is_not_a_refusal_and_is_left_out():
    """A FAIL is a wrong answer: xrdroot has the thing, so it is not on this list."""
    found = unsupported.causes(
        _run(
            {"path": "a.C", "status": "FAIL", "details": ["total = 12"]},
            {"path": "b.C", "status": "PASS", "details": []},
        )
    )
    assert found == []


def test_the_report_counts_the_refusals_and_names_the_first_tutorial():
    said = "AttributeError: ROOT has TMacro; xrdroot.pyroot does not yet"
    found = unsupported.report(_run({"path": "z.C", "status": "UNSUPPORTED", "details": [said]}))
    assert found.startswith("# 1 refusals in 1 causes")
    assert f"| 1 | `{said}` | z.C |" in found


def test_a_run_is_read_from_the_results_file(tmp_path):
    path = Path(tmp_path) / "results.json"
    said = "not supported"
    path.write_text(json.dumps(_run({"path": "a.C", "status": "UNSUPPORTED", "details": [said]})))
    assert unsupported.read(path)["tutorials"][0]["path"] == "a.C"


def test_the_command_prints_the_report_for_a_run(tmp_path, capsys):
    path = Path(tmp_path) / "results.json"
    said = "x is not supported"
    path.write_text(json.dumps(_run({"path": "a.C", "status": "UNSUPPORTED", "details": [said]})))
    status = unsupported.run(type("Args", (), {"results": path})())
    assert status == 0
    assert "refusals" in capsys.readouterr().out
