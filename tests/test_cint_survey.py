"""``tools/cint_survey.py``: how much of a tutorial tree translates, and why the rest does not."""

from __future__ import annotations

from pathlib import Path

import pytest
from tools import cint_survey


def test_the_survey_counts_what_translates_is_refused_and_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "good.C").write_text("void good() { int a = 1; }\n")
    (tmp_path / "jump.cxx").write_text("void jump() { goto end; }\n")
    (tmp_path / "odd.cpp").write_text("void odd() {}\n")
    (tmp_path / "notes.txt").write_text("not a macro\n")

    original = cint_survey.translate

    def broken(text: str, file: str) -> str:
        if file.endswith("odd.cpp"):
            raise RuntimeError("a bug")
        return original(text, file)

    monkeypatch.setattr(cint_survey, "translate", broken)
    assert cint_survey.main([str(tmp_path), "--list", "refused"]) == 0
    report = capsys.readouterr().out
    assert report.startswith(f"3 macros under {tmp_path}\n  translated     1   33.3%\n")
    assert "  refused        1   33.3%" in report and "  crashed        1   33.3%" in report
    assert "goto, which jumps to a label" in report
    assert "RuntimeError: a bug" in report
    assert "  jump.cxx: goto" in report


def test_a_reason_keeps_its_wording_and_drops_its_particulars() -> None:
    why = "the static local 'count' in f() on line 12 is refused"
    assert cint_survey.reason(why) == "the static local … in … on line … is refused"
