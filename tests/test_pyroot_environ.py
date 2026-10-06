"""``TEnv`` and ``gEnv``: resources read from ``.rootrc`` files, set in code, typed when asked."""

from __future__ import annotations

from pathlib import Path

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import environ


def test_gEnv_has_roots_defaults_and_the_rootrc_files_over_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert ROOT.gEnv.GetValue("Gui.BackgroundColor", "") == "#e8e8e8"
    home = tmp_path / "home"
    home.mkdir()
    (home / ".rootrc").write_text("Gui.BackgroundColor: #ffffff\n# a comment: here\nWeird\n")
    (tmp_path / ".rootrc").write_text("Unix.Local: yes\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    made = environ._global()
    assert made.GetValue("Gui.BackgroundColor", "") == "#ffffff"
    assert made.Lookup("Gui.BackgroundColor").GetLevel() == ROOT.kEnvUser
    assert made.GetValue("Unix.Local", 0) == 1 and not made.Defined("# a comment")


def test_a_value_comes_back_as_the_type_of_the_default_asked_with() -> None:
    env = ROOT.TEnv()
    env.SetValue("WebGui.HttpThrd", "yes")
    env.SetValue("Count", 3)
    env.SetValue("Ratio: 0.5", None)
    env.SetValue("Flag", False)
    assert env.GetValue("WebGui.HttpThrd", 0) == 1 and env.GetValue("Flag", True) == 0
    assert env.GetValue("Count", 0) == 3 and env.GetValue("Count", "") == "3"
    assert env.GetValue("Ratio", 1.0) == 0.5 and env.GetValue("Missing", 7) == 7
    record = env.Lookup("Count")
    assert (record.GetName(), record.GetValue(), record.GetLevel()) == ("Count", "3",
                                                                       ROOT.kEnvChange)  # fmt: skip
    assert [r.GetName() for r in env.GetTable()] == ["WebGui.HttpThrd", "Count", "Ratio", "Flag"]


def test_a_resource_file_is_read_written_and_listed(tmp_path: Path, capsys) -> None:
    path = tmp_path / "my.rc"
    path.write_text("A.b: 1\nC.d:   text  \n")
    env = ROOT.TEnv(str(path))
    assert env.GetRcName() == str(path) and env.GetValue("C.d", "") == "text"
    assert env.ReadFile(tmp_path / "none.rc") == -1
    env.SetValue("E.f", 2.5)
    env.WriteFile(tmp_path / "local.rc", ROOT.kEnvLocal)
    assert (tmp_path / "local.rc").read_text() == "A.b: 1\nC.d: text\n"
    env.WriteFile(tmp_path / "all.rc")
    assert (tmp_path / "all.rc").read_text().endswith("E.f: 2.5\n")
    env.Print()
    assert "A.b:" in capsys.readouterr().out
