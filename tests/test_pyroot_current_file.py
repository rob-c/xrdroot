"""``gFile``: the current file, as ``gDirectory`` is the current directory.

It is the file of the directory last gone into that was in one - going back
to memory leaves it be, as in ROOT - and it is null, so false, once that file
closes. Going above the top of a file fails quietly, as ROOT's ``cd("..")`` does.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core.directories import current_file, forget_file


@pytest.fixture(autouse=True)
def _no_current_file():
    forget_file(current_file())
    yield
    forget_file(current_file())


def test_gfile_is_the_file_last_gone_into_and_null_once_it_closes(tmp_path, capsys):
    assert not ROOT.gFile and repr(ROOT.gFile) == "None"
    with pytest.raises(AttributeError, match="gFile is null"):
        ROOT.gFile.GetName()
    first = ROOT.TFile(str(tmp_path / "a.root"), "RECREATE")
    sub = first.mkdir("sub")
    sub.cd()
    assert ROOT.gFile == first and ROOT.gFile.GetName().endswith("a.root")
    assert ROOT.gFile.__real__() is first and ROOT.gFile == ROOT.gFile
    assert hash(ROOT.gFile) == hash(ROOT.gFile)
    ROOT.gROOT.cd()
    assert ROOT.gFile == first
    first.Close()
    assert not ROOT.gFile


def test_above_the_top_of_a_file_is_nowhere_and_no_error_says_so(tmp_path, capsys):
    first = ROOT.TFile(str(tmp_path / "a.root"), "RECREATE")
    assert first.cd("..") is False
    assert capsys.readouterr().err == ""
    assert first.cd("nowhere") is False
    assert "Unknown directory nowhere" in capsys.readouterr().err
    first.Close()


def test_cd_static_goes_by_a_files_path_or_from_where_one_is(tmp_path, capsys):
    first = ROOT.TFile(str(tmp_path / "cd.root"), "RECREATE")
    first.mkdir("top").mkdir("sub")
    name = first.GetName()
    assert ROOT.TDirectory.Cd(f"{name}:/top")
    assert ROOT.gDirectory.GetName() == "top"
    assert ROOT.TDirectory.Cd("sub") and ROOT.gDirectory.GetName() == "sub"
    assert ROOT.TDirectory.Cd("..") and ROOT.gDirectory.GetName() == "top"
    assert not first.cd("top/sub/deeper")
    assert capsys.readouterr().err == "Error in <TDirectoryFile::cd>: Unknown directory deeper\n"
    assert not ROOT.TDirectory("memory", "").cd("nowhere.root:/top")
    assert "Unknown directory nowhere.root:/top" in capsys.readouterr().err
    first.Close()


def test_what_was_printed_before_a_message_comes_out_before_it(capsys, monkeypatch):
    flushed: list[bool] = []
    monkeypatch.setattr("sys.stdout.flush", lambda: flushed.append(True))
    ROOT.Error("where", "what")
    assert flushed == [True]


def test_assigning_gdirectory_and_gfile_moves_them(tmp_path):
    from xrdroot.pycint.runtime.root import ROOT as HOST

    first = ROOT.TFile(str(tmp_path / "assign.root"), "RECREATE")
    below = first.mkdir("below")
    ROOT.gROOT.cd()
    HOST.gDirectory = below
    assert ROOT.gDirectory == below and ROOT.gDirectory.GetName() == "below"
    HOST.gDirectory = ROOT.gDirectory
    assert ROOT.gDirectory == below
    HOST.gFile = 0
    assert not ROOT.gFile
    HOST.gFile = first
    assert ROOT.gFile == first
    first.Close()


def test_keys_are_found_by_cycle_here_or_below_and_listed_by_name(tmp_path):
    path = str(tmp_path / "keys.root")
    out = ROOT.TFile(path, "RECREATE")
    named = ROOT.TNamed("bla", "blub")
    for _ in range(3):
        named.Write()
    below = out.mkdir("below")
    below.cd()
    ROOT.TNamed("deep", "").Write()
    out.Close()
    back = ROOT.TFile(path)
    assert back.FindKey("bla").GetCycle() == 3 and back.FindKey("bla;2").GetCycle() == 2
    assert back.FindKeyAny("deep").GetName() == "deep" and back.FindKeyAny("none") is None
    listed = back.GetListOfKeys().GetListForObject("bla")
    assert [key.GetCycle() for key in listed] == [3, 2, 1]
    assert len(back.GetListOfKeys().GetListForObject(named)) == 3
    back.Close()
