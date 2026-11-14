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
