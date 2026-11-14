"""Run as PyROOT runs a script, ROOT's classes answer Python as PyROOT's pythonizations do."""

from __future__ import annotations

import warnings

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.cli import shims
from xrdroot.pyroot.core import messages


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(messages, "PYROOT_SCRIPT", True)


def test_outside_a_script_python_is_answered_as_cpp_is():
    assert not messages.from_script()


def test_in_a_script_python_is_the_caller_and_translated_cpp_is_not(script):
    assert messages.from_script(1)
    translated = compile("found = from_script(1)", "<translation of m.C>", "exec")
    space = {"from_script": messages.from_script}
    exec(translated, space)
    assert space["found"] is False


def test_in_a_script_a_warning_is_pythons_runtime_warning(script):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ROOT.Warning("Fit", "Fit data is empty")
        ROOT.Warning("", "nowhere")
    assert [(w.category, str(w.message), w.filename) for w in caught] == [
        (RuntimeWarning, "Fit data is empty", "Fit"), (RuntimeWarning, "nowhere", "")]  # fmt: skip


def test_in_a_script_a_file_that_will_not_open_is_an_os_error(script, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError, match=r"Failed to open file missing\.root"):
        ROOT.TFile("missing.root")
    with pytest.raises(OSError, match=r"Failed to open file missing\.root"):
        ROOT.TFile.Open(ROOT.TFile.AsyncOpen("missing.root"))


def test_an_async_open_handle_opens_the_file_when_open_is_given_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ROOT.TFile("there.root", "RECREATE").Close()
    handle = ROOT.TFile.AsyncOpen("there.root")
    assert handle.GetName() == "there.root"
    opened = ROOT.TFile.Open(handle)
    assert opened.GetName() == "there.root" and not opened.IsZombie()
    opened.Close()


def test_the_shim_turns_script_mode_on_and_says_what_it_was():
    before = shims.python_warnings(True)
    try:
        assert messages.PYROOT_SCRIPT and shims.python_warnings(True) is True
    finally:
        shims.python_warnings(before)
    assert messages.PYROOT_SCRIPT is before
