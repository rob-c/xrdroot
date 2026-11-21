"""Which ``ProcessID`` record a reference written after a clone names.

``TRef::Streamer`` writes ``fPID`` as an index among the ``ProcessID`` records of the file it
is written into, and a process has no such index of its own: a file numbers the records in the
order it meets the processes that have objects in it, which a clone's carrying the copied
file's records changes (roottest/root/io/tree/test-rcr2). Only ``TFile::WriteProcessID`` can
say the index, so the clone that asks it leaves the answer where a reference written after it
finds it.
"""

from __future__ import annotations

import struct
from typing import Any

import xrdroot.pyroot as ROOT
from xrdroot.pyroot.trees import copying
from xrdroot.wbranch import tref_record


class Ref:
    """A reference as the writer is handed one: its number, its bits, and its process."""

    def __init__(self, number: int, pid: Any) -> None:
        self.number, self._pid = number, pid

    def GetUniqueID(self) -> int:
        return self.number

    def _bitword(self) -> int:
        return 0x03000000


class Process:
    """A process as a reference holds one: ROOT's ``TProcessID``, and whatever it was left."""

    def __init__(self, index: int, written: int | None = None) -> None:
        self.index = index
        if written is not None:
            self._file_index = written

    def GetUniqueID(self) -> int:
        return self.index


def fpid(pid: Any) -> int:
    """The ``fPID`` in one written record: ``TObject``'s version, its number, its bits, it."""
    return struct.unpack(">HIIH", tref_record(Ref(2, pid)))[3]


def test_a_reference_to_no_object_names_the_first_record():
    """A reference that names nothing was numbered by no process, and ROOT writes zero."""
    assert fpid(None) == 0


def test_a_process_no_file_has_numbered_names_its_index_in_the_session():
    """Nothing has asked a file for the process's record: the session's index is what a file
    whose own process comes first gives it, which is the case this writer has always written."""
    assert fpid(Process(0)) == 0
    assert fpid(Process(1)) == 1


def test_a_reference_names_the_record_the_file_gave_its_process():
    """The index a clone asked ``TFile::WriteProcessID`` for is the one a reference written
    after it names - here the session's second process, which is the file's first record."""
    assert fpid(Process(0, written=1)) == 1


def test_the_index_the_file_gave_a_process_is_left_on_it(monkeypatch, tmp_path):
    """``_process_index`` asks the file, and leaves the answer where a reference written later
    - by a branch of its own, after the clone's - finds it."""
    asked: list[Any] = []

    class File:
        def WriteProcessID(self, pid: Any) -> int:
            asked.append(pid)
            return 2

    monkeypatch.setattr(copying, "current_file", lambda: None, raising=False)
    from xrdroot.pyroot.core import directories

    monkeypatch.setattr(directories, "current_file", lambda: File())
    pid = ROOT.TProcessID("ProcessID0", "a-uuid")
    assert copying._process_index(pid) == 2
    assert asked == [pid]
    assert pid._file_index == 2


def test_nothing_is_being_written_so_a_process_is_left_as_it_is(monkeypatch):
    """A file opened to read has no records to give a process: the index is ``-1`` and the
    process is not told anything a later write would take for one."""
    from xrdroot.pyroot.core import directories

    monkeypatch.setattr(directories, "current_file", lambda: None)
    pid = ROOT.TProcessID("ProcessID0", "a-uuid")
    assert copying._process_index(pid) == -1
    assert not hasattr(pid, "_file_index")
