"""``TChain::GetCurrentFile``: the file of the entry the chain read last.

ROOT's chain opens the file it reads an entry from and holds it - one file at
a time, the one before deleted as the next is opened - and, being in
``gROOT``'s list of cleanups, drops it when that file is deleted from
``gROOT``'s list of files. ``roottest/root/tree/chain/execCleanup.C`` is the
test of the last part: it empties ``gROOT->GetListOfFiles()`` and asks the
chain what file it points at.
"""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
from xrdroot.pyroot.core.files import TFile
from xrdroot.pyroot.trees import TChain, TTree


@pytest.fixture
def two_files(tmp_path):
    """Two files of the same tree, ten entries each: entries 0-9 and 10-19."""
    made = []
    for number in range(2):
        x = np.zeros(1)
        t = TTree("events", "")
        t.Branch("x", x, "x/D")
        for i in range(10):
            x[0] = number * 10 + i
            t.Fill()
        path = tmp_path / f"run{number}.root"
        with xrdroot.create(str(path)) as f:
            t.SetDirectory(f)
            t.Write()
        made.append(str(path))
    return made


@pytest.fixture
def chained(two_files):
    chain = TChain("events")
    for path in two_files:
        chain.Add(path)
    yield chain
    chain._destruct()
    chain._xrd.close()


def test_a_chain_holds_the_file_the_entry_it_read_came_from(chained, two_files):
    assert chained.GetCurrentFile() is None
    assert chained.GetEntry(0) > 0
    first = chained.GetCurrentFile()
    assert first is not None and first.GetName() == two_files[0]
    assert first in _groot_files()
    assert chained.GetEntry(15) > 0
    second = chained.GetCurrentFile()
    assert second is not None and second.GetName() == two_files[1]
    assert first not in _groot_files()  # let go as the next file was taken up


def test_a_chain_holds_no_file_once_groot_has_deleted_the_files_it_read(chained):
    chained.GetEntry(0)
    assert chained.GetCurrentFile() is not None
    _groot_files().Delete()
    assert chained.GetCurrentFile() is None


def test_a_chain_reads_on_after_the_file_it_read_was_deleted(chained, two_files):
    chained.GetEntry(0)
    held = chained.GetCurrentFile()
    _groot_files().Delete()
    assert chained.GetEntry(0) > 0
    again = chained.GetCurrentFile()
    assert again is not None and again is not held and again.GetName() == two_files[0]


def test_reading_a_chain_leaves_the_current_directory_where_it_was(tmp_path, chained, two_files):
    from xrdroot.pyroot import gDirectory

    out = TFile(str(tmp_path / "out.root"), "RECREATE")
    t = TTree("t", "")
    t.Branch("x", np.zeros(1), "x/D")
    t.Fill()
    before = gDirectory
    chained.GetEntry(0)
    # ROOT's LoadTree opens its file inside a TDirectory::TContext, so nothing moves
    assert gDirectory is before and chained.GetCurrentFile().GetName() == two_files[0]
    out.Close()


def test_a_chain_that_reads_no_entry_holds_no_file(chained):
    assert chained.GetEntry(99) == 0 and chained.GetCurrentFile() is None
    assert chained.LoadTree(99) == -2 and chained.GetCurrentFile() is None
    assert chained.LoadTree(12) == 2 and chained.GetCurrentFile() is not None


def test_a_chain_lets_go_of_the_file_it_opened_when_it_is_deleted(two_files):
    chain = TChain("events")
    chain.Add(two_files[0])
    chain.GetEntry(0)
    held = chain.GetCurrentFile()
    assert held is not None and held in _groot_files()
    chain._destruct()
    chain._xrd.close()
    assert not held.IsOpen() and held not in _groot_files()


def _groot_files():
    from xrdroot.pyroot import gROOT

    return gROOT.GetListOfFiles()
