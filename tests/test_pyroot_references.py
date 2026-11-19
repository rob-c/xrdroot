"""``TRef``, ``TRefArray``, ``TProcessID`` and ``TBits``: in memory, and through a file both ways.

``references-6.40.root`` was written by ROOT 6.40 (``references.C`` beside it):
a ``TNamed`` a ``TRef`` points at, the ``TRef``, a ``TRefArray`` of it and of a
bare ``TObject``, that ``TObject``, a ``TBits`` and ROOT's ``ProcessID0``. A
reference read from it finds the object read beside it, and what is written
here reads back the same way - in ROOT too, which ``roottest/root/io/references``
checks.
"""

from __future__ import annotations

import pathlib

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.pyroot.core import references
from xrdroot.pyroot.core.references import TProcessID

DATA = pathlib.Path(__file__).parent / "data"
ROOT_FILE = DATA / "references-6.40.root"
REFERENCED = 1 << 4


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch):
    """A session of its own: no process numbered anything yet."""
    monkeypatch.setattr(TProcessID, "_pids", [])
    monkeypatch.setattr(TProcessID, "_current", None)
    monkeypatch.setattr(TProcessID, "_number", 0)
    monkeypatch.setattr(ROOT.TRef, "_execs", [])


def test_a_reference_numbers_its_object_in_the_session_process_and_finds_it():
    target = ROOT.TNamed("t", "")
    ref = ROOT.TRef(target)
    session = TProcessID.GetSessionProcessID()
    assert ref.GetObject() is target and ref.GetPID() is session and ref.IsValid()
    assert target.TestBit(REFERENCED) and target.GetUniqueID() == 1 == ref.GetUniqueID()
    assert TProcessID.GetObjectCount() == 1 and TProcessID.AssignID(target) == 1
    assert ROOT.TRef(ref) == ref and ref == target and hash(ref) == hash(ROOT.TRef(target))
    assert session.GetObjects() == [target] and repr(session).startswith("<TProcessID")
    assert TProcessID.GetPIDs() == [session] and TProcessID.GetNProcessIDs() == 1
    assert TProcessID.GetProcessID(5) is None


def test_a_null_reference_and_one_of_a_process_gone_find_nothing():
    ref = ROOT.TRef()
    assert ref.GetObject() is None and not ref and not ref.IsValid()
    target = ROOT.TObject()
    ref.Set(target)
    assert ref
    ref.Clear()
    assert ref.GetUniqueID() == 0 and ref.GetObject() is None
    ref.Set(target)
    ref._pid = TProcessID("ProcessIDx", "not the session's")
    assert ref.GetObject() is None


def test_past_the_most_a_process_numbers_the_next_process_takes_over():
    first = ROOT.TObject()
    ROOT.TRef(first)
    TProcessID.SetObjectCount(references.MOST)
    later = ROOT.TObject()
    ref = ROOT.TRef(later)
    assert TProcessID.GetPID() is not TProcessID.GetSessionProcessID()
    assert TProcessID.GetObjectCount() == 1 and later.GetUniqueID() >> 24 == 1
    assert ref.GetObject() is later and ROOT.TRef(first).GetObject() is first


def test_a_process_past_the_255th_is_found_by_the_object_it_holds():
    for _ in range(references.FAR + 1):
        TProcessID.AddProcessID()
    far = TProcessID._pids[references.FAR]
    TProcessID._current = far
    target = ROOT.TObject()
    ROOT.TRef(target)
    assert target.GetUniqueID() >> 24 == references.FAR
    assert TProcessID.GetProcessWithUID(target) is far
    assert TProcessID.GetProcessWithUID(target.GetUniqueID(), target) is far


def test_an_object_python_cannot_point_at_weakly_is_held_outright():
    pid = TProcessID.GetSessionProcessID()
    pid.PutObjectWithID(7, 3)
    assert pid.GetObjectWithID(3) == 7
    target = ROOT.TObject()
    ROOT.TRef(target)
    pid.RecursiveRemove(target)
    pid.RecursiveRemove(target)
    assert pid.GetObjectWithID(target.GetUniqueID()) is None
    pid.Clear()
    assert pid.GetObjects() == []


def test_an_action_runs_when_the_object_is_not_there_and_execs_are_numbered(monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(ROOT.gROOT, "ProcessLine", lambda line: ran.append(line))
    ref = ROOT.TRef(ROOT.TObject())
    ref.SetAction("load()")
    TProcessID.GetSessionProcessID().Clear()
    assert ref.GetObject() is None and ran == ["load()"] and ref.GetAction() == "load()"
    ref.SetAction(ROOT.TObject())
    assert ref.GetObject() is None and ran == ["load()"]
    assert ROOT.TRef.AddExec("a") == 1 and ROOT.TRef.AddExec("b") == 2
    assert ROOT.TRef.AddExec("a") == 1 and ROOT.TRef.GetListOfExecs() == ["a", "b"]


def _filled() -> tuple[ROOT.TRefArray, ROOT.TNamed, ROOT.TNamed]:
    first, second = ROOT.TNamed("a", ""), ROOT.TNamed("b", "")
    array = ROOT.TRefArray()
    array.Add(first)
    array.AddAt(second, 3)
    array.AddAt(None, 1)
    return array, first, second


def test_a_reference_array_keeps_numbers_in_slots_and_finds_their_objects():
    array, first, second = _filled()
    assert list(array) == [first, None, None, second]
    assert len(array) == 4
    assert array[3] is second
    assert array.At(9) is None
    assert array.UncheckedAt(0) is first
    assert (array.GetEntries(), array.GetLast(), array.GetAbsLast()) == (2, 3, 3)
    assert array.GetUID(3) == second.GetUniqueID()
    assert (array.IndexOf(second), array.IndexOf(ROOT.TObject())) == (3, -1)
    assert array.GetLowerBound() == 0


def test_a_reference_array_lets_go_of_slots_and_closes_them_up():
    array, first, second = _filled()
    assert array.RemoveAt(0) is first
    assert array.Remove(second) is second
    assert array.Remove(first) is None
    assert array.IsEmpty()
    array.Add(first)
    array.Compress()
    assert list(array) == [first]
    array.SetName("refs")
    array.Clear()
    assert array.GetName() == "refs" and array.GetEntriesFast() == 0


def test_an_object_of_another_process_is_refused_by_a_reference_array(capsys):
    array = ROOT.TRefArray()
    stranger = ROOT.TObject()
    other = TProcessID.AddProcessID()
    other.PutObjectWithID(stranger, 1)
    stranger.SetUniqueID(1 + (other.GetUniqueID() << 24))
    stranger.SetBit(REFERENCED)
    array.Add(stranger)
    array.AddAt(stranger, 2)
    uuid = ROOT.TObject()
    uuid.SetBit(references.HAS_UUID)
    array.Add(uuid)
    assert array.GetEntries() == 0
    err = capsys.readouterr().err
    assert err.count("is not registered in the process the TRefArray points to") == 3
    assert f"(pid = ProcessID0/{TProcessID.GetSessionProcessID().GetTitle()})" in err
    assert ROOT.TRefArray(other).GetPID() is other


def test_bits_grow_as_roots_do_and_say_which_are_set(capsys):
    bits = ROOT.TBits()
    assert (bits.GetNbits(), bits.GetNbytes()) == (8, 1)
    bits.SetBitNumber(13)
    bits[3] = True
    assert (bits.GetNbits(), bits.GetNbytes(), bits.CountBits()) == (14, 4, 2)
    assert (bits.FirstSetBit(), bits.FirstNullBit(), bits.LastSetBit()) == (3, 0, 13)
    assert bits[13] and not bits.TestBitNumber(99) and bits.CountBits(4) == 1
    bits.Print()
    assert capsys.readouterr().out == " bit:   3 = 1\n bit:  13 = 1\n"



def test_bits_compact_reset_and_clear_as_roots_do():
    bits = ROOT.TBits()
    bits.SetBitNumber(13)
    bits[3] = True
    bits.SetBitNumber(13, False)
    bits.ResetBitNumber(99)
    bits.Compact()
    assert (bits.GetNbits(), bits.GetNbytes()) == (8, 1)
    bits.Compact()
    assert bits.GetNbits() == 8 and ROOT.TBits(bits) == bits
    assert hash(bits) == hash(ROOT.TBits(bits))
    bits.ResetAllBits(True)
    assert bits.CountBits() == 0 and bits.FirstNullBit() == 0  # ROOT clears, whatever it is told
    bits.Clear()
    assert bits.GetNbits() == 0 and ROOT.TBits(0).GetNbits() == 8
    assert ROOT.TBits().LastSetBit() == 8 and ROOT.TBits() != 3  # none set: fNbits, as ROOT


def test_what_root_wrote_reads_back_each_reference_finding_its_object():
    found = ROOT.TFile(str(ROOT_FILE))
    target, ref, array = found.Get("therefobj"), found.Get("theref"), found.Get("thearray")
    other, bits = found.Get("other"), found.Get("bits")
    assert ref.GetObject() is target and target.GetName() == "target"
    assert list(array) == [target, other] and array.GetPID() is ref.GetPID()
    assert found.ReadProcessID(0) is ref.GetPID() and found.ReadProcessID(4) is None
    assert ref.GetPID() is not TProcessID.GetSessionProcessID()
    assert found.Get("ProcessID0") is ref.GetPID() and found.WriteProcessID(None) == -1
    assert [bit for bit in range(bits.GetNbits()) if bits[bit]] == [1, 17]
    assert bits.GetNbytes() == 6
    found.Close()


def test_what_is_written_here_reads_back_each_reference_finding_its_object(tmp_path):
    path = str(tmp_path / "refs.root")
    out = ROOT.TFile(path, "RECREATE")
    target, other = ROOT.TNamed("target", "t"), ROOT.TObject()
    ref, array, bits = ROOT.TRef(target), ROOT.TRefArray(), ROOT.TBits(5)
    array.Add(target)
    array.Add(other)
    bits.SetBitNumber(17)
    for name, obj in [("o", target), ("r", ref), ("a", array), ("x", other), ("b", bits)]:
        out.WriteObject(obj, name)
    assert out.WriteProcessID(None) == 0 and out.WriteProcessID(TProcessID.AddProcessID()) == 1
    out.Close()
    TProcessID._pids[0].SetTitle("another session")  # as a new session reading it would be
    back = ROOT.TFile(path)
    read, plain = back.Get("o"), back.Get("x")  # in memory, as a reference finds it only then
    assert read is not target and back.Get("r").GetObject() is read
    assert list(back.Get("a")) == [read, plain] and back.Get("b")[17]
    with xrdroot.open_root(path) as engine:
        assert engine.classnames()["ProcessID0"] == "TProcessID"
        assert engine["r"]["fPID"] == 0 and engine["o"]["TObject"]["pidf"] == 0
    back.Close()


def test_updating_a_file_finds_its_processes_and_adds_none_twice(tmp_path):
    path = str(tmp_path / "update.root")
    out = ROOT.TFile(path, "RECREATE")
    out.WriteObject(ROOT.TRef(ROOT.TObject()), "r")
    out.Close()
    again = ROOT.TFile(path, "UPDATE")
    assert again.WriteProcessID(TProcessID.GetSessionProcessID()) == 0
    again.Close()
    with xrdroot.open_root(path) as engine:
        assert [k for k in engine.keys() if k.startswith("ProcessID")] == ["ProcessID0"]


def test_settling_ignores_what_no_file_or_record_names():
    ref = ROOT.TRef()
    references.settle(ref, None, None)
    references.settle(ref, {"fPID": 0}, None)
    assert ref.GetPID() is None


def test_a_null_reference_and_a_list_of_references_are_written_and_read(tmp_path):
    path = str(tmp_path / "null.root")
    out = ROOT.TFile(path, "RECREATE")
    out.WriteObject(ROOT.TRef(), "null")
    held = ROOT.TList()
    target = ROOT.TNamed("t", "")
    held.Add(ROOT.TRef(target))
    held.Write("refs", ROOT.TObject.kSingleKey)
    out.Close()
    with xrdroot.open_root(path) as engine:
        assert engine["null"]["fPID"] == 0
        assert engine["refs"][0]["TObject"]["fUniqueID"] == target.GetUniqueID()
        assert engine.classnames()["refs"] == "TList"


def test_a_record_naming_a_process_its_file_lacks_is_left_unresolved():
    found = ROOT.TFile(str(ROOT_FILE))
    ref = ROOT.TRef()
    references.settle(ref, {"fPID": 7}, found)
    plain = ROOT.TObject()
    references.settle(plain, {"fUniqueID": 3, "fBits": REFERENCED, "pidf": 7}, found)
    assert ref.GetPID() is None and not plain.TestBit(REFERENCED)
    found.Close()


def test_bits_read_without_their_bytes_and_grown_within_their_bytes_keep_count():
    bits = ROOT.TBits()
    bits.SetBitNumber(13)
    bits.SetBitNumber(20)  # past fNbits, within the bytes the last growth made
    assert (bits.GetNbits(), bits.GetNbytes()) == (21, 4)
    made = ROOT.TBits._from_members({"fNbits": 3})
    assert made.GetNbits() == 3 and made.GetNbytes() == 1


def test_the_branch_style_is_the_trees_to_keep():
    ROOT.TTree.SetBranchStyle(0)
    assert ROOT.TTree.GetBranchStyle() == 0
    ROOT.TTree.SetBranchStyle()
    assert ROOT.TTree.GetBranchStyle() == 1


def test_bits_combine_shift_and_print_as_roots_operators_do():
    """What ROOT 6.40 printed for the same operations (``roottest/root/collection/runtbits.C``)."""
    a, b = ROOT.TBits(), ROOT.TBits()
    for bit in (2, 12):
        a.SetBitNumber(bit)
    for bit in (2, 15):
        b.SetBitNumber(bit)
    assert a.cxx_ostream() == "00000000000000000001000000000100"
    assert (a & b).cxx_ostream() == "00000000000000000000000000000100"
    assert (a ^ b).cxx_ostream() == "00000000000000001001000000000000"
    either = a | b
    assert (either << 2).cxx_ostream() == "00000000000000100100000000010000"
    assert ((either << 2) >> 2).cxx_ostream() == either.cxx_ostream()


def test_bits_of_other_sizes_combine_and_shift_as_roots_do():
    small, wide = ROOT.TBits(), ROOT.TBits(20)
    wide.SetBitNumber(3)
    wide.SetBitNumber(15)
    assert ((small | wide).GetNbits(), (small | wide).CountBits()) == (8, 1)
    assert (wide.LastSetBit(10), wide.LastSetBit(2), wide.LastNullBit(3)) == (3, 20, 2)
    wide <<= 2
    assert (wide.FirstSetBit(), wide.LastSetBit(), wide.GetNbits()) == (5, 17, 22)
    wide >>= 3
    assert (wide.FirstSetBit(), wide.LastSetBit(), wide.GetNbits()) == (2, 14, 19)
    assert ((~wide).CountBits(), (~wide).GetNbits()) == (22, 19)


def test_a_reference_carries_the_memory_bits_of_an_object_made_on_the_heap():
    # ROOT's ``TObject()`` asks ``TStorage::UpdateIsOnHeap`` where it was made; a ``TRef``
    # read back from a file is made by ``TClass::New``, on the heap, so it carries both bits.
    assert ROOT.TRef().TestBits(0xFFFFFFFF) == references.HEAP_BITS


def test_set_action_numbers_the_action_in_the_bits_and_runs_what_it_numbers(monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(ROOT.gROOT, "ProcessLine", lambda line: ran.append(line))
    assert ROOT.TRef.AddExec("load()") == 1 and ROOT.TRef.AddExec("fill()") == 2
    first = ROOT.TRef(ROOT.TObject())
    first.SetAction("load()")
    second = ROOT.TRef(ROOT.TObject())
    second.SetAction("fill()")
    assert first.TestBits(0xFF0000) == 1 << 16 and second.TestBits(0xFF0000) == 2 << 16
    # A reference read back from a file knows only the number in its bits.
    read = references._tref_from({"TObject": {"fUniqueID": 1, "fBits": 2 << 16}})
    read._pid = TProcessID.GetSessionProcessID()
    TProcessID.GetSessionProcessID().Clear()
    assert read.GetObject() is None and ran == ["fill()"]


def test_an_action_no_bit_numbers_and_one_past_the_execs_run_nothing(monkeypatch):
    ran: list[str] = []
    monkeypatch.setattr(ROOT.gROOT, "ProcessLine", lambda line: ran.append(line))
    ROOT.TRef.AddExec("load()")
    session = TProcessID.GetSessionProcessID()
    session.Clear()
    unnumbered = ROOT.TRef(ROOT.TObject())
    past_the_end = ROOT.TRef(ROOT.TObject())
    past_the_end.SetBit(9 << 16)
    assert unnumbered.GetObject() is None and past_the_end.GetObject() is None
    assert ran == []


def test_assigning_into_a_reference_keeps_the_bits_the_action_is_numbered_in():
    # ``fReference = obj`` in C++ writes into the reference - ``TRef::operator=`` - so the
    # bits, and the action numbered in them, stay; only the number and the process move.
    ROOT.TRef.AddExec("load()")
    numbered = ROOT.TRef(ROOT.TObject())
    numbered.SetAction("load()")
    target = ROOT.TNamed("t", "")
    numbered._assign(target)
    assert numbered.GetObject() is target and numbered.TestBits(0xFF0000) == 1 << 16
    other = ROOT.TRef(ROOT.TNamed("other", ""))
    numbered._assign(other)
    assert numbered.GetPID() is other.GetPID()
    assert numbered.GetUniqueID() == other.GetUniqueID() and numbered.TestBits(0xFF0000) == 1 << 16
