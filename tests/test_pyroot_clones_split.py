"""A ``TClonesArray`` split into a branch per member, as ``TTree::Branch`` splits one.

The top branch counts the objects each entry holds (``fType`` 3) and each
member of every object is a branch of its own (``fType`` 31) - the objects'
``TObject`` part split into ``fUniqueID`` and ``fBits``, a ``TString`` member a
row of strings. ``TTree::Print`` of such a tree is ROOT 6.40's, line for line,
and ROOT reads the file back object for object (checked with the oracle when
this was written); here it is read back by xrdroot both ways.
"""

from __future__ import annotations

import struct
from typing import Any, ClassVar

import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core.declared import find_class
from xrdroot.pyroot.trees import copying, objects
from xrdroot.pyroot.trees.objectinfo import object_info
from xrdroot.wbranch import CloneMember, SplitClones, clones_members, tref_record
from xrdroot.wclasses import declared

#: ``TTree::Print`` of ROOT 6.40 for the tree :func:`_written` makes, its sizes and all.
PRINTED = """\
*Br    0 :N         : Int_t N_                                               *
*Entries :        3 : Total  Size=       3322 bytes  File Size  =         97 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.00     *
*............................................................................*
*Br    1 :N.fUniqueID : UInt_t fUniqueID[N_]                                 *
*Entries :        3 : Total  Size=        708 bytes  File Size  =        106 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.12     *
*............................................................................*
*Br    2 :N.fBits   : UInt_t fBits[N_]                                       *
*Entries :        3 : Total  Size=        688 bytes  File Size  =        102 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.13     *
*............................................................................*
*Br    3 :N.fName   : TString fName[N_]                                      *
*Entries :        3 : Total  Size=        700 bytes  File Size  =        124 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
*Br    4 :N.fTitle  : TString fTitle[N_]                                     *
*Entries :        3 : Total  Size=        705 bytes  File Size  =        125 *
*Baskets :        1 : Basket Size=      32000 bytes  Compression=   1.02     *
*............................................................................*
"""


def _written(path: str) -> None:
    """Three entries of one, two and three ``TNamed``, the branch split as ROOT splits it."""
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TNamed")
    tree.Branch("N", held)
    for entry in range(3):
        held.Clear()
        for at in range(entry + 1):
            held.AddAt(ROOT.TNamed(f"N_{entry}_{at}", f"N_{entry}_{at}"), at)
        tree.Fill()
    tree.Write()
    tree.Print()
    out.Close()


def test_a_split_clones_array_is_printed_as_root_prints_it(tmp_path, capsys):
    _written(str(tmp_path / "clones.root"))
    assert PRINTED in capsys.readouterr().out


def test_each_member_reads_back_as_a_row_per_entry(tmp_path):
    path = tmp_path / "clones.root"
    _written(str(path))
    with xrdroot.open_root(path) as found:
        tree = found["T"]
        assert tree.keys() == ["N", "N.fUniqueID", "N.fBits", "N.fName", "N.fTitle"]
        assert tree.arrays(["N.fName"])["N.fName"][2] == ["N_2_0", "N_2_1", "N_2_2"]
        record = tree.branches["N.fBits"].record
        assert record.classname == "TObject"
        assert record.element[1:] == ("", 2417737773, 1, 1, 31, 15, 0)


def test_a_script_reads_the_objects_back_into_its_clones_array_or_a_new_one(tmp_path):
    path = str(tmp_path / "clones.root")
    _written(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    mine = ROOT.TClonesArray("TNamed")
    tree.SetBranchAddress("N", mine)
    tree.GetEntry(1)
    assert [o.GetName() for o in mine] == ["N_1_0", "N_1_1"] and mine[1].GetTitle() == "N_1_1"
    back.Close()
    back = ROOT.TFile(path)
    tree = back.Get("T")
    cell = ROOT.MakeNullPointer(ROOT.TClonesArray) if hasattr(ROOT, "MakeNullPointer") else None
    holder = type("Cell", (), {"value": cell})()
    tree.SetBranchAddress("N", holder)
    tree.GetEntry(2)
    assert holder.value.GetClass().GetName() == "TNamed" and len(holder.value) == 3
    back.Close()


def test_long_names_go_out_with_four_bytes_of_length(tmp_path):
    path = str(tmp_path / "long.root")
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("TObjString")
    tree.Branch("S", held, 32000, 1)
    held.AddAt(ROOT.TObjString("x" * 300), 0)
    tree.Fill()
    tree.Write()
    out.Close()
    with xrdroot.open_root(path) as found:
        assert found["T"].arrays(["S.fString"])["S.fString"][0] == ["x" * 300]


def test_a_class_without_a_description_or_with_members_not_split_here_is_refused():
    with pytest.raises(UnsupportedFeatureError, match="carries a description of"):
        clones_members("MyMacroClass")
    with pytest.raises(UnsupportedFeatureError, match="TH1::fXaxis is a Object member"):
        clones_members("TH1F")
    names = [(m.owner, m.name) for m in clones_members("TLine")]
    assert ("TAttLine", "fLineColor") in names and ("TLine", "fX1") in names


def _cloned(tmp_path, name: str, entries: int = -1, cut: str | None = None) -> ROOT.TTree:
    """The tree of a written file, cloned or copied by a cut into a file of its own."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CopyTree(cut) if cut else upstream.CloneTree(entries)
    clone.SetName(name)
    source.Close()
    with xrdroot.create(str(tmp_path / "clone.root")) as out:
        clone.SetDirectory(out)
        clone.Write()
    return clone


def test_a_clone_hands_the_writer_one_branch_for_the_split_collection(tmp_path):
    """ROOT writes a split ``TClonesArray`` as a count and a branch per member, so a clone
    of one is that same branch again, not a column for each member of its own.

    A member's column is ``N.fName``, and no column of a tree may be named with a dot.
    """
    clone = _cloned(tmp_path, "C")
    assert clone.GetEntries() == 3
    assert list(clone._store.slots) == ["N"] and "ClonedSlot" in repr(clone._store.slots["N"])
    assert sorted(clone._store.columns()) == [
        "N",
        "N\x00N.fBits",
        "N\x00N.fName",
        "N\x00N.fTitle",
        "N\x00N.fUniqueID",
    ]
    with xrdroot.open_root(str(tmp_path / "clone.root")) as found:
        written = found["C"]
        assert written.keys() == ["N", "N.fUniqueID", "N.fBits", "N.fName", "N.fTitle"]
        assert written.arrays(["N.fName"])["N.fName"][2] == ["N_2_0", "N_2_1", "N_2_2"]
        assert written.branches["N"].record.element == ("", "TNamed", 3066088035, 4, 0, 3, -1, 3)
        bits = written.branches["N.fBits"].record.element
        assert bits == ("TNamed", "", 2417737773, 1, 1, 31, 15, 0)


def test_a_clone_filled_entry_by_entry_copies_the_objects_each_entry_holds(tmp_path):
    """``CloneTree(0)`` then ``GetEntry`` and ``Fill``: the clone reads its objects from the
    entry the source last read, as ROOT's clone shares its source's branch buffers."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CloneTree(0)
    assert upstream.GetEntry(2)
    assert clone.Fill()
    source.Close()
    with xrdroot.create(str(tmp_path / "filled.root")) as out:
        clone.SetDirectory(out)
        clone.Write()
    with xrdroot.open_root(str(tmp_path / "filled.root")) as found:
        assert found["T"].arrays(["N.fName"])["N.fName"] == [["N_2_0", "N_2_1", "N_2_2"]]


def test_a_copy_tree_of_a_split_collection_keeps_the_entries_the_cut_keeps(tmp_path):
    clone = _cloned(tmp_path, "C", cut="Entry$ % 2 == 0")
    assert clone.GetEntries() == 2
    with xrdroot.open_root(str(tmp_path / "clone.root")) as found:
        assert found["C"].arrays(["N.fName"])["N.fName"] == [["N_0_0"], ["N_2_0", "N_2_1", "N_2_2"]]


def test_a_clone_says_so_when_a_member_of_a_split_collection_is_not_copied(tmp_path):
    """A collection is one branch, so a member the clone does not read leaves it half written."""
    path = str(tmp_path / "clones.root")
    _written(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    upstream.SetBranchStatus("N.fName", 0)
    with pytest.raises(UnsupportedFeatureError, match="turn that branch back on"):
        upstream.CloneTree()


class _Pid:
    """The shape a macro's ``tref_test_pid`` gives the split: a name and one whole ``TRef``."""

    def __init__(self, name: str, ref: object) -> None:
        self.fName = name
        self.fRef = ROOT.TRef(ref)


def test_a_member_that_is_one_whole_object_a_row_is_that_object_and_not_a_number(monkeypatch):
    """ROOT splits ``tref_test_pid::fRef`` as a whole ``TRef`` for every object of the
    collection (``fType`` 31, ``fStreamerType`` 61), and a ``TRef`` holds no number to pack:
    its rows are the references themselves, one list per entry, which a column streams.

    The class is one the file describes rather than one of ROOT's own, so its members are
    spelled out here - ``clones_members`` reads them once the writer carries its description.
    """
    members = [
        CloneMember("TNamed", 1, "fName", 65, ""),
        CloneMember("tref_test_pid", 0, "fRef", 61, "a reference"),
    ]
    monkeypatch.setattr(objects, "clones_members", lambda element: members)
    kind = objects.SplitArray("tref_test_pid")
    named = [ROOT.TNamed(f"n{at}", "") for at in range(3)]
    entries = [
        kind.entry([_Pid("a", named[0]), _Pid("b", named[1])]),
        kind.entry([_Pid("c", named[2])]),
    ]
    keys = ["r1", "r1.fName", "r1.fRef"]
    columns = kind.columns(keys, entries)
    assert list(columns["r1"]) == [2, 1]
    assert columns["r1.fName"] == [["a", "b"], ["c"]]
    refs = columns["r1.fRef"]
    assert [(row[0].GetObject().GetName(), row[1].GetObject().GetName()) for row in refs[:1]] == [
        ("n0", "n1")
    ]
    assert [ref.GetObject().GetName() for ref in refs[1]] == ["n2"]


class _TrefPid:
    """``roottest``'s ``tref_test_pid`` as the translator declares one: a ``TNamed``
    with one whole ``TRef`` member."""

    _cxx_layout_ = ("tref_test_pid", ("TNamed",), (("fRef", "TRef", "a reference", ()),))


def test_a_split_collection_of_a_macros_class_is_carried_and_not_named_to_root():
    """A ``TClonesArray`` of a class a macro declared is split member by member off its own
    declaration, bases' members first, and the file is told what that class is: ROOT writes
    ``tref_test_pid``'s ``TStreamerInfo`` into the file, and names no such class among the
    ones it harvested a description of - there is none of ROOT's to look it up in.
    """
    layout = declared(_TrefPid)
    assert [(member.owner, member.name) for member in clones_members(layout)][-3:] == [
        ("TNamed", "fName"),
        ("TNamed", "fTitle"),
        ("tref_test_pid", "fRef"),
    ]
    split = SplitClones(layout, split=2)
    assert split.declared() == (layout,)
    assert "tref_test_pid" not in split.classes() and {"TNamed", "TObject"} <= set(split.classes())
    top = split.build("r1", 32000)
    assert top.info.clones == "tref_test_pid"
    assert [child.name for child in top.children] == [
        "r1.fUniqueID", "r1.fBits", "r1.fName", "r1.fTitle", "r1.fRef"
    ]
    assert [child.leaves()[0].ftype for child in top.children][-1] == 61


def test_a_split_collection_of_a_class_of_roots_names_it_and_carries_nothing():
    """A class ROOT has is described as :data:`~.winfo.INFOS` describes it, so the file names
    it among the classes it uses and carries no description of the writer's own."""
    split = SplitClones("TLine", split=2)
    assert split.declared() == ()
    assert "TLine" in split.classes()


class _A:
    """``roottest``'s ``io/references`` ``A``: a ``TObject`` with one whole ``TRef`` member."""

    _cxx_layout_ = ("A", ("TObject",), (("_o", "TRef", "", ()),))


def _declared_in_cxx(monkeypatch, cls: Any) -> None:
    """``cls`` as the interpreter holds a class a macro declared, which is what ``find_class``
    searches - the translation registers every class it makes there."""
    from xrdroot.pycxx.runtime.namespace import DECLARED

    monkeypatch.setitem(DECLARED, str(cls._cxx_layout_[0]), cls)


def test_a_clones_array_of_a_macros_class_is_split_by_the_declaration_it_carries(monkeypatch):
    """``roottest``'s ``io/references/lotsRef`` asks for a ``TClonesArray`` of a macro's ``A``
    at split level 3, and the array names its class as a string no description answers for:
    the class is one the translation declared, so it is split member by member off that
    declaration - the ``TObject`` part first, then the reference - and not refused.
    """
    _declared_in_cxx(monkeypatch, _A)
    held = ROOT.TClonesArray("A", 1)
    kind = objects._kind(held, 3)
    assert isinstance(kind, objects.SplitArray)
    assert [member.name for member in kind.members] == ["fUniqueID", "fBits", "_o"]
    assert kind.members[-1].typename == "TRef"  # the leaf names it, the branch does not
    spec = kind.spec(held, 3, 32000)
    assert spec.declared() == (declared(_A),)
    assert spec.build("C1", 32000).info.clones == "A"


def test_a_clones_array_of_a_class_root_has_is_still_split_by_name(monkeypatch):
    """A class ROOT describes itself is split by that description, whether or not a macro
    declared one of the same name: ``INFOS`` is what the file's classes are read from."""
    _declared_in_cxx(monkeypatch, _A)
    held = ROOT.TClonesArray("TNamed")
    kind = objects._kind(held, 3)
    assert isinstance(kind, objects.SplitArray) and kind.element == "TNamed"


def test_a_split_array_writes_its_numbers_and_strings_as_the_entries_hold_them(monkeypatch):
    """Every kind whose rows are numbers or strings hands its columns to the writer as they
    are; only a row holding another class's object needs anything done to it."""
    kind = objects.Macro()
    columns = {"a": 1}
    assert kind.written(["a"], columns) is columns


def test_a_member_that_is_one_whole_ref_is_written_as_that_refs_record():
    """ROOT writes a whole ``TRef`` member of a split ``TClonesArray`` as the ``TRef``'s own
    record, so the member's rows are its bytes rather than the objects, the whole entry's
    references one after another. Checked against the 6.40 oracle, which writes a heap
    ``TRef`` referring to object 1 of ``ProcessID0`` as ``0001 00000001 03000000 0000``:
    ``TObject``'s version, the number it refers to, its bits, and which process record
    numbered that object - as many records as the entry's count says.
    """
    split = objects.SplitArray(declared(_A))
    keys = ["C1", "C1.fUniqueID", "C1.fBits", "C1._o"]
    ref = ROOT.TRef(ROOT.TNamed("n", "t"))
    number = ref.GetUniqueID()
    columns = split.columns(keys, [((0, 0, ref), (0, 0, ref))])
    record = struct.pack(">HIIH", 1, number, 0x03000000, ref._pid.GetUniqueID())
    assert record == bytes.fromhex(f"0001{number:08x}030000000000")
    assert split.written(keys, columns)["C1._o"] == [record * 2]
    # A reference that points at nothing names no process: ROOT writes ``fPID`` as 0.
    assert tref_record(ROOT.TRef()) == bytes.fromhex("0001" "00000000" "03000000" "0000")


def test_a_whole_member_of_another_class_is_refused_by_the_class_it_holds():
    """Only a ``TRef`` has a streamer here, so a split collection's member that is a whole
    object of any other class is refused by that class's name rather than written wrong."""
    member = CloneMember("A", 2, "_o", 61, "", typename="TGraph")
    with pytest.raises(UnsupportedFeatureError, match=r"A holds _o, a whole TGraph"):
        objects._objects(member, [])


def test_a_whole_object_members_leaf_is_typed_by_that_members_class():
    """``TLeafElement::GetTypeName`` of ``C1._o`` is ``TRef``, as ROOT prints it: the branch
    names the class that owns the member - ``A`` - so the member's own class travels on the
    leaf, where nothing of the record could hold it."""
    split = SplitClones(declared(_A), split=2)
    holder = type("Slot", (), {"name": "C1", "spec": lambda self: split})()
    info = object_info(holder, None, 1, False)
    leaves = {child.name: child.leaves[0].typename for child in info.children}
    assert leaves["C1._o"] == "TRef" and leaves["C1.fBits"] == "UInt_t"


def test_a_clones_branch_names_the_class_it_holds_and_a_vector_branch_names_none(tmp_path):
    """``TBranchElement::GetClonesName``: what a macro reads to make the ``TClonesArray`` it
    reads the branch into - ``roottest``'s ``io/tree/tref_test_pid.C`` does. ROOT keeps
    ``fClonesName`` for a ``TClonesArray`` alone, leaving it empty for a collection of a
    ``std::vector``; checked against the 6.40 oracle.
    """
    path = str(tmp_path / "clones.root")
    _written(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    assert tree.GetBranch("N").GetClonesName() == "TNamed"
    back.Close()
    out = ROOT.TFile(str(tmp_path / "plain.root"), "RECREATE")
    plain = ROOT.TTree("p", "plain")
    numbers = ROOT.std.vector["int"]()
    plain.Branch("v", numbers)
    plain.Fill()
    plain.Write()
    out.Close()
    back = ROOT.TFile(str(tmp_path / "plain.root"))
    read = back.Get("p")
    assert (read.GetBranch("v").GetClonesName(), read.GetBranch("v").GetClassName()) == (
        "",
        "vector<int>",
    )
    back.Close()


#: ``roottest``'s ``io/tree/tref_test_pid.C``'s class: a ``TNamed`` with a whole ``TRef``.
TREF_PID = r"""
class tref_test_pid: public TNamed {
public:
   tref_test_pid() {}
   TRef fRef;
   ClassDef(tref_test_pid, 1)
};
"""


def _written_pid(path: str) -> None:
    """``tref_test_pid``'s tree: two entries of three named objects, the ``TClonesArray``
    split member by member as ROOT splits one."""
    run_source(TREF_PID, "tref_test_pid.C", call=False)
    made_class = find_class("tref_test_pid")
    assert made_class is not None
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    held = ROOT.TClonesArray("tref_test_pid")
    tree.Branch("r1", held)
    for entry in range(2):
        held.Clear()
        for at in range(3):
            made = made_class()
            made.SetName(f"r1_N_{entry}_{at}")
            made.SetTitle(f"title {at}")
            held.AddAt(made, at)
        tree.Fill()
    tree.Write()
    out.Close()


def test_a_split_collection_read_into_a_clones_array_fills_every_elements_members(
    tmp_path,
):
    """``roottest``'s ``io/tree/tref_test_pid.C`` hands ``SetAddress`` a ``TClonesArray`` and
    reads each element's members off the split branches - the ``TNamed`` part among them,
    which the declaration names as a base and no branch names at all. Every element is an
    object of the branch's class with the name and title read, not an empty one.
    """
    path = str(tmp_path / "pid.root")
    _written_pid(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    held = ROOT.TClonesArray(tree.GetBranch("r1").GetClonesName())
    tree.SetBranchAddress("r1", held)
    tree.GetEntry(1)
    assert [o.GetName() for o in held] == ["r1_N_1_0", "r1_N_1_1", "r1_N_1_2"]
    assert [o.GetTitle() for o in held] == ["title 0", "title 1", "title 2"]
    assert all(o.GetName() != "" for o in held)
    back.Close()


def test_a_split_collection_read_into_a_pointer_is_the_clones_array_it_points_at(tmp_path):
    """``b->SetAddress(&caN)`` points at an array the tree fills: the cell keeps the array
    made for it, and each element of it is an object of the branch's class again."""
    path = str(tmp_path / "pid.root")
    _written_pid(path)
    back = ROOT.TFile(path)
    tree = back.Get("T")
    cell = type("Cell", (), {"value": ROOT.TClonesArray("tref_test_pid")})()
    tree.SetBranchAddress("r1", cell)
    tree.GetEntry(0)
    assert [o.GetName() for o in cell.value] == ["r1_N_0_0", "r1_N_0_1", "r1_N_0_2"]
    back.Close()


class _Base:
    """A class a macro declared, which another class of the macro's derives from."""

    _cxx_layout_ = ("_Base", (), (("fX", "int", "the x", ()),))


class _Derived:
    """The derived class: one member of its own, and its base's."""

    _cxx_layout_ = ("_Derived", ("_Base",), (("fY", "int", "the y", ()),))


def test_a_base_a_macro_declared_takes_its_members_under_its_own_name(monkeypatch):
    """A split branch names ``fX`` and ``fY`` side by side, and a class's reader takes only
    the members it declares itself: the row nests a declared base's members under the base's
    name, and leaves out a member of the file that nothing declares.
    """
    from xrdroot.pyroot.trees.rebuild import _by_base

    _declared_in_cxx(monkeypatch, _Base)
    _declared_in_cxx(monkeypatch, _Derived)
    assert _by_base(_Derived, {"fX": 1, "fY": 2, "fLost": 3}) == {
        "fY": 2,
        "_Base": {"fX": 1},
    }


def test_a_class_nobody_declared_is_made_by_the_part_of_the_kit_that_knows_it():
    """A collection of one of ROOT's own classes - ``TNamed`` - is made by the factory
    registered for it, and one of neither is refused by the namespace rather than made
    empty, so a session that never declared the class does not read silent blanks.
    """
    from xrdroot.pyroot.trees.rebuild import _member

    assert _member("TNamed", {"fName": "n", "fTitle": ""}) is None
    assert _member("Nowhere", {"fName": "n"}) is None
    assert _member("TNamed", ["n"]) is None  # not a row of members at all


def _written_refs(path: str) -> None:
    """A tree whose objects hold a whole ``TRef`` apiece, referring to a named object of a
    collection beside them: ``roottest``'s ``create-r`` stage, which writes the file it reads."""
    run_source(TREF_PID, "tref_test_pid.C", call=False)
    made_class = find_class("tref_test_pid")
    counted = ROOT.TProcessID.GetObjectCount()  # the numbers made here are this file's alone
    out = ROOT.TFile(path, "RECREATE")
    tree = ROOT.TTree("T", "T")
    names = ROOT.TClonesArray("TNamed")
    tree.Branch("N", names)
    refs = ROOT.TClonesArray("tref_test_pid")
    tree.Branch("r1", refs)
    for at in range(2):
        names.Clear()
        picked = ROOT.TNamed(f"N_{at}", f"N_{at}")
        names.AddAt(picked, 0)
        refs.Clear()
        made = made_class()
        made.SetName(f"r1_N_{at}")
        made.fRef = ROOT.TRef(picked)
        refs.AddAt(made, 0)
        tree.Fill()
    tree.Write()
    out.Close()
    ROOT.TProcessID.SetObjectCount(counted)


def test_a_clone_of_a_reference_carries_the_process_record_its_file_knew(tmp_path):
    """A clone copies a ``TRef``'s record as its source file wrote it, so its ``fPID`` names an
    index among *that* file's ``ProcessID`` records - which the file the clone goes into has
    never seen. ROOT writes those records into the clone as it copies the entries, and a
    reader of the clone finds both the object a reference names and the process to look it
    up in; without them a reference reads back as nothing at all
    (roottest/root/io/tree/test-rc).
    """
    path = str(tmp_path / "refs.root")
    _written_refs(path)
    with xrdroot.open_root(path) as before:
        recorded = before["ProcessID0"]["TNamed"]["fTitle"]
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CloneTree()
    made = str(tmp_path / "clone.root")
    out = ROOT.TFile(made, "RECREATE")  # the source file stays open while the clone is written
    clone.SetDirectory(out)
    clone.Write()
    out.Close()
    source.Close()
    with xrdroot.open_root(made) as after:
        assert after.keys() == ["ProcessID0", "T"]
        assert after["ProcessID0"]["TNamed"]["fTitle"] == recorded
    back = ROOT.TFile(made)
    read = back.Get("T")
    held = ROOT.TClonesArray(read.GetBranch("r1").GetClonesName())
    read.SetBranchAddress("r1", held)
    read.GetEntry(0)
    assert held[0].fRef.GetObject().GetName() == "N_0"
    back.Close()


def test_a_clone_with_no_file_to_say_which_process_keeps_the_index_it_read():
    """A tree read of no file at all - one a clone copied from memory - has no ``ProcessID``
    record to carry, so a reference keeps the index its own record has. Every other index
    written here would name a process nothing could look up.
    """
    member = CloneMember("tref_test_pid", 0, "fRef", 61, "a reference")
    described = {"TObject": {"fUniqueID": 3, "fBits": 0}, "fPID": 0}
    columns = {"r1.fRef": [[described]]}
    objects._carried_processes(["r1.fRef"], [member], columns, None)
    # nor has a file nothing is being written to: the index is left as it was read, whatever
    # process the source files named
    objects._carried_processes(
        ["r1.fRef"], [member], columns, {"ProcessID0": {"TNamed": {"fTitle": "a-uuid"}}}
    )
    assert columns["r1.fRef"] == [[described]]
    assert objects._source_uuid({"ProcessID1": {}}, 0) is None  # a file that recorded no process
    assert objects._source_uuid({"ProcessID0": {"TNamed": {"fTitle": "a-uuid"}}}, 0) == "a-uuid"


def _cloned_live_refs(tmp_path) -> tuple[str, list[int]]:
    """``roottest``'s ``create-rcr``: a clone of a collection whose objects are numbered *after*
    they were read, and what each of them was numbered as.

    ``addref`` binds the source's ``r1`` branch to a ``TClonesArray`` and makes a ``TRef`` from
    every object it read, which is ``TProcessID::AssignID`` numbering them in the session - so
    the numbers are not the ones the entry holds, and the clone has to copy the objects.
    """
    path = str(tmp_path / "refs.root")
    _written_refs(path)
    # the file was written by a process that is not this session's, as ``create-rc``'s was: a
    # session reading it files its ``ProcessID0`` beside its own, at index 1
    ROOT.TProcessID._pids[0].SetTitle("another session")
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CloneTree(0)
    out = ROOT.TFile(str(tmp_path / "clone.root"), "RECREATE")
    clone.SetDirectory(out)
    held = ROOT.TClonesArray(upstream.GetBranch("r1").GetClonesName())
    upstream.SetBranchAddress("r1", held)
    numbers: list[int] = []
    counted = ROOT.TProcessID.GetObjectCount()
    for entry in range(upstream.GetEntries()):
        upstream.GetEntry(entry)
        for at in range(len(held)):
            ROOT.TRef(held[at])  # what a ``tref_test_pid`` constructor does to its object
            numbers.append(held[at].GetUniqueID())
        clone.Fill()
        ROOT.TProcessID.SetObjectCount(counted)
    clone.Write()
    out.Close()
    source.Close()
    return str(tmp_path / "clone.root"), numbers


def test_a_clone_writes_the_identifiers_its_objects_got_after_they_were_read(tmp_path):
    """An identifier is not a value ``GetEntry`` settles: ``AssignID`` numbers an object when a
    ``TRef`` is made to it, and nothing writes that number back into the entry. A clone of a
    collection the tree was handed copies the objects, so its ``fUniqueID`` carries the number
    and its ``fBits`` says a reference may point at the object; the byte above the number is
    the index of the ``ProcessID`` record holding it, plus one - the session's own process is
    the second record here, because the source file's has to keep the index it was read with
    (roottest/root/io/tree/test-rcr).
    """
    made, numbers = _cloned_live_refs(tmp_path)
    with xrdroot.open_root(made) as found:
        written = found["T"].arrays(["r1.fUniqueID", "r1.fBits"])
    assert written["r1.fUniqueID"].tolist() == [
        [(2 << 24) | (number & 0xFFFFFF)] for number in numbers
    ]
    assert written["r1.fBits"].tolist() == [[0x03000018] for _ in numbers]


def test_a_collection_the_tree_holds_no_objects_of_reads_its_identifiers_from_the_entry(
    tmp_path,
):
    """Nothing bound to the branch: the entry is all there is to copy, and its identifiers as
    well as its names are the clone's."""
    path = str(tmp_path / "refs.root")
    _written_refs(path)
    source = ROOT.TFile(path)
    upstream = source.Get("T")
    clone = upstream.CloneTree()
    with xrdroot.create(str(tmp_path / "clone.root")) as out:  # the source's baskets are read
        clone.SetDirectory(out)
        clone.Write()
    source.Close()
    with xrdroot.open_root(str(tmp_path / "clone.root")) as found:
        copied = found["T"].arrays(["r1.fUniqueID", "r1.fBits", "r1.fName"])
    assert copied["r1.fUniqueID"].tolist() == [[0], [0]]
    assert copied["r1.fBits"].tolist() == [[0x03000000], [0x03000000]]
    assert copied["r1.fName"] == [["r1_N_0"], ["r1_N_1"]]


def test_an_object_nothing_numbered_keeps_the_identifier_it_has():
    """An object no process numbered has no record to name, so its identifier is written as it
    stands - a number with no byte above it, which is what makes the byte mean "a process"."""
    plain = ROOT.TNamed("n", "n")
    plain.SetUniqueID(7)
    assert copying._identifier(plain) == 7


def test_an_identifier_numbered_with_no_file_to_write_keeps_the_number_it_has():
    """Nothing is being written, so there is no record to number the object into: the number
    stays the object's own rather than naming a process the file would not hold."""
    numbered = ROOT.TNamed("n", "n")
    ROOT.TProcessID.AssignID(numbered)
    assert copying._identifier(numbered) == numbered.GetUniqueID()
    copying._carried_records({"ProcessID0": {"TNamed": {"fTitle": "a-uuid"}}})


class _HeldNumber:
    """An address a number was bound to, which is no collection to read objects from."""

    def get(self, count: int | None = None) -> int:
        return 0


class _Bound:
    """A tree holding one address, as ``SetBranchAddress`` leaves it."""

    _addresses: ClassVar[dict[str, Any]]

    def __init__(self) -> None:
        self._addresses = {"r1": _HeldNumber()}


def test_an_address_holding_a_number_is_no_collection_of_objects():
    assert copying._live_objects(_Bound(), "r1") is None
    assert copying._live_objects(_Bound(), "nothing bound") is None


def test_a_source_file_that_recorded_no_process_gives_the_clone_none_to_carry(tmp_path):
    """Nothing to carry means nothing written: a file a clone goes into has a ``ProcessID``
    record for the processes its references name and no others."""
    path = str(tmp_path / "none.root")
    out = ROOT.TFile(path, "RECREATE")
    copying._carried_records({})  # a source file of no reference at all
    out.Close()
    with xrdroot.open_root(path) as found:
        assert [key for key in found.keys() if key.startswith("ProcessID")] == []
