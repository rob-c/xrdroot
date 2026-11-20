"""``TTree::MakeSelector``'s skeleton, ``TSelector`` and ``TTree::Process`` running it.

ROOT writes a class deriving from ``TSelector`` whose ``Process`` reads the tree
through a ``TTreeReader``, and ``TTree::Process`` builds that class from a macro
and calls it entry by entry. The skeleton is C++ here as it is in ROOT, so a
macro that fills in ``Process`` - ``#define Tree_cxx``, include the header,
define the methods out of line - runs unchanged; what these tests check is the
readers the option asks for, the macro being run over the entries, and the
names ROOT refuses.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.errors import UnsupportedFeatureError
from xrdroot.pycint.runtime.root import DECLARED as declared
from xrdroot.pyroot.trees import reader as made_reader
from xrdroot.pyroot.trees import selector as made
from xrdroot.pyroot.trees.layout import BranchInfo, LeafInfo, element_class, held_typename
from xrdroot.streamers import Member

#: A leaf of numbers, as the tree's layout reports one.
FLOAT = LeafInfo("x", "x", "x/F", "Float_t", "TLeafF", 1, None, "x", False, False)


class Fake:
    """As much of a tree as ``MakeSelector`` reads: its name, its leaves and an entry."""

    def __init__(self, leaves, branches, values, backing=None):
        self._given = leaves
        self._branches = branches
        self._values = values
        self._read_entry = -1
        self._reader_buffers: dict[str, object] = {}
        #: What the file says about the classes it holds, as a tree read from one has it.
        self._backing: object = (lambda: backing) if backing is not None else None

    def GetName(self):
        return "fake"

    def GetTitle(self):
        return "a tree that never was"

    def GetEntries(self):
        return 0

    def _layout(self):
        return list(self._branches)

    def _every_branch(self):
        return [branch for top in self._branches for branch in top.walk()]

    def _leaves(self):
        return list(self._given)

    def _leaf_info(self, name):
        return next((leaf for leaf in self._given if leaf.column == name
                     or leaf.name == name), None)  # fmt: skip

    def _branch_info(self, name):
        return next((branch for branch in self._every_branch() if branch.name == name), None)

    def _current(self, column):
        if column not in self._values:
            raise KeyError(column)
        return self._values[column]


def _branch(name, leaves, classname="", clones="", children=()):
    return BranchInfo(name, name, classname, list(leaves), 1, 0, 0, 0, 32000,
                      children=tuple(children), clones=clones)  # fmt: skip


def _read_from(streamers, names):
    """A tree read from a file: the branches it has, and what the file says of each class."""

    class Source:
        def streamers(self):
            return streamers

    class Held:
        _source = Source()

    class Backing:
        branches: ClassVar[dict[str, object]] = {name: Held() for name in names}

    return Backing()


def test_a_skeleton_is_written_with_a_reader_per_leaf_of_the_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tree = ROOT.TTree("Tree", "Tree with basic leaf variables")
    for name, kind, leaflist in (("px", np.float32, "px/F"), ("random", np.float64, "random/D"),
                                 ("ev", np.int32, "ev/I")):  # fmt: skip
        tree.Branch(name, np.zeros(1, dtype=kind), leaflist)
    assert tree.MakeSelector() == 1
    header = (tmp_path / "Tree.h").read_text()
    assert "TTreeReaderValue<Float_t> px = {fReader, \"px\"};" in header
    assert "TTreeReaderValue<Int_t> ev = {fReader, \"ev\"};" in header
    assert "class Tree : public TSelector {" in header
    assert (tmp_path / "Tree.C").read_text().startswith("// The class definition")


def test_the_readers_an_option_asks_for_are_the_ones_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outer = LeafInfo("ClassB_branch", "ClassB_branch", "", "ClassB", "TLeafElement",
                     1, None, "ClassB_branch", False, False)  # fmt: skip
    inner = LeafInfo("fC.fPx", "fC.fPx", "", "Float_t", "TLeafElement", 1, None,
                     "fC", False, False)  # fmt: skip
    base = LeafInfo("fBuckets", "fBuckets", "", "Int_t", "TLeafI", 1, None, "fC", False, False)
    sub = _branch("fC", [inner, base])
    top = _branch("ClassB_branch", [outer], "ClassB", children=[sub])
    tree = Fake([outer, inner, base], [top], {
        "ClassB_branch": {"fPx": 1.0},
        "fC.fPx": np.float32(2.0),
        "fBuckets": np.int32(3),
    })
    assert made._declarations(tree, "@") == [
        '   TTreeReaderValue<ClassB> ClassB_branch = {fReader, "ClassB_branch"};'
    ]
    # ``@fC`` asks for the branch whole, which reads every leaf it has; the leaves of a
    # branch are those whose path is under it.
    assert len(made._declarations(tree, "@fC")) == 2
    assert made._declarations(tree, "fC") == [
        '   TTreeReaderValue<Float_t> fC_fPx = {fReader, "fC.fPx"};'
    ]
    assert made._declarations(tree, "")[0].startswith("   TTreeReader")


def test_a_branch_the_tree_does_not_have_is_root_s_warning(capsys):
    tree = Fake([FLOAT], [_branch("x", [FLOAT])], {"x": np.float32(1)})
    found = made._chosen(tree, "@noSuchBranch;@x")
    assert [leaf.column for leaf in found] == ["x"]
    assert made._chosen(tree, "noSuchLeaf") == []  # a warning, and no reader
    assert "Warning in <ParseOptions>: Tree fake does not contain a branch named noSuchBranch." in (
        capsys.readouterr().err
    )


def test_a_reader_is_an_array_for_a_fixed_size_array_and_a_value_for_a_number():
    many = LeafInfo("arr", "arr", "arr[10]/F", "Float_t", "TLeafF", 10, None, "arr", False, False)
    tree = Fake([many, FLOAT], [_branch("arr", [many]), _branch("x", [FLOAT])], {
        "arr": np.zeros(10, dtype=np.float32),
        "x": np.float32(1.0),
    })
    assert made._reader(tree, many) == ("Array", "Float_t")
    assert made._reader(tree, FLOAT) == ("Value", "Float_t")


def test_a_vector_of_bools_is_a_value_and_a_vector_of_numbers_an_array():
    flag = LeafInfo("vb", "vb", "", "vector<bool>", "TLeafElement", 1, None, "vb", False, True)
    nums = LeafInfo("vpx", "vpx", "", "vector<float>", "TLeafElement", 1, None,
                    "vpx", False, True)  # fmt: skip
    tree = Fake([flag, nums], [_branch("vb", [flag]), _branch("vpx", [nums])], {
        "vb": np.array([True]), "vpx": np.array([1.0], dtype=np.float32),
    })
    assert made._reader(tree, flag) == ("Value", "vector<bool>")
    assert made._reader(tree, nums) == ("Array", "float")


def test_a_branch_of_objects_reads_as_a_value_and_a_collection_as_an_array():
    one = LeafInfo("r", "r", "", "Particle", "TLeafElement", 1, None, "r", False, False)
    many = LeafInfo("arr_", "arr", "", "TClonesArray", "TLeafElement", 1, None,
                    "arr", False, False)  # fmt: skip
    tree = Fake([one, many], [_branch("r", [one], "Particle"),
                              _branch("arr", [many], "TClonesArray", clones="Particle")],
                {"r": {"fPosX": 1.0}, "arr": {"fPosX": np.array([1.0])}})  # fmt: skip
    assert made._reader(tree, one) == ("Value", "Particle")
    assert made._reader(tree, many) == ("Array", "Particle")


def test_a_branch_of_a_collection_is_an_array_and_one_of_a_class_a_value():
    assert made._many(None) is False  # a leaf with no branch of its own
    assert made._many(_branch("one", [FLOAT], "Particle")) is False
    assert made._many(_branch("list", [FLOAT], "TClonesArray")) is True
    assert made._many(_branch("objs", [FLOAT], "TObjArray")) is True
    assert made._many(_branch("vec", [FLOAT], "vector<Particle>")) is True
    assert made._many(_branch("clones", [FLOAT], "", clones="Particle")) is True


def test_a_class_one_of_whose_members_is_a_collection_reads_as_one_object():
    leaf = LeafInfo("ClassWithVector_branch", "ClassWithVector_branch", "", "ClassWithVector",
                    "TLeafElement", 1, None, "ClassWithVector_branch", False, False)  # fmt: skip
    held = {"vec": np.zeros(5), "vecBool": np.zeros(5)}
    tree = Fake([leaf], [_branch("ClassWithVector_branch", [leaf], "ClassWithVector")],
                {"ClassWithVector_branch": held})
    assert made._reader(tree, leaf) == ("Value", "ClassWithVector")


def test_a_reader_of_a_member_branch_is_typed_by_the_member_its_class_declares():
    outer = LeafInfo("ClassB_branch", "ClassB_branch", "", "ClassB", "TLeafElement",
                     1, None, "ClassB_branch", False, False)  # fmt: skip
    member = LeafInfo("fC", "fC", "", "ClassB", "TLeafElement", 1, None,
                      "fC", False, False)  # fmt: skip
    top = _branch("ClassB_branch", [outer], "ClassB", children=[_branch("fC", [member], "ClassB")])
    values = {"ClassB_branch": {"fPy": 1.0}, "fC": {"fPx": 2.0, "fEv": 3}}
    described = {"ClassB": {"fC": Member("fC", "", 61, "ClassC", 1)}}
    tree = Fake([outer, member], [top], values, backing=_read_from(described, ["fC"]))
    assert made._reader(tree, member) == ("Value", "ClassC")
    assert made._declarations(tree, "@fC") == [
        '   TTreeReaderValue<ClassC> fC = {fReader, "fC"};'
    ]
    # the file's description of another class says nothing about this branch's member
    other = Fake([outer, member], [top], values, backing=_read_from({}, ["fC"]))
    assert made._reader(other, member) == ("Value", "ClassB")
    # a tree being filled keeps no description of its classes, and a friend no branches
    nothing = Fake([outer, member], [top], values, backing=object())
    assert made._reader(nothing, member) == ("Value", "ClassB")
    unfound = Fake([outer, member], [top], values, backing=_read_from({}, ["elsewhere"]))
    assert made._reader(unfound, member) == ("Value", "ClassB")


def test_an_unreadable_member_is_typed_by_the_collection_the_file_says_it_holds():
    """``EventData::fParticles`` in a tree written one split level deep: the branch is the
    member whole, one ``vector<Particle>`` block of the entry, so the entry cannot be read
    as columns and the file's account of the member is what the reader is declared from -
    ROOT writes ``TTreeReaderArray<Particle> fParticles``, whose ``fParticles[i].fPosX``
    then reads an element. A member holding no collection stays what the leaf was typed as.
    """
    parts = LeafInfo("fParticles", "fParticles", "", "EventData", "TLeafElement",
                     1, None, "fParticles", False, False)  # fmt: skip
    flags = LeafInfo("fFlags", "fFlags", "", "EventData", "TLeafElement",
                     1, None, "fFlags", False, False)  # fmt: skip
    other = LeafInfo("fOther", "fOther", "", "EventData", "TLeafElement",
                     1, None, "fOther", False, False)  # fmt: skip
    branches = [_branch(name, [leaf], "EventData")
                for name, leaf in (("fParticles", parts), ("fFlags", flags),
                                   ("fOther", other))]  # fmt: skip
    described = {"EventData": {"fParticles": Member("fParticles", "", 62, "vector<Particle>", 1),
                               "fFlags": Member("fFlags", "", 61, "vector<bool>", 1),
                               "fOther": Member("fOther", "", 61, "Particle", 1)}}  # fmt: skip
    tree = Fake([parts, flags, other], branches, {},
                backing=_read_from(described, ["fParticles", "fFlags", "fOther"]))  # fmt: skip
    assert made._reader(tree, parts) == ("Array", "Particle")
    assert made._reader(tree, flags) == ("Value", "vector<bool>")
    # a member that is no collection is still the number a nameless leaf reads as
    assert made._reader(tree, other) == ("Value", "Double_t")
    assert made._declarations(tree, "@fParticles") == [
        '   TTreeReaderArray<Particle> fParticles = {fReader, "fParticles"};'
    ]


def test_a_leaf_whose_type_root_did_not_name_is_typed_by_what_it_held():
    odd = LeafInfo("odd", "odd", "", "SomeClass", "TLeafElement", 1, None, "odd", False, False)
    unknown = LeafInfo("huge", "huge", "", "SomeClass", "TLeafElement", 1, None,
                       "huge", False, False)  # fmt: skip
    tree = Fake([odd, unknown], [_branch("odd", [odd]), _branch("huge", [unknown])], {
        "odd": np.float64(1.0), "huge": np.array([1j]),
    })
    assert made._numeric(odd, np.float64(1.0)) == "Double_t"
    assert made._numeric(unknown, np.array([1j])) == "Double_t"
    assert made._element(tree, LeafInfo("v", "v", "", "vector<unsigned int>", "TLeafElement",
                                        1, None, "v", False, True)) == "unsigned int"  # fmt: skip
    assert made._element(Fake([], [], {"w": np.array([True])}),
                         LeafInfo("w", "w", "", "std::set<bool>", "TLeafElement",
                                  1, None, "w", False, True)) == "bool"  # fmt: skip
    assert made._element(Fake([], [], {}), LeafInfo("v", "v", "", "std::set<int>",
                                                    "TLeafElement", 1, None, "v", False,
                                                    True)) == "int"  # fmt: skip


def test_the_class_of_a_branch_of_objects_comes_from_what_its_record_says():
    leaf = LeafInfo("vp.fPosX", "vp.fPosX", "", "Particle", "TLeafElement", 1, None,
                    "vp", False, False)  # fmt: skip
    assert element_class(None, leaf) == "Particle"  # a leaf with no branch of its own
    assert element_class(_branch("vp", [leaf], "vector<Particle>"), leaf) == "Particle"
    assert element_class(_branch("vp", [leaf], "TClonesArray"), leaf) == "Particle"
    assert element_class(_branch("vp", [leaf], "TObjArray"), leaf) == "Particle"
    assert element_class(_branch("vp", [leaf], "Particle"), leaf) == "Particle"


def test_the_type_a_split_member_is_held_as_comes_from_the_files_streamers():
    leaf = LeafInfo("fC.fPx", "fC.fPx", "", "Float_t", "TLeafElement", 1, None,
                    "fC", False, False)  # fmt: skip
    branch = _branch("fC", [leaf], "ClassB")
    found = _read_from({"ClassB": {"fPx": Member("fPx", "", 18, "ClassC", 0)}}, ["fC"])
    assert held_typename(Fake([leaf], [branch], {}, found), branch, leaf) == "ClassC"


def test_a_member_the_streamer_information_does_not_describe_is_typed_by_its_branch():
    leaf = LeafInfo("fC.fPx", "fC.fPx", "", "Float_t", "TLeafElement", 1, None,
                    "fC", False, False)  # fmt: skip
    branch = _branch("fC", [leaf], "ClassB")
    other = {"ClassB": {"fOther": Member("fOther", "", 18, "ClassC", 0)}}
    unnamed = Fake([leaf], [branch], {}, _read_from(other, ["fC"]))
    assert held_typename(unnamed, branch, leaf) is None  # the class, but not this member
    undescribed = Fake([leaf], [branch], {}, _read_from({}, ["fC"]))
    assert held_typename(undescribed, branch, leaf) is None  # nor the class
    assert held_typename(undescribed, None, leaf) is None  # a leaf with no branch


def test_a_tree_not_read_from_a_file_holds_no_streamer_information():
    leaf = LeafInfo("fPx", "fPx", "", "Float_t", "TLeafF", 1, None, "fPx", False, False)
    branch = _branch("fPx", [leaf], "ClassB")
    assert held_typename(Fake([leaf], [branch], {}), branch, leaf) is None
    backing = _read_from({}, [])
    assert held_typename(Fake([leaf], [branch], {}, backing), branch, leaf) is None


def test_a_tree_with_no_interpreter_installed_refuses_to_run_a_selector(tmp_path, monkeypatch):
    import xrdroot.plugins as plugins

    macro = tmp_path / "S.C"
    macro.write_text("void S() { }\n")
    monkeypatch.setattr(plugins, "lookup", lambda point, key="default": None)
    with pytest.raises(UnsupportedFeatureError, match="none is installed here"):
        ROOT.TTree("t", "t").Process(str(macro))


def test_a_column_that_cannot_be_read_falls_back_on_its_leaf_type():
    tree = Fake([FLOAT], [_branch("x", [FLOAT])], {})
    assert made._observed(tree, "x") is None
    assert made._reader(tree, FLOAT) == ("Value", "Float_t")


def test_a_name_two_leaves_share_is_declared_once():
    first = LeafInfo("fPx", "C1.fPx", "", "Float_t", "TLeafElement", 1, None, "C1", False, False)
    again = LeafInfo("fPx", "C2.fPx", "", "Float_t", "TLeafElement", 1, None, "C2", False, False)
    tree = Fake([first, again], [_branch("C1", [first]), _branch("C2", [again])],
                {"C1.fPx": np.float32(1.0), "C2.fPx": np.float32(2.0)})  # fmt: skip
    declared = made._declarations(tree, "")
    assert [line for line in declared if "fPx" in line] == [
        '   TTreeReaderValue<Float_t> fPx = {fReader, "C1.fPx"};'
    ]


def test_a_macro_that_declares_no_selector_is_refused_by_name(tmp_path):
    macro = tmp_path / "Nope.C"
    macro.write_text("void Nope() { }\n")
    tree = ROOT.TTree("t", "t")
    with pytest.raises(UnsupportedFeatureError, match="declares no class Nope deriving"):
        tree.Process(str(macro))


def test_an_include_is_spelled_out_only_when_only_the_run_directory_has_it(tmp_path, monkeypatch):
    beside = tmp_path / "sub" / "Header.h"
    beside.parent.mkdir()
    beside.write_text("// nothing\n")
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen" / "Made.h").write_text("// nothing\n")
    macro = tmp_path / "sub" / "M.C"
    text = '#include "Header.h"\n#include "gen/Made.h"\n#include "/absolute/Nowhere.h"\n'
    monkeypatch.chdir(tmp_path)
    resolved = made._resolved(text, macro)
    assert '"Header.h"' in resolved  # beside the macro, where the translator looks first
    assert f'"{Path(tmp_path, "gen", "Made.h").resolve()}"' in resolved
    assert '"/absolute/Nowhere.h"' in resolved
    monkeypatch.chdir(beside.parent.parent.parent)
    assert made._resolved(text, macro) == text  # a file nowhere found stays as it was


def test_a_selector_runs_the_entries_it_is_given(tmp_path, capsys):
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(tmp_path)
    try:
        (tmp_path / "generated_selectors").mkdir()
        tree = ROOT.TTree("Tree", "a tree")
        px = np.zeros(1, dtype=np.float32)
        tree.Branch("px", px, "px/F")
        for i in range(4):
            px[0] = 100.0 + i
            tree.Fill()
        tree.MakeSelector()
        (tmp_path / "generated_selectors" / "Tree.h").write_text((tmp_path / "Tree.h").read_text())
        macro = tmp_path / "test_selectors" / "Tree.C"
        macro.parent.mkdir()
        macro.write_text(
            '#define Tree_cxx\n#include "generated_selectors/Tree.h"\n'
            "bool Tree::Process(Long64_t entry)\n{\n   fReader.SetEntry(entry);\n"
            '   fprintf(stderr, "%.1f\\n", *px);\n   return true;\n}\n'
        )
        assert tree.Process(str(macro), "", 2, 1) == 2
        assert capsys.readouterr().err.split() == ["101.0", "102.0"]
    finally:
        monkeypatch.undo()


def test_the_base_selector_answers_as_root_s_does():
    selector = made.TSelector()
    assert selector.Version() == 2
    selector.SetOption("x")
    assert selector.GetOption() == "x"
    obj, inputs = object(), [1]
    selector.SetObject(obj)
    selector.SetInputList(inputs)
    assert selector.fObject is obj and selector.fInput == inputs
    assert selector.GetOutputList() is None
    assert selector.Notify() and selector.Process(0)
    for call in (selector.Init, selector.Begin, selector.SlaveBegin):
        assert call(None) is None
    assert selector.SlaveTerminate() is None and selector.Terminate() is None


def test_a_reader_of_a_tree_it_does_not_have_yet_resolves_at_the_first_read(tmp_path):
    yield_from = fresh(tmp_path)
    next(yield_from)
    try:
        tree = ROOT.TTree("t", "t")
        x = np.zeros(1, dtype=np.float32)
        tree.Branch("x", x, "x/F")
        for i in range(2):
            x[0] = float(i)
            tree.Fill()
        reader = ROOT.TTreeReader()
        value = ROOT.TTreeReaderValue[np.float32](reader, "x")
        assert not value.IsValid()
        reader.SetTree(tree)
        assert value.IsValid()
        reader.SetEntry(1)
        assert value.__deref__() == 1.0
    finally:
        yield_from.close()


def test_a_reader_of_a_leaf_name_two_branches_share_warns_and_reads_the_first(capsys):
    first = LeafInfo("C1.fPx", "C1.fPx", "", "Float_t", "TLeafElement", 1, None, "C1", False, False)
    again = LeafInfo("C2.fPx", "C2.fPx", "", "Float_t", "TLeafElement", 1, None, "C2", False, False)
    tree = Fake([first, again], [_branch("C1", [first]), _branch("C2", [again])],
                {"C1.fPx": np.float32(1.0), "C2.fPx": np.float32(2.0)})  # fmt: skip

    class FakeReader:
        _tree = tree

    value = ROOT.TTreeReaderValue[np.float32](FakeReader(), "fPx")
    assert value.IsValid() and value.GetBranchName() == "fPx"
    assert "Warning in <AddReader>: Ignored branch fPx because a branch with the same name" in (
        capsys.readouterr().err
    )


def test_a_collection_of_objects_reads_as_an_element_per_index(monkeypatch):
    held = LeafInfo("arr", "arr", "", "TClonesArray", "TLeafElement", 1, None,
                    "arr", False, False)  # fmt: skip
    tree = Fake([held], [_branch("arr", [held], "TClonesArray", clones="Particle")],
                {"arr": {"fPosX": np.array([1.0, 2.0])}})  # fmt: skip

    class FakeReader:
        _tree = tree

    tree._read_entry = 0
    array = ROOT.TTreeReaderArray[np.float64](FakeReader(), "arr")
    assert list(array) == [{"fPosX": 1.0}, {"fPosX": 2.0}]
    assert array.GetSize() == 2 and not array.IsEmpty() and array.At(1) == {"fPosX": 2.0}
    monkeypatch.setattr(made_reader, "made_from", lambda name, members: (name, members))
    assert array[0] == ("Particle", {"fPosX": 1.0})


def test_a_reader_of_a_branch_the_tree_does_not_have_is_refused_by_name():
    tree = ROOT.TTree("t", "t")
    reader = ROOT.TTreeReader(tree)
    with pytest.raises(KeyError, match="has no branch called 'nope'"):
        ROOT.TTreeReaderValue[np.float32](reader, "nope")


def test_a_value_reader_hands_out_the_class_it_was_declared_for(monkeypatch):
    """``TTreeReaderValue<ClassC> fC(reader, "fC")`` of a branch split out of a ``ClassB``:
    the branch carries the class it was split from, and ROOT types its reader from the ``T``
    it was declared with, so the object handed over is a ``ClassC`` - roottest's
    selectorreader/Options reads ``fC->GetEv()``, which a ``ClassB`` does not have."""
    outer = LeafInfo("ClassB_branch", "ClassB_branch", "", "ClassB", "TLeafElement",
                     1, None, "ClassB_branch", False, False)  # fmt: skip
    member = LeafInfo("fC", "fC", "", "ClassB", "TLeafElement", 1, None, "fC", False, False)
    top = _branch("ClassB_branch", [outer], "ClassB", children=[_branch("fC", [member], "ClassB")])
    tree = Fake([outer, member], [top], {"fC": {"fPx": 101.0, "fEv": 201}})
    tree._read_entry = 0

    class FakeReader:
        _tree = tree

    class ClassC:
        _cxx_layout_ = ("ClassC", (), (("fPx", "Float_t", "", ()), ("fEv", "Int_t", "", ())))
        def __init__(self):
            self.fPx = 0.0
            self.fEv = 0

        def GetEv(self):
            return self.fEv

    class ClassB:
        _cxx_layout_ = ("ClassB", (), ())

    monkeypatch.setitem(declared, "ClassC", ClassC)
    monkeypatch.setitem(declared, "ClassB", ClassB)
    value = ROOT.TTreeReaderValue["ClassC"](FakeReader(), "fC")
    assert isinstance(value.__deref__(), ClassC) and value.GetEv() == 201
    # the branch's own class answers for a template argument naming no class declared here,
    # and for a reader made with no template argument at all
    assert isinstance(ROOT.TTreeReaderValue["Elsewhere"](FakeReader(), "fC").__deref__(), ClassB)
    assert isinstance(ROOT.TTreeReaderValue(FakeReader(), "fC").__deref__(), ClassB)


def test_a_collection_its_writer_did_not_split_reads_as_its_elements(monkeypatch):
    """A ``std::vector`` branch with no split holds its elements in the block the file's
    streamer information names ``This``, one record each; ROOT's reader hands a macro the
    objects, so ``vp[i].fPosX`` is that element's member - roottest's
    selectorreader/CollectionClasses prints them."""
    held = LeafInfo("vp", "vp", "", "vector<Particle>", "TLeafElement", 1, None,
                    "vp", False, True)  # fmt: skip
    rows = {"This": [{"fPosX": 110.0}, {"fPosX": 111.0}]}
    tree = Fake([held], [_branch("vp", [held], "vector<Particle>")], {"vp": rows})
    tree._read_entry = 0

    class FakeReader:
        _tree = tree

    class Particle:
        _cxx_layout_ = ("Particle", (), (("fPosX", "double", "", ()),))

        def __init__(self):
            self.fPosX = 0.0

    monkeypatch.setitem(declared, "Particle", Particle)
    array = ROOT.TTreeReaderArray["Particle"](FakeReader(), "vp")
    assert [element.fPosX for element in array] == [110.0, 111.0]
    assert array.GetSize() == 2 and array.At(1).fPosX == 111.0
    # a class never declared here keeps the record the file read
    monkeypatch.delitem(declared, "Particle", raising=False)
    kept = ROOT.TTreeReaderArray[Particle](FakeReader(), "vp")
    assert list(kept) == [{"fPosX": 110.0}, {"fPosX": 111.0}]
