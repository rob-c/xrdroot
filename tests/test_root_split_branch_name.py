"""What a split branch of an object is called, and what its leaf list is.

``roottest/root/tree/stl/runsimple.C`` writes ``t->Branch("obj.", &myobj)`` for
a ``std::vector<pair<int,long> >`` and reads it back with ``SetBranchAddress``.
The trailing dot is ROOT's mark that the name is a *prefix*: the branch keeps it
and every member it splits out is named ``prefix + member``, so ROOT 6.40.04
names a ``TNamed``'s split branch ``named.`` with the leaves ``named.:TNamed``
under it and the sub-branches ``named.TObject`` (no leaves of its own),
``named.fName`` and ``named.fTitle`` - measured with the oracle in
``~/.local/root-oracle`` (splitlevel 99; at splitlevel 0 the sub-branches are
absent and only the whole-object leaf is left).

The measurements here are the ones three rounds of ``tree/stl`` work needed and
did not have:

- a macro class with a base is *written whole* - one leaf, the class's own name -
  because :func:`~xrdroot.wclasses.declared` refuses a base it has no donor
  description of, and a whole-object branch is what ROOT makes of the same
  object without a dictionary;
- a macro class without a base is split, and each member's sub-branch is
  ``obj.fZ`` with the leaf ``obj.fZ:Double_t``, the naming ROOT uses;
- ``.typedef`` and ``.autodict 0`` - the two commands ``rootlogon.C`` runs
  before ``runsimple.C`` - change neither the name nor the leaf list, so they
  are not what moves this test;
- a ``std::vector`` of a class is refused today, with the sentence that says so.

These are characterisations, kept deliberately: the numbers are ROOT's, the
refusal is xrdroot's, and the difference is the work still to do.
"""

from __future__ import annotations

import contextlib
import io
import re

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot import UnsupportedFeatureError
from xrdroot.pycint.execute import process_line, run_source
from xrdroot.pycxx.runtime.namespace import DECLARED
from xrdroot.wclasses import declared

#: A class standing on a base of its own, and one that stands on nothing: ROOT splits
#: the second and, with a dictionary, both.
SOURCE = """
struct Point3 { double fX; double fY; Point3() : fX(0), fY(0) {} };
class Two : public Point3 { public: double fZ; Two() : fZ(0) {} };
class Alone { public: double fZ; Alone() : fZ(0) {} };
"""


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    declared_before = dict(DECLARED)
    yield from fresh(tmp_path)
    DECLARED.clear()
    DECLARED.update(declared_before)


def _translate() -> None:
    run_source(SOURCE, "<macro>")


def _leaves(branch) -> list[tuple[str, str]]:
    """A branch's leaves: each one's name and the type ROOT's ``TLeaf::GetTypeName`` gives it."""
    return [(leaf.GetName(), leaf.GetTypeName()) for leaf in branch.GetListOfLeaves() or []]


def _branched(obj) -> tuple[str, str, list[tuple[str, str]], dict[str, list[tuple[str, str]]]]:
    """One ``Branch("obj.", &obj)``: name, title, its leaves, and its sub-branches' leaves."""
    tree = ROOT.TTree("T", "T")
    tree.Branch("obj.", obj)
    branches = tree.GetListOfBranches()
    assert len(branches) == 1  # "obj." is a prefix, so it is one branch whatever it holds
    branch = branches[0]
    under = {sub.GetName(): _leaves(sub) for sub in branch.GetListOfBranches() or []}
    return branch.GetName(), branch.GetTitle(), _leaves(branch), under


def test_a_macro_class_with_a_base_is_written_whole_under_the_name_it_was_given() -> None:
    """ROOT splits ``named.`` into a sub-branch per member; xrdroot cannot, and says why."""
    _translate()
    name, title, leaves, under = _branched(DECLARED["Two"]())
    assert (name, title) == ("obj.", "obj.")  # the trailing dot is ROOT's prefix mark, kept
    assert leaves == [("obj.", "Two")]  # one leaf, named for the branch, typed by the class
    assert under == {}  # nothing is split out of it


def test_a_base_this_writer_has_no_description_of_is_what_leaves_the_object_whole() -> None:
    """The refusal names the base and the way out, which is why the branch above has no splits."""
    _translate()
    with pytest.raises(UnsupportedFeatureError, match="Point3 is a base class this writer"):
        declared(DECLARED["Two"])


def test_a_macro_class_without_a_base_names_each_member_under_the_branch() -> None:
    """ROOT's convention, ``named.fName`` for a ``TNamed``: prefix, dot, member."""
    _translate()
    name, _title, leaves, under = _branched(DECLARED["Alone"]())
    assert name == "obj." and leaves == [("obj.", "Alone")]
    assert under == {"obj.fZ": [("obj.fZ", "Double_t")]}  # ROOT's ``named.fName:TString``


def test_the_typedef_and_autodict_commands_leave_the_split_branch_as_they_found_it() -> None:
    """``rootlogon.C``'s two commands are not what renames the branch, measured both sides."""
    _translate()
    before = _branched(DECLARED["Two"]())
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        assert process_line(".typedef") == 0
        assert process_line(".autodict 0") == 0
    # CINT's heading, then the typedefs of whatever macros this session already translated -
    # which is ROOT's behaviour and depends on the session, so the heading and the shape of the
    # lines are what is asserted here rather than a list another test could change.
    lines = said.getvalue().splitlines()
    assert lines[0] == "List of typedefs"
    assert all(re.match(r"^\S+\s+\d+\s+typedef .+ [A-Za-z_]\w*$", line) for line in lines[1:])
    assert _branched(DECLARED["Two"]()) == before


def test_a_vector_of_pairs_is_refused_rather_than_written_without_its_inner_counts() -> None:
    """``runsimple.C:30`` today: ROOT writes ``obj.first``/``obj.second``, xrdroot refuses."""
    from xrdroot.cppstd.containers import vector

    pairs: object = vector("std::pair<int, long>")()
    pairs.push_back((1, 100))
    tree = ROOT.TTree("T", "T")
    with pytest.raises(UnsupportedFeatureError, match=r"the branch 'obj\.' holds a std::vector"):
        tree.Branch("obj.", pairs)
