"""``tree.branch`` as ``GetBranchAttr`` reads it: an alias, a branch, then a leaf.

PyROOT pythonizes ``TTree.__getattr__`` by asking ROOT's ``GetBranchAttr``
(``bindings/pyroot/pythonizations/src/TTreePyz.cxx:187``) for the name, which takes it as an
*alias* first, then as a branch (``name``, or ``name.`` because a sub-branch is named with a
trailing dot), and only then as a leaf: the tree's leaf of that name, the branch's own leaf of
that name, or the one leaf of a branch that holds exactly one. A branch holding an object
answers the object it points at, which is why ``entry.structb`` has the struct's members.

The numbers here are ROOT 6.40's own: the leaf names and titles a leaflist writes were read off
the oracle (`branch [myint1b] title=[myint1balias/I]`, `leaf [myint1balias]`), and the
multi-dimensional title's dims are what ``WrapLeaf``'s ``getMultiDims`` parses out of one.
"""

from __future__ import annotations

import array

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import fresh
from xrdroot.pyroot.trees.core import array_dims


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def _tree_of_one_entry(name: str = "t") -> tuple[ROOT.TFile, ROOT.TTree]:
    """A tree of one entry with a struct branch, a 2-D array, a float and a vector."""
    ROOT.gInterpreter.Declare("struct MyStruct99 { int myint1; int myint2; };")
    handle = ROOT.TFile(f"{name}.root", "RECREATE")
    tree = ROOT.TTree(name, name)
    struct = ROOT.MyStruct99()
    struct.myint1, struct.myint2 = 7, 14
    tree.Branch("structb", struct)
    tree.Branch("array2Db", array.array("f", [float(at) for at in range(20)]), "array2Db[10][2]/F")
    tree.Branch("floatb", array.array("f", [7.0]), "floatb/F")
    tree.Branch("vectorb", ROOT.std.vector("double")([1.0, 2.0]))
    tree.Fill()
    handle.Write()
    handle.Close()
    return handle, tree


def _read_back(name: str = "t") -> tuple[ROOT.TFile, ROOT.TTree]:
    _handle, _written = _tree_of_one_entry(name)
    handle = ROOT.TFile(f"{name}.root")
    tree = handle.Get(name)
    tree.GetEntry(0)
    return handle, tree


def test_an_alias_is_read_before_a_branch_of_the_same_name() -> None:
    """ROOT asks ``TTree::GetAlias`` first, so a name that is both is the alias's."""
    handle, tree = _read_back()
    tree.SetAlias("myalias", "floatb")
    assert tree.myalias == pytest.approx(7.0)
    tree.SetAlias("floatb", "array2Db")  # an alias over the branch's own name: the alias wins
    assert tree.floatb is not None and getattr(tree.floatb, "shape", None) == (20,)
    handle.Close()


def test_a_struct_branch_is_the_object_it_points_at_rather_than_a_row_of_it() -> None:
    """ROOT hands back ``*(void **)branch->GetAddress()`` as an instance of the branch's class,
    so ``entry.structb.myint1`` answers where this kit's row of members would not."""
    handle, tree = _read_back()
    held = tree.structb
    assert type(held).__name__ == "MyStruct99"
    assert (held.myint1, held.myint2) == (7, 14)
    handle.Close()


def test_a_branch_of_vectors_is_a_vector_and_not_an_object() -> None:
    """Only a branch whose value is a row of members is an object: a ``std::vector`` is one of
    the containers a reader already hands out."""
    handle, tree = _read_back()
    assert list(tree.vectorb) == [1.0, 2.0]
    handle.Close()


def test_a_branchs_one_leaf_is_found_by_its_own_name() -> None:
    """ROOT falls back to the single leaf of a branch - ``MyStruct99``'s ``myint1`` - so a
    program may read ``entry.myint1`` where the branch is called ``structb``."""
    handle, tree = _read_back()
    assert tree.myint1 == 7
    handle.Close()


def test_a_name_the_tree_has_not_is_refused_in_a_full_sentence() -> None:
    handle, tree = _read_back()
    asked = "nosuchthing"  # a name the tree can only be asked for by string
    with pytest.raises(AttributeError, match=f"ROOT's TTree has {asked}"):
        getattr(tree, asked)
    handle.Close()


def test_a_leaf_titles_brackets_are_the_dimensions_root_reads_out_of_it() -> None:
    """``getMultiDims`` parses every ``[n]`` of a leaf's title, so ``array2Db[10][2]`` is two
    dimensions and a one-dimensional ``x[20]`` is one."""
    assert array_dims("array2Db[10][2]") == (10, 2)
    assert array_dims("array2Db[10][2]/F") == (10, 2)
    assert array_dims("x[20]") == (20,)
    assert array_dims("x") == ()
    assert array_dims("x[n]") == ()  # a counted leaf's size is another leaf's, not a number


def test_a_row_is_not_made_into_an_object_it_cannot_be() -> None:
    """Three rows answer themselves instead: one of a class this session never declared, one
    of a class whose constructor needs arguments - ROOT's IO makes such an object without
    calling its constructor, which a Python one cannot - and a member whose name is private
    to the class it came from."""
    from xrdroot.pyroot.trees.objects import row_object

    assert row_object({"a": 1}, "NeverDeclaredAnywhere") is None
    ROOT.gInterpreter.Declare(
        "struct NeedsAnArgument99 { int &held; NeedsAnArgument99(int &at) : held(at) {} };"
    )
    assert row_object({"a": 1}, "NeedsAnArgument99") is None
    ROOT.gInterpreter.Declare("struct WithPrivate99 { int a; };")
    made = row_object({"a": 5, "_hidden": 9}, "WithPrivate99")
    assert made is not None and made.a == 5 and not hasattr(made, "_hidden")


def test_a_shape_whose_product_is_not_the_leaves_length_is_left_flat() -> None:
    """A title's dimensions are taken only when they count the values the leaf holds; one that
    does not is left as the flat run of values it is."""
    from xrdroot.pyroot.trees.core import python_value
    from xrdroot.pyroot.trees.layout import LeafInfo

    leaf = LeafInfo("x", "x", "x[2][2]", "Float_t", "TLeafF", 6, None, "x", False, False)
    flat = python_value(np.zeros(6), leaf)
    assert flat.shape == (6,)


def test_a_branch_whose_one_leaf_is_named_otherwise_is_reached_by_the_branch() -> None:
    """ROOT's third rule: a branch holding exactly one leaf answers with it, whatever the name
    asked for - which is how a leaflist's branch is read by the branch's own name."""
    from xrdroot.pyroot.trees.layout import BranchInfo, LeafInfo
    from xrdroot.pyroot.trees.reading import _Reading

    leaf = LeafInfo("ianalias", "i.ianalias", "ianalias", "Int_t", "TLeafI", 1, None, "i",
                    False, False)  # fmt: skip
    branch = BranchInfo("i", "ianalias/I", "", [leaf], 1, 8, 0, 1, 8)
    reader = object.__new__(_Reading)
    reader._layout_cache = []  # no tree behind it: the rule itself is what is measured
    assert reader._asked_leaf("i", branch) is leaf
    other = leaf._replace(name="ianother", column="i.ianother")
    two = branch._replace(leaves=[leaf, other])
    assert reader._asked_leaf("c", two) is None  # a branch of two leaves answers neither


def test_a_leaves_title_shapes_it_when_the_values_it_counts_are_all_there() -> None:
    """The dimensions ROOT parses are the array's shape, so a leaf of four values titled
    ``x[2][2]`` is a ``(2, 2)`` view of them."""
    from xrdroot.pyroot.trees.core import python_value
    from xrdroot.pyroot.trees.layout import LeafInfo

    leaf = LeafInfo("x", "x", "x[2][2]", "Float_t", "TLeafF", 4, None, "x", False, False)
    shaped = python_value(np.arange(4, dtype="float32"), leaf)
    assert shaped.shape == (2, 2) and float(shaped[1][0]) == 2.0


def test_a_branch_of_several_leaves_is_no_object_at_all() -> None:
    """ROOT's ``ResolveBranch`` answers an object only for a branch whose address is one
    object's; a branch of several leaves has a buffer of each and none to hand back."""
    from xrdroot.pyroot.trees.layout import BranchInfo, LeafInfo
    from xrdroot.pyroot.trees.reading import _Reading

    one = LeafInfo("a", "a", "a", "Int_t", "TLeafI", 1, None, "b", False, False)
    two = [one, one._replace(name="c", column="c")]
    branch = BranchInfo("b", "a/I:c/I", "", two, 1, 8, 0, 1, 8)
    reader = object.__new__(_Reading)
    reader._layout_cache = []
    assert reader._branch_value(branch) is None
