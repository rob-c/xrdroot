"""``TClass::BuildRealData`` and ``GetListOfRealData``: a class's persistent members flattened,
each at the place in the object it is held at, as ROOT lists them for ``checkRealData``.

The offsets below are the ones a compiler lays the same declarations out with: each member at
the next place its own alignment allows, ``int`` 4 bytes and aligned to 4, ``double`` 8 and
aligned to 8, and an object of a class aligned to the widest of its members.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect
from xrdroot.pycint.execute import run_source
from xrdroot.pyroot.core import reflection


@pytest.fixture(autouse=True)
def _no_real_data_carried_over():
    """The lists ROOT caches with a class are this module's, so each test starts without
    one - a listing built by an earlier test would otherwise be the one read here."""
    reflection.REAL_INFOS.clear()
    yield
    reflection.REAL_INFOS.clear()

SOURCE = """
struct Inner {
   int a;
   double b;
};

struct Base {
   int fromBase;
};

struct Outer : Base {
   double x;
   Inner in;   //! a member nothing persists
   Inner kept;
   int arr[3];
   Inner *pointed;
};
"""


def _declared():
    run_source(SOURCE, "real_data.C", call=False)


def _listed(name):
    cls = ROOT.TClass.GetClass(name)
    cls.BuildRealData()
    return [(data.GetName(), data.GetThisOffset()) for data in cls.GetListOfRealData()], cls


def test_a_classes_members_are_listed_at_the_offsets_a_compiler_gives_them():
    _declared()
    listed, cls = _listed("Outer")
    expect(
        (
            listed,
            [
                ("x", 8),
                ("kept", 16),
                ("kept.a", 16),
                ("kept.b", 24),
                ("arr[3]", 32),
                ("pointed", 48),
                ("fromBase", 0),
            ],
        ),
        (cls.GetName(), "Outer"),
    )


def test_a_member_of_a_class_of_its_own_is_followed_by_that_objects_members():
    _declared()
    listed, cls = _listed("Inner")
    inner = cls.GetListOfRealData().At(1)
    expect(
        (listed, [("a", 0), ("b", 8)]),
        (inner.GetName(), "b"),
        (inner.GetOffset(), 8),
        (inner.GetDataMember().GetTypeName(), "double"),
        (inner.GetClass().GetName(), "Inner"),
        (inner.GetTitle(), "Description of persistent data members"),
        (inner.IsPointer(), False),
        ([data.GetName() for data in cls.GetListOfRealData()], ["a", "b"]),
    )


def test_a_member_a_comment_marks_transient_is_not_a_persistent_member():
    """``Inner in; //! ...`` is held in memory and written nowhere, so ROOT lists it not.

    A base's members follow the class's own, as ROOT lists them, though the base's bytes come
    first in the object."""
    _declared()
    listed, _cls = _listed("Outer")
    assert "in" not in [name for name, _offset in listed]
    assert [data.GetName() for data in ROOT.TClass.GetClass("Outer").GetListOfRealData()][:2] == [
        "x",
        "kept",
    ]


def test_a_pointer_member_is_not_followed_into_the_object_it_points_at():
    _declared()
    listed, _cls = _listed("Outer")
    assert not [name for name, _offset in listed if name.startswith("pointed.")]
    pointed = ROOT.TClass.GetClass("Outer").GetListOfRealData().At(5)
    expect((pointed.IsPointer(), True), (pointed.GetThisOffset(), 48))


def test_the_real_data_of_a_class_is_built_once_and_handed_over_as_a_list():
    _declared()
    cls = ROOT.TClass.GetClass("Inner")
    first = cls.GetListOfRealData()
    second = cls.GetListOfRealData()
    expect(
        (first is second, False),
        ([data.GetName() for data in first], ["a", "b"]),
        (first.GetName(), "TList"),
        (first.GetTitle(), "Doubly linked list"),
    )


def test_build_real_data_takes_a_class_to_build_instead():
    """ROOT lets a caller hand ``BuildRealData`` the ``TClass`` to build, which
    ``emulatePairs`` does with a pair's."""
    _declared()
    ROOT.TClass.GetClass("Outer").BuildRealData(ROOT.TClass.GetClass("Inner"))
    assert [data.GetName() for data in ROOT.TClass.GetClass("Inner").GetListOfRealData()] == [
        "a",
        "b",
    ]


def test_an_array_of_a_standard_class_is_as_wide_as_its_values():
    """``std::array<T, N>`` is ``N`` values of ``T``, which ROOT names ``m_a[3]``."""
    run_source(
        '#include <array>\nstruct Held {\n   std::array<float, 3> m_a;\n'
        "   int after;\n};\n",
        "real_data_array.C",
        call=False,
    )  # fmt: skip
    listed, _cls = _listed("Held")
    expect((listed, [("m_a[3]", 0), ("after", 12)]))


def test_a_container_member_is_one_member_and_not_a_class_of_its_own():
    """A ``std::vector``'s elements are not the object's members, and the kit writes a
    vector without knowing how wide one is, so it is listed where it is held and no more."""
    run_source(
        "#include <vector>\nstruct Holder {\n   std::vector<double> m_v;\n   int n;\n};\n",
        "real_data_vector.C",
        call=False,
    )  # fmt: skip
    listed, _cls = _listed("Holder")
    expect((listed, [("m_v", 0), ("n", 24)]))


def test_a_class_only_a_file_describes_lists_no_members_it_has_no_layout_for():
    """Nothing here lays out a class of a template instantiation nothing declares, so it is
    one member of no width rather than a refusal."""
    cls = ROOT.TClass.GetClass("NothingDeclares<int>")
    cls.BuildRealData()
    expect((cls.GetListOfRealData().GetEntries(), 0))
    assert reflection.REAL_INFOS["NothingDeclares<int>"] == []


def test_a_class_that_holds_an_object_of_itself_stops_at_the_one_object():
    """A layout may say a class holds itself, which no C++ class can and an emulated one can:
    the walk stops rather than following it for ever."""
    run_source("struct Self { Self inner; int n; };\n", "real_data_self.C", call=False)
    listed, _cls = _listed("Self")
    expect((listed, [("inner", 0), ("n", 0)]))


def test_a_member_of_a_class_of_its_own_is_placed_by_that_classs_width():
    """The member after an object starts where that object ends - a ``std::vector`` 24 bytes
    wide - and a member a comment marks transient takes room in no listing."""
    run_source(
        "struct WithVec {\n   std::vector<int> v;\n   int skip;  //! not written\n   int n;\n};\n"
        "struct Holder2 { WithVec w; };\n",
        "real_data_holder.C",
        call=False,
    )  # fmt: skip
    listed, _cls = _listed("Holder2")
    expect((listed, [("w", 0), ("w.v", 0), ("w.n", 24)]))


def test_a_member_of_a_class_with_a_base_is_placed_by_the_bases_it_carries():
    """A class sized as a member starts with its bases' bytes, which is what places the
    member after it, and its array members take their values' room."""
    run_source(
        "struct Base2 { int fromBase; };\nstruct Based : Base2 { int arr[2]; };\n"
        "struct HoldsBased { Based b; int after; };\n",
        "real_data_based.C",
        call=False,
    )  # fmt: skip
    listed, _cls = _listed("HoldsBased")
    expect((listed, [("b", 0), ("b.arr[2]", 4), ("b.fromBase", 0), ("after", 12)]))


def test_asking_for_the_real_data_of_a_class_builds_what_was_never_asked_for():
    """``GetListOfRealData`` builds the list a caller never built, as ROOT does on demand."""
    _declared()
    cls = ROOT.TClass.GetClass("Inner")
    assert [data.GetName() for data in cls.GetListOfRealData()] == ["a", "b"]
