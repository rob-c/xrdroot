"""``ROOT::Detail::TRangeStaticCast``/``TRangeDynCast``: the type a range's element has.

C++ gives a range-for's body a ``TObject *`` whatever a collection holds, and ``Class`` is
a static member of that type, so ``bcl->Class()`` answers ``TObject``'s - see
:mod:`xrdroot.pyroot.core.rangecast`. The wrapper a collection's element arrives in is
the object in every other respect, which is what these check.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from pyrootsupport import expect, fresh
from xrdroot.pycint import runtime as rt
from xrdroot.pyroot.core.rangecast import TRangeDynCast, TRangeStaticCast, TypedElement


@pytest.fixture(autouse=True)
def _fresh(tmp_path):
    yield from fresh(tmp_path)


def held(*objects):
    found = ROOT.TList()
    for obj in objects:
        found.Add(obj)
    return found


def test_a_range_over_a_list_gives_each_object_as_the_tobject_the_range_declares():
    base = ROOT.TBaseClass("a base", ROOT.TNamed.Class())
    items = rt.iterate(held(base))
    expect(
        (items[0].Class().GetName(), "TObject"),
        (items[0].GetName(), "a base"),
        (items[0].InheritsFrom("TObject"), True),
    )


def test_anything_a_collection_did_not_hand_out_is_iterated_as_it_was():
    expect(
        (list(rt.iterate([1, 2, 3])), [1, 2, 3]),
        ([pair.second for pair in rt.iterate(ROOT.map["int", "int"]({7: 1}))], [1]),
    )


def test_a_static_cast_names_the_class_it_was_given_and_no_other():
    items = TRangeStaticCast["TBaseClass"](held(ROOT.TBaseClass("b", ROOT.TNamed.Class())))
    expect(
        (items[0].Class().GetName(), "TBaseClass"),
        (items[0].GetName(), "b"),
    )


def test_a_dynamic_cast_keeps_only_the_objects_that_really_are_one():
    base = ROOT.TBaseClass("b", ROOT.TNamed.Class())
    found = held(base)
    expect(
        (TRangeDynCast["TClass"](found), [None]),
        ([item.GetName() for item in TRangeDynCast["TBaseClass"](found)], ["b"]),
    )


def test_a_cast_of_a_class_the_namespace_does_not_have_is_refused_in_a_sentence():
    with pytest.raises(TypeError, match="names no class this namespace has"):
        TRangeStaticCast["NoSuchClass"](held())


def test_the_detail_namespace_offers_the_two_casts_as_root_does():
    expect(
        (ROOT.Detail.TRangeStaticCast, TRangeStaticCast),
        (ROOT.Detail.TRangeDynCast, TRangeDynCast),
    )


def test_the_wrapper_is_the_object_for_everything_the_macro_asks_of_it():
    named = ROOT.TNamed("obj", "title")
    one = TypedElement(named, ROOT.TObject)
    other = TypedElement(named, ROOT.TObject)
    different = TypedElement(ROOT.TNamed("other", ""), ROOT.TObject)
    one.marker = 7
    expect(
        (bool(one == named), True),
        (bool(one == other), True),
        (bool(one != different), True),
        (bool(one == different), False),
        (hash(one), hash(named)),
        (str(one), str(named)),
        (repr(one), repr(named)),
        (bool(one), True),
        (isinstance(one, ROOT.TNamed), True),
        (one.marker, 7),
        (named.marker, 7),
    )
    del one.marker
    expect((hasattr(named, "marker"), False))
