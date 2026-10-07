"""The parts of writing objects the donor has no example of, and what is refused.

``test_root_wbranch.py`` holds every branch to ROOT's bytes; here are the
packings of a ``Double32_t`` and a ``Float16_t`` the donor did not use, the
classes a macro may declare and which of them are refused, an object
streamed with its ``TObject``, a clones array streamed object by object, a
collection of a class of plain members, and entries given one at a time.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot import UnsupportedFeatureError
from xrdroot.interp import _packing
from xrdroot.wbranch import nested
from xrdroot.wclasses import Member, checksum, declared, harvested
from xrdroot.wpacking import DOUBLE32, FLOAT16, pack_member, packed_size

VALUES = np.array([-3.0, -1.0, -0.123456, 0.0, 0.5, 2.718281828, 3.1])


#: Each recipe: the type, the comment, how wide a value is on file, and how far from the
#: value - clipped to the range, if there is one - its packing may come back.
RECIPES = [
    (DOUBLE32, "", 4, (None, 1e-7)),
    (DOUBLE32, "[-pi,pi]", 4, ((-np.pi, np.pi), 2 * np.pi / 2**32)),
    (DOUBLE32, "[-pi,pi,20] twenty bits", 4, ((-np.pi, np.pi), 2 * np.pi / 2**20)),
    (DOUBLE32, "[0,0,14] a mantissa", 3, (None, 2.0**-14)),
    (DOUBLE32, "[0,0,20] too wide a mantissa: a float", 4, (None, 1e-7)),
    (FLOAT16, "", 3, (None, 2.0**-12)),
    (FLOAT16, "[-2,2,10]", 4, ((-2.0, 2.0), 4 / 2**10)),
    (FLOAT16, "[0,0,6]", 3, (None, 2.0**-6)),
]


@pytest.mark.parametrize(("stype", "title", "size", "bounds"), RECIPES)
def test_a_packed_float_reads_back_as_the_reader_unpacks_its_recipe(stype, title, size, bounds):
    raw = pack_member(stype, title, VALUES)
    assert len(raw) == size * len(VALUES) and packed_size(stype, title) == size
    prim, unpack = _packing(title, stype == DOUBLE32)
    assert prim.itemsize == size
    limits, step = bounds
    wanted = VALUES if limits is None else np.clip(VALUES, *limits)
    tolerance = step if limits is not None else step * np.abs(VALUES)
    assert np.all(np.abs(unpack(raw) - wanted) <= tolerance)


def test_a_value_in_a_range_is_the_whole_number_of_steps_root_writes():
    """ROOT wrote -2.0 into ``[-10,10,12]`` as 1638 steps, the donor's ``fP`` says."""
    assert pack_member(DOUBLE32, "[-10,10,12] a packed double", [-2.0]) == bytes.fromhex("00000666")
    assert pack_member(FLOAT16, "[0,0,8] a packed float", [-2.0]) == bytes.fromhex("800200")


def test_a_recipe_root_cannot_read_is_refused_rather_than_guessed():
    with pytest.raises(UnsupportedFeatureError, match="not a range ROOT's GetRange reads"):
        pack_member(DOUBLE32, "[a,b,c]", [1.0])
    assert pack_member(5, "", [1.5]) == bytes.fromhex("3fc00000")


class Hit:
    _cxx_layout_ = ("Hit", (), (("fE", "Double_t", "energy", ()), ("fN", "Int_t", "", ())))

    @staticmethod
    def Class_Version() -> int:
        return 3


def test_a_macros_class_is_described_as_its_declaration_and_classdef_say():
    layout = declared(Hit)
    assert (layout.name, layout.version, [m.stype for m in layout.members]) == ("Hit", 3, [8, 3])
    assert layout.checksum == checksum("Hit", layout.elements())
    assert declared(type("Plain", (), {"_cxx_layout_": ("Plain", (), ())})).version == 1


@pytest.mark.parametrize(
    ("layout", "words"),
    [
        (("Derived", ("TObject",), ()), "derives from TObject"),
        (("Holder", (), (("fV", "std::vector<float>", "", ()),)), "is a std::vector<float>"),
        (("Array", (), (("fA", "float", "", (3,)),)), r"is a float\[3\]"),
    ],
)
def test_a_class_of_more_than_plain_numbers_is_refused_by_name(layout, words):
    with pytest.raises(UnsupportedFeatureError, match=words):
        declared(type(layout[0], (), {"_cxx_layout_": layout}))


def test_a_class_root_has_is_split_only_when_its_members_allow():
    assert harvested("TH1F") is None and harvested("NoSuchClass") is None
    assert harvested("TVector3") is None  # its TObject is a base, which is not split here
    coordinates = harvested("ROOT::Math::PxPyPzE4D<double>")
    assert [m.name for m in coordinates.members] == ["fX", "fY", "fZ", "fT"]
    with pytest.raises(UnsupportedFeatureError, match="no description of member by member"):
        nested(Member("fH", 62, "TH1F", "", 8))
