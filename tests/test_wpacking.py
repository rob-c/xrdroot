"""A member's numbers go on file the same way whether they are one or a column.

The hot path of writing objects - and of ``TBufferFile`` streaming the same object a
million times - hands :func:`~xrdroot.wpacking.pack_member` a one-element array, so a
single value is written through ``struct`` and never through NumPy. That shortcut must
give the very bytes the NumPy path gives, including where NumPy converts and ``struct``
would refuse: a float into an integer member, and a number too wide for its member.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from xrdroot.wpacking import BASIC, DOUBLE32, FLOAT16, PACKERS, _packer, pack_member

#: One value of every plain streamer type whose file image ``struct`` can write, with
#: values that exercise the width, the sign and what a conversion must do.
PLAIN = (1, 2, 3, 4, 5, 6, 8, 11, 12, 13, 14, 15, 16, 17, 18)
VALUES = (0, 1, -1, 127, 255, 1000, -1000, 2**31 - 1, -(2**31), 2**53 + 1, 1.5, -1.5, True)


def numpy_bytes(stype: int, value: object) -> bytes:
    """What the NumPy path alone writes for one value of a plain streamer type."""
    return bytes(np.asarray([value]).astype(BASIC[stype][0]).tobytes())


@pytest.mark.parametrize("stype", PLAIN)
@pytest.mark.parametrize("value", VALUES)
def test_a_single_value_is_written_as_a_column_of_one_writes_it(stype: int, value: object) -> None:
    assert pack_member(stype, "", np.asarray([value])) == numpy_bytes(stype, value)


def test_a_float_in_an_integer_member_truncates_towards_zero_as_numpy_does() -> None:
    """``struct`` refuses a float where an ``Int_t`` goes; NumPy truncates it, and ROOT's
    reader gets the truncated number, so that is what is on file."""
    assert pack_member(3, "", np.asarray([1.9])) == bytes.fromhex("00000001")
    assert pack_member(3, "", np.asarray([-1.9])) == bytes.fromhex("ffffffff")
    with pytest.raises(struct.error):
        struct.Struct(">i").pack(1.9)


def test_a_number_too_wide_for_its_member_wraps_as_astype_wraps_it() -> None:
    assert pack_member(3, "", np.asarray([2**31])) == numpy_bytes(3, 2**31) == b"\x80\x00\x00\x00"


def test_a_column_of_values_is_written_whole_whichever_shape_it_comes_in() -> None:
    assert pack_member(3, "", np.asarray([1, 2, 3])) == bytes.fromhex("000000010000000200000003")
    assert pack_member(3, "", [[1, 2], [3, 4]]) == bytes.fromhex("00000001000000020000000300000004")
    assert pack_member(5, "", [1.5]) == bytes.fromhex("3fc00000")


def test_the_struct_a_type_is_written_with_is_made_once_and_reused() -> None:
    PACKERS.pop(3, None)
    assert pack_member(3, "", np.asarray([7])) == bytes.fromhex("00000007")
    assert PACKERS[3].size == 4
    made = _packer(3)
    assert _packer(3) is made and made.format == ">i"


def test_a_packed_float_still_takes_the_recipe_its_comment_gives() -> None:
    assert pack_member(DOUBLE32, "", np.asarray([1.5])) == bytes.fromhex("3fc00000")
    assert pack_member(FLOAT16, "[0,0,8]", np.asarray([-2.0])) == bytes.fromhex("800200")
