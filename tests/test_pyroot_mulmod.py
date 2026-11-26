"""``ranluxpp/mulmod.h``: the 576-bit arithmetic ROOT's ranlux++ generator works in.

ROOT writes these four functions limb by limb to avoid 128-bit integers; this kit does the
same arithmetic in Python's integers and masks the answer back into the limbs. The
expectations below are ROOT's own, from ``math/mathcore/test/mulmod.icc`` and
``ranlux_lcg.cxx``.
"""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT

#: ``m = 2 ** 576 - 2 ** 240 + 1``: the modulus every reduction here is by.
MODULUS = (1 << 576) - (1 << 240) + 1
MAX = (1 << 64) - 1


def limbs(value: int, count: int = 9) -> list[int]:
    return [(value >> (64 * index)) & MAX for index in range(count)]


def number(words: list[int]) -> int:
    return sum((int(word) & MAX) << (64 * index) for index, word in enumerate(words))


def test_the_product_of_two_nine_limb_numbers_is_eighteen_limbs() -> None:
    first, second, out = [0] * 9, [0] * 9, [0] * 18
    first[0], second[0] = 2, 3
    ROOT.multiply9x9(first, second, out)
    assert out == [6] + [0] * 17
    first, second = list(range(1, 10)), list(range(1, 10))
    ROOT.multiply9x9(first, second, out)
    assert out[:10] == [1, 4, 10, 20, 35, 56, 84, 120, 165, 200]
    assert out[10:] == [224, 236, 235, 220, 190, 144, 81, 0]


def test_the_whole_range_multiplies_as_roots_limbs_do() -> None:
    """``(2 ** 576 - 1) ** 2``, whose expectation ROOT checked against its assembly."""
    everything = [MAX] * 9
    out = [0] * 18
    ROOT.multiply9x9(everything, everything, out)
    assert out[0] == 1 and out[1:9] == [0] * 8
    assert out[9] == 0xFFFFFFFFFFFFFFFE and out[10:] == [MAX] * 8


def test_a_number_below_the_modulus_is_its_own_residue() -> None:
    out = [0] * 9
    mul = [0] * 18
    mul[8] = 1  # 2 ** 512, which is below m
    ROOT.mod_m(mul, out)
    assert out == [0] * 8 + [1]


def test_the_modulus_itself_reduces_to_nothing() -> None:
    out = [0] * 9
    ROOT.mod_m(limbs(MODULUS, 18), out)
    assert out == [0] * 9


def test_twice_the_modulus_and_beyond_reduce_to_nothing_too() -> None:
    out = [0] * 9
    for factor in (1, 2, 3, 1000):
        ROOT.mod_m(limbs(MODULUS * factor, 18), out)
        assert out == [0] * 9


def test_a_number_one_past_the_modulus_is_the_smallest_residue_it_has() -> None:
    """``2 ** 576`` is ``m + (2 ** 240 - 1)``: the first 240 bits all set."""
    out = [0] * 9
    mul = [0] * 18
    mul[9] = 1
    ROOT.mod_m(mul, out)
    assert out[:3] == [MAX] * 3 and out[3] == 0x0000FFFFFFFFFFFF and out[4:] == [0] * 5


def test_a_product_is_taken_modulo_the_modulus_in_place() -> None:
    """``(2 ** 288) ** 2`` is ``2 ** 576``, which is the same residue as above."""
    first, inout = [0] * 9, [0] * 9
    first[4] = inout[4] = 1 << 32
    ROOT.mulmod(first, inout)
    assert inout[:3] == [MAX] * 3 and inout[3] == 0x0000FFFFFFFFFFFF and inout[4:] == [0] * 5


def test_a_multiple_of_the_modulus_multiplies_to_nothing() -> None:
    first, inout = limbs(MODULUS), [1] * 9
    ROOT.mulmod(first, inout)
    assert inout == [0] * 9


def test_a_power_is_taken_modulo_the_modulus() -> None:
    base, res = [0] * 9, [0] * 9
    base[0] = 2
    ROOT.powermod(base, res, 15)
    assert res == [1 << 15] + [0] * 8
    ROOT.powermod(base, res, 576)
    assert res[:3] == [MAX] * 3 and res[3] == 0x0000FFFFFFFFFFFF and res[4:] == [0] * 5


def test_a_power_may_write_over_the_base_it_reads() -> None:
    """``powermod(a, a, n)``: ROOT copies the base first, as this does."""
    a = [0] * 9
    a[0] = 2
    ROOT.powermod(a, a, 15)
    assert a == [1 << 15] + [0] * 8


def test_a_limb_as_wide_as_the_machine_word_is_kept_unsigned() -> None:
    """``mulmod`` and ``mod_m`` mask each limb to 64 bits, as ``uint64_t`` assignment does."""
    out = [0] * 9
    ROOT.mod_m([MAX + 2] + [0] * 17, out)  # 2 ** 64 + 1 as uint64_t's, which is 1
    assert out[0] == 1 and out[1:] == [0] * 8


@pytest.mark.parametrize(
    ("state", "carry", "residue"),
    [
        (0, 0, 0),
        (1, 0, 1),
        (0, 1, 1),
        (
            number([1 + (1 << 24) + (1 << 48), (1 << 8) + (1 << 32) + (1 << 56),
                    (1 << 16) + (1 << 40)] * 3),
            1,
            number([1, 0, 0, 0x0001000000000000, 0x0100000100000100, 0x0000010000010000,
                    0x0001000001000001, 0x0100000100000100, 0x0000010000010000]),
        ),
        ((1 << 576) - 2, 1, MODULUS - 1),
    ],
)
def test_a_state_and_its_carry_are_the_residue_root_keeps(
    state: int, carry: int, residue: int
) -> None:
    """``to_lcg``: ``r - (r >> 336) + c``, which is the limb loop ROOT runs."""
    out = [0] * 9
    ROOT.to_lcg(limbs(state), carry, out)
    assert number(out) == residue
