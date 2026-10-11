"""``std::bitset`` and ``ROOT::Math::LFSR``: shift-register bit sequences, as ROOT 6.40 makes them.

The sequences are those PRBS.C prints under ROOT; the corner cases follow
``Math/LFSR.h`` line by line: the bits shifted the other way, the bit read
off the register's far end instead of the one fed in, a register let wrap.
"""

from __future__ import annotations

from typing import Any

import pytest

import xrdroot.pyroot as ROOT


def bits(sequence: Any) -> str:
    return "".join(str(int(one)) for one in sequence)


def test_a_bitset_holds_its_width_of_bits_lowest_first() -> None:
    made = ROOT.std.bitset[4]("0110")
    assert (made.size(), made.to_ulong(), str(made), made.count()) == (4, 6, "0110", 2)
    assert (made[1], made[0], made.any(), made.none()) == (True, False, True, False)
    made <<= 3
    assert made.to_string() == "0000" and made.none()
    made[3] = 1
    made >>= 2
    assert made.to_ulong() == 2 and made.flip(0).to_ulong() == 3
    assert made.flip() == ROOT.std.bitset[4](12) and made != ROOT.std.bitset[5](12)
    assert repr(ROOT.std.bitset) == "<std.bitset template>"
    assert len({ROOT.std.bitset[3](5), ROOT.std.bitset[3](5)}) == 1
    assert ROOT.std.bitset[0]().to_string() == ""


def test_the_maximal_sequences_are_roots() -> None:
    taps = ROOT.std.array["unsigned short", 2]([5, 3])
    sequence = ROOT.Math.LFSR.GenerateSequence(ROOT.std.bitset[5]().flip(), taps)
    assert sequence.size() == 31 and bits(sequence) == "0001101110101000010010110011111"
    three = ROOT.Math.LFSR.GenerateSequence(ROOT.std.bitset[3]().flip(), [3, 2])
    assert bits(three) == "0010111"


def test_a_right_shift_the_far_bit_and_wrapping_follow_lfsr_h() -> None:
    seed, generate = ROOT.std.bitset[4]("1001"), ROOT.Math.LFSR.GenerateSequence
    assert bits(generate(seed, [4, 3], False)) == "10110"
    assert bits(generate(seed, [4, 3], True, False, True)) == "100110101111000"
    assert bits(generate(seed, [4, 3], False, False, True)) == "10011"
    # Taps of no maximal sequence bring the seed round sooner - six steps here - and taps
    # that drop the top bit never do: wrapping, they run as long as a maximal sequence.
    assert bits(generate(ROOT.std.bitset[4]("0001"), [4, 2], True, True)) == "010001"
    assert bits(generate(ROOT.std.bitset[4]("1000"), [3, 2], True, True)) == "0" * 15
    assert bits(generate(ROOT.std.bitset[4]("0001"), [4, 2])) == "010001"
    assert bits(generate(ROOT.std.bitset[4]("0011"), [4, 2], True, True)) == "110011"


def test_bad_taps_and_a_zero_seed_are_said_and_give_nothing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ROOT.Math.LFSR.GenerateSequence(ROOT.std.bitset[3](1), [4, 2]).size() == 0
    assert ROOT.Math.LFSR.GenerateSequence(ROOT.std.bitset[3](), [3, 2]).size() == 0
    assert capsys.readouterr().err.splitlines() == [
        "Error in <ROOT::Math::LFSR>: Tap 4 is out of range [1,3]",
        "Error in <ROOT::Math::LFSR>: A non-zero start value is needed",
    ]
    assert ROOT.Math.MathMoreLibrary.Load() is None
