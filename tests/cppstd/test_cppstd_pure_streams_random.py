"""Local streams and local engines exercise generic state without ROOT fixtures."""

from __future__ import annotations

import copy

from xrdroot.cppstd import random, streams


def test_stream_manipulators_keep_boolean_formatting_and_reset_width_after_one_field() -> None:
    output = streams.ostringstream()
    output << streams.boolalpha << True << " " << streams.noboolalpha << False
    output << " " << streams.setfill("0") << streams.setw(4) << 7 << " " << 8
    assert output.str() == "true 0 0007 8"


def test_formatter_registration_uses_the_nearest_registered_base() -> None:
    class Base:
        pass

    class Derived(Base):
        pass

    streams.stream_formatter(Base, lambda output, value: output << "base")
    output = streams.ostringstream()
    output << Derived()
    assert output.str() == "base"


def test_the_twister_keeps_the_standards_ten_thousandth_word_and_seed_sequence() -> None:
    engine = random.mt19937()
    engine.discard(9999)
    assert engine() == 4123659995
    seeded = random.mt19937(1)
    reseeded = random.mt19937(7)
    reseeded.seed(1)
    assert seeded() == reseeded()
    assert seeded() == 4282876139
    assert random.mt19937.min() == 0 and random.mt19937.max() == 4294967295


def test_normal_distribution_reuses_its_pair_then_reset_uses_the_current_engine_state() -> None:
    engine = random.mt19937(9)
    distribution = random.normal_distribution(5, 2)
    distribution(engine)
    before = engine._state.tobytes(), tuple(engine._words)
    distribution(engine)
    assert (engine._state.tobytes(), tuple(engine._words)) == before
    distribution(engine)
    clone = copy.deepcopy(engine)
    distribution.reset()
    fresh = random.normal_distribution(5, 2)
    assert distribution(engine) == fresh(clone)
