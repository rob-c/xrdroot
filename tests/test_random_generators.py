"""ROOT's generators, draw for draw: ``TRandom3``, ``TRandom2``, ``TRandom1`` and ``TRandom``.

Three things vouch for the streams. ``TRandom3``'s words are NumPy's legacy
Mersenne Twister's, which knows nothing of ROOT, and the first few for three
seeds are pinned as go-hep's ``rrand`` pins them. Every generator is held to
``rootrandom``, ROOT's C++ transcribed a statement at a time, over thousands
of draws taken in every mixture of one at a time and arrays - which is what
checks the fast paths, since each generator here makes its arrays by quite
different means from ROOT's loop. And ``test_hist_fill`` fills histograms
from ``TRandom3`` at ROOT's default seed and gets the very bins go-hep's ROOT
macros wrote into ``tefficiency.root`` and ``tprofile.root``.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import rootrandom
import xrdroot
from xrdroot.random import (
    TRandom,
    TRandom1,
    TRandom2,
    TRandom3,
    gRandom,
    libm,
    ranlux,
    tausworthe,
    trandom,
)

SEEDS = [1, 42, 4357, 12345, 987654321, 2**31 + 5, 2**32 - 1]
#: go-hep's ``TestMT19937Golden``: the first five words for three seeds.
GOLDEN_WORDS = {
    1: [1791095845, 4282876139, 3093770124, 4005303368, 491263],
    12345: [3992670690, 3823185381, 1358822685, 561383553, 789925284],
    4357: [4293858116, 699692587, 1213834231, 4068197670, 994957275],
}


def words(generator, count):
    """The next ``count`` draws of a 32-bit generator as the words they were made from."""
    return (generator.rndm(count) * 2**32).astype(np.int64).tolist()


def mixed(generator, count):
    """``count`` draws taken every way there is: one, a few, many, and arrays that span runs."""
    parts = [generator.rndm(7), np.array([generator.rndm() for _ in range(40)]), generator.rndm(0)]
    parts += [generator.rndm(3000), generator.rndm(1)]
    return np.concatenate([*parts, generator.rndm(count - sum(len(p) for p in parts))])


# -- the streams ------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 42, 12345, 4357, 987654321])
def test_trandom3_draws_the_words_numpys_legacy_mersenne_twister_draws(seed):
    numpy_words = np.random.RandomState(seed).randint(0, 2**32, size=2000, dtype=np.uint32)
    assert words(TRandom3(seed), 2000) == numpy_words.tolist()


@pytest.mark.parametrize("seed", sorted(GOLDEN_WORDS))
def test_trandom3_starts_with_the_words_go_hep_pins(seed):
    assert words(TRandom3(seed), 5) == GOLDEN_WORDS[seed]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    ("made", "oracle"),
    [(TRandom3, rootrandom.MT), (TRandom2, rootrandom.Taus), (TRandom, rootrandom.LCG)],
)
def test_every_generator_draws_what_roots_source_draws(made, oracle, seed):
    got = mixed(made(seed), 5000)
    assert np.array_equal(got, rootrandom.draws(oracle(seed), 5000))


@pytest.mark.parametrize("lux", [0, 1, 2, 3, 4, 10, 24, 30])
@pytest.mark.parametrize("seed", [1, 4357, 2**32 - 1])
def test_ranlux_draws_what_roots_loop_draws_and_leaves_roots_state(seed, lux):
    generator, oracle = TRandom1(seed, lux), rootrandom.Ranlux(seed, lux)
    assert np.array_equal(mixed(generator, 4000), rootrandom.draws(oracle, 4000))
    state = generator.state
    assert np.array_equal(state["fFloatSeedTable"], np.array(oracle.table, dtype=np.float32))
    assert (state["fIlag"], state["fJlag"], state["fCount24"]) == (
        oracle.ilag,
        oracle.jlag,
        oracle.count24,
    )
    assert (state["fCarry"], state["fNskip"], state["fLuxury"]) == (oracle.carry, oracle.nskip, lux)
    assert generator.get_seed() == round(float(oracle.table[0]) * 2**24)


def test_ranlux_tops_small_digits_up_from_nine_digits_back_as_root_does():
    drawn = TRandom1(1, 0).rndm(200_000)
    small = drawn[drawn < 2**-12]
    assert len(small) > 10 and np.all(small * 2**24 != np.floor(small * 2**24))
    assert np.array_equal(drawn, rootrandom.draws(rootrandom.Ranlux(1, 0), len(drawn)))


def test_ranlux_is_an_lcg_whose_numerator_gives_the_digits_roots_loop_makes():
    for fixed in (0, -ranlux.M):
        assert ranlux.advance(fixed, 1000) == fixed
    history = TRandom1(99)._history()
    numerator = ranlux.number(history[14:]) - ranlux.number(history)
    assert -ranlux.M <= numerator <= 0
    oracle = rootrandom.Ranlux(99)
    steps = [round(float(oracle.step()) * 2**24) for _ in range(70)]
    assert ranlux.expand(numerator, 30).tolist() == steps[:30]
    assert ranlux.expand(ranlux.advance(numerator, 30), 40).tolist() == steps[30:]


# -- words that come out zero -----------------------------------------------------------


def test_a_zero_word_is_thrown_away_and_the_next_one_drawn():
    state = TRandom3(5).state
    state["fMt"][10:12] = 0  # tempering keeps zero zero
    state["fCount624"] = 10
    generator = TRandom3()
    generator.set_state(state)
    oracle = rootrandom.MT(5)
    oracle.mt, oracle.count = state["fMt"].tolist(), 10
    assert np.array_equal(generator.rndm(700), rootrandom.draws(oracle, 700))


def test_the_lcg_passing_through_zero_skips_it_as_root_does():
    before_zero = (-12345 * pow(1103515245, -1, 2**31)) % 2**31
    drawn = TRandom(before_zero).rndm(5)
    assert np.array_equal(drawn, rootrandom.draws(rootrandom.LCG(before_zero), 5))
    assert drawn[0] == 4.6566128730774e-10 * 12345  # the step after zero


# -- seeds ------------------------------------------------------------------------------


@pytest.mark.parametrize("made", [TRandom, TRandom1, TRandom2, TRandom3])
def test_a_zero_seed_is_unrepeatable_but_an_ordinary_stream(made):
    a, b = made(0).rndm(100), made(0).rndm(100)
    assert not np.array_equal(a, b)
    assert np.all((a > 0) & (a < 1)) and len(np.unique(a)) == 100


def test_a_zero_seed_is_made_from_a_uuid_the_way_root_makes_it(monkeypatch):
    raw = bytes(range(1, 17))
    monkeypatch.setattr(trandom, "uuid_bytes", lambda: raw)
    monkeypatch.setattr(tausworthe, "uuid_bytes", lambda: raw)
    assert TRandom(0).get_seed() == int.from_bytes(raw[:4], "little")
    registers = [int.from_bytes(raw[i : i + 4], "little") for i in (0, 4, 8, 12)]
    oracle = rootrandom.Taus(1)
    oracle.s, oracle.s1, oracle.s2 = (
        registers[0],
        registers[1],
        (registers[2] + registers[3]) % 2**32,
    )
    rootrandom.draws(oracle, 6)
    assert np.array_equal(TRandom2(0).rndm(50), rootrandom.draws(oracle, 50))
    again = rootrandom.Taus(1)
    again.s, again.s1, again.s2 = registers[0], registers[1], (registers[2] + registers[3]) % 2**32
    mt = rootrandom.MT(1)
    rootrandom.draws(again, 6)
    mt.mt = (rootrandom.draws(again, 624) * 2**32).astype(np.int64).tolist()
    rootrandom.draws(mt, 10)
    assert np.array_equal(TRandom3(0).rndm(700), rootrandom.draws(mt, 700))
    assert np.array_equal(TRandom3(0).rndm(10), TRandom3(0).rndm(10))


def test_a_seed_restarts_the_stream_and_is_cut_to_roots_widths():
    generator = TRandom3(1)
    generator.gaus(n=17)
    generator.set_seed(42)
    assert np.array_equal(generator.rndm(10), TRandom3(42).rndm(10))
    assert np.array_equal(TRandom3(2**32 + 5).rndm(10), TRandom3(5).rndm(10))  # UInt_t
    generator.set_seed(2**32)  # ULong_t: not zero, so fMt[0] = UInt_t(2**32) = 0
    oracle = rootrandom.MT(0)
    assert np.array_equal(generator.rndm(10), rootrandom.draws(oracle, 10))
    assert np.array_equal(TRandom1(2**32 + 7).rndm(30), TRandom1(7).rndm(30))


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_a_seed_that_is_not_an_unsigned_long_is_refused(seed):
    with pytest.raises(ValueError, match="unsigned 64-bit"):
        TRandom3(seed)


def test_a_seed_that_is_not_a_whole_number_is_refused():
    with pytest.raises(TypeError, match="whole number"):
        TRandom3(1.5)


def test_get_seed_answers_what_each_of_roots_classes_answers():
    generator = TRandom3(4357)
    assert generator.get_seed() == 4357
    generator.rndm()
    assert generator.get_seed() == generator.state["fMt"][1]
    assert TRandom(77).get_seed() == 77 and TRandom2(77).get_seed() == TRandom2(77).state["fSeed"]
    lcg, oracle = TRandom(77), rootrandom.LCG(77)
    lcg.rndm(3)
    rootrandom.draws(oracle, 3)
    assert lcg.get_seed() == oracle.seed


def test_the_default_seeds_are_roots():
    assert np.array_equal(TRandom3().rndm(5), TRandom3(4357).rndm(5))
    assert np.array_equal(TRandom2().rndm(5), TRandom2(1).rndm(5))
    assert np.array_equal(TRandom().rndm(5), TRandom(65539).rndm(5))


def test_trandom1s_default_constructor_takes_roots_seed_table_in_turn(monkeypatch):
    monkeypatch.setattr(TRandom1, "engines", 0)
    assert np.array_equal(TRandom1().rndm(30), TRandom1(9876).rndm(30))
    assert np.array_equal(TRandom1().rndm(30), TRandom1(1299961164).rndm(30))
    monkeypatch.setattr(TRandom1, "engines", 215)
    assert np.array_equal(TRandom1().rndm(30), TRandom1(9876 ^ (1 << 8)).rndm(30))


def test_grandom_is_a_trandom3_at_roots_default_seed():
    assert xrdroot.gRandom is gRandom and xrdroot.TRandom3 is TRandom3
    assert isinstance(gRandom, TRandom3)


# -- state ------------------------------------------------------------------------------


@pytest.mark.parametrize("made", [TRandom, TRandom1, TRandom2, TRandom3])
def test_a_state_taken_mid_stream_carries_on_elsewhere_exactly(made):
    generator = made(31)
    generator.gaus()  # leaves draws made ahead and held
    saved = generator.state
    other = made(1)
    other.set_state(saved)
    assert np.array_equal(generator.rndm(3000), other.rndm(3000))
    generator.gaus()
    copy = pickle.loads(pickle.dumps(generator))
    assert np.array_equal(copy.gaus(n=500), generator.gaus(n=500))


@pytest.mark.parametrize("made", [TRandom, TRandom1, TRandom2, TRandom3])
def test_draws_made_ahead_and_none_handed_out_leave_the_state_untouched(made):
    generator = made(12)
    before = generator.state
    generator._peek(50)
    after = generator.state
    assert all(np.array_equal(before[key], after[key]) for key in before)
    assert np.array_equal(generator.rndm(60), made(12).rndm(60))


def test_a_state_is_what_the_draws_handed_out_left_not_the_ones_made_ahead():
    generator = TRandom3(8)
    generator.rndm()
    oracle = rootrandom.MT(8)
    oracle.rndm()
    state = generator.state
    assert state["fCount624"] == oracle.count and state["fMt"].tolist() == oracle.mt


@pytest.mark.parametrize(
    ("made", "change", "message"),
    [
        (TRandom3, {"fMt": np.zeros(5)}, "624 unsigned"),
        (TRandom3, {"fCount624": 625}, "below 625"),
        (TRandom, {"fSeed": -1}, "at least 0"),
        (TRandom2, {"fSeed1": 2**32}, "below"),
        (TRandom1, {"fJlag": 3}, "ten places"),
        (TRandom1, {"fCarry": 0.5}, "fCarry"),
        (TRandom1, {"fFloatSeedTable": np.full(24, 0.3)}, "multiples"),
    ],
)
def test_a_state_root_could_not_be_in_is_refused_by_name(made, change, message):
    state = {**made(3).state, **change}
    with pytest.raises(ValueError, match=message):
        made(3).set_state(state)


def test_a_state_missing_a_member_is_refused_by_name():
    with pytest.raises(ValueError, match="needs fCount624"):
        TRandom3().set_state({"fMt": np.zeros(624, dtype=np.uint32)})


# -- the C library's functions ----------------------------------------------------------


def test_numpys_functions_are_used_only_where_they_are_the_c_librarys():
    import math

    # The same numbers as the C library by construction, so kept, on any machine;
    # whether NumPy's own log is is the machine's business (not on AVX-512 Linux).
    same = np.vectorize(math.log)
    assert libm.choose(same, math.log, 0.1, 1.0) is same
    exact = libm.choose(lambda x: np.nextafter(np.log(x), 0), math.log, 0.1, 1.0)
    values = np.array([[0.5, 0.25], [0.125, 0.3]])
    assert np.array_equal(exact(values), [[math.log(x) for x in row] for row in values.tolist()])
