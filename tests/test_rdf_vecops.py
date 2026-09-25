"""``xrdroot.rdf.vecops``: ROOT::VecOps over a whole batch of collections at once.

Every function is checked against the same thing done one entry at a time in
plain Python, the way ROOT's C++ loops do it, over collections that include
empty ones - which is where a vectorised version most easily goes wrong.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from xrdroot import Jagged
from xrdroot.rdf import kernels, vecops

PT = Jagged(np.array([40.0, 12.5, 33.0, 5.0, 50.0, 7.5], np.float32), [0, 2, 2, 3, 6])
ETA = Jagged(np.array([0.5, -1.0, 2.0, 0.1, -0.3, 1.2], np.float32), [0, 2, 2, 3, 6])
PHI = Jagged(np.array([0.1, 3.0, -3.0, 1.0, -2.5, 2.9], np.float32), [0, 2, 2, 3, 6])
MASS = Jagged(np.full(6, 0.105658, np.float32), [0, 2, 2, 3, 6])


def rows(values):
    return [list(row) for row in values.tolist()]


def test_sums_products_and_means_are_one_number_per_entry():
    assert vecops.Sum(PT).tolist() == [52.5, 0.0, 33.0, 62.5]
    assert vecops.Sum(PT).dtype == np.float32  # C++ sums an RVec<float> in float
    assert vecops.Sum(PT, 0.0).dtype == np.float64  # Sum(v, 0.) sums in double
    assert vecops.Sum(Jagged(np.array([True, True, False]), [0, 2, 3])).tolist() == [2, 0]
    assert vecops.Product(PT).tolist() == [500.0, 1.0, 33.0, 1875.0]
    assert vecops.Mean(PT).tolist() == [26.25, 0.0, 33.0, 62.5 / 3]
    assert vecops.Size(PT).tolist() == [2, 0, 1, 3]


def test_the_spread_of_a_collection_is_over_n_minus_one_and_zero_below_two():
    wanted = [np.var([40.0, 12.5], ddof=1), 0.0, 0.0, np.var([5.0, 50.0, 7.5], ddof=1)]
    assert np.allclose(vecops.Var(PT), wanted)
    assert np.allclose(vecops.StdDev(PT), np.sqrt(wanted))


def test_the_extremes_of_an_empty_collection_are_the_default_asked_for():
    assert vecops.Max(PT).tolist()[::2] == [40.0, 33.0]
    assert math.isnan(vecops.Max(PT)[1])
    assert vecops.Min(PT, default=-1).tolist() == [12.5, -1.0, 33.0, 5.0]
    full = Jagged(np.array([3, 1, 2]), [0, 2, 3])
    assert vecops.Max(full).dtype == full.content.dtype  # nothing empty: nothing widened
    assert vecops.ArgMax(PT).tolist() == [0, 0, 0, 1]
    assert vecops.ArgMin(PT).tolist() == [1, 0, 0, 0]


def test_any_and_all_are_the_logic_of_empty_collections_too():
    cut = Jagged(np.array([1, 0, 0, 1, 1, 1]), [0, 2, 2, 3, 6])
    assert vecops.Any(cut).tolist() == [True, False, False, True]
    assert vecops.All(cut).tolist() == [False, True, False, True]


def test_dot_pairs_two_collections_and_refuses_numbers():
    assert vecops.Dot(PT, PT).tolist() == vecops.Sum(vecops.Map(np.square, PT)).tolist()
    with pytest.raises(TypeError, match="two collections"):
        vecops.Dot(np.ones(3), np.ones(3))


def test_take_picks_positions_counts_from_either_end_and_pads_on_request():
    order = vecops.Argsort(PT)
    assert rows(vecops.Take(PT, order)) == [sorted(row) for row in rows(PT)]
    assert rows(vecops.Take(PT, 1, default=-1)) == [[40.0], [-1.0], [33.0], [5.0]]
    assert rows(vecops.Take(PT, -2, default=0)) == [[40.0, 12.5], [0, 0], [0, 33.0], [50.0, 7.5]]
    assert rows(vecops.Take(PT, np.array([1, 0, 0, 3]), default=9)) == [
        [40.0],
        [],
        [],
        [5.0, 50.0, 7.5],
    ]


def test_take_refuses_to_run_off_the_end_without_a_default():
    with pytest.raises(IndexError, match="row 1"):
        vecops.Take(PT, 1)
    with pytest.raises(IndexError, match="past the end"):
        vecops.Take(PT, Jagged(np.array([5]), [0, 1, 1, 1, 1]))


def test_nonzero_where_and_drop_are_element_by_element():
    cut = Jagged(np.array([0, 2, 0, 1, 1, 0]), [0, 2, 2, 3, 6])
    assert rows(vecops.Nonzero(cut)) == [[1], [], [], [0, 1]]
    assert rows(vecops.Where(cut, PT, -1.0)) == [[-1.0, 12.5], [], [-1.0], [5.0, 50.0, -1.0]]
    assert vecops.Where(np.array([1, 0]), np.array([1, 2]), 7).tolist() == [1, 7]
    assert rows(vecops.Drop(PT, Jagged(np.array([0, 9, 1]), [0, 1, 1, 2, 3]))) == [
        [12.5],
        [],
        [33.0],
        [5.0, 7.5],
    ]


def test_sorting_and_reversing_keep_each_collection_to_itself():
    assert rows(vecops.Sort(PT)) == [sorted(row) for row in rows(PT)]
    assert rows(vecops.Reverse(PT)) == [row[::-1] for row in rows(PT)]
    ties = Jagged(np.array([2, 1, 2, 1]), [0, 4])
    assert rows(vecops.StableArgsort(ties)) == [[1, 3, 0, 2]]


def test_concatenate_enumerate_and_range_build_new_collections():
    assert rows(vecops.Concatenate(PT, ETA))[3] == rows(PT)[3] + rows(ETA)[3]
    assert vecops.Concatenate(PT, ETA).content.dtype == np.float32
    assert rows(vecops.Enumerate(PT)) == [[0, 1], [], [0], [0, 1, 2]]
    assert rows(vecops.Range(np.array([2, 0, 3]))) == [[0, 1], [], [0, 1, 2]]
    assert rows(vecops.Range(1, np.array([4, 1]))) == [[1, 2, 3], []]
    assert rows(vecops.Range(5, 0, -2)) == [[5, 3, 1]]


def test_combinations_are_every_choice_in_order_or_every_pairing():
    first, second = vecops.Combinations(PT, 2)
    pairs = [list(zip(a, b)) for a, b in zip(rows(first), rows(second))]
    assert pairs == [list(itertools.combinations(range(len(row)), 2)) for row in rows(PT)]
    i, j = vecops.Combinations(PT, ETA)
    assert list(zip(rows(i)[3], rows(j)[3])) == list(itertools.product(range(3), range(3)))
    assert vecops.Combinations(Jagged(np.zeros(0), [0, 0]), 2)[0].tolist() == [[]]


def _delta_phi(a: float, b: float) -> float:
    turn = math.fmod(b - a, 2 * math.pi)
    if turn < -math.pi:
        return turn + 2 * math.pi
    return turn - 2 * math.pi if turn > math.pi else turn


def test_delta_phi_folds_the_difference_the_way_root_folds_it():
    other = Jagged(np.array([-3.0, 3.0, 3.0, -1.0, 2.5, -2.9], np.float32), [0, 2, 2, 3, 6])
    got = vecops.DeltaPhi(PHI, other)
    assert got.content.dtype == np.float32
    wanted = [_delta_phi(a, b) for a, b in zip(PHI.content.tolist(), other.content.tolist())]
    assert np.allclose(got.content, wanted)
    assert np.allclose(vecops.DeltaPhi(np.array([3.0]), np.array([-3.0])), [2 * math.pi - 6.0])
    assert vecops.DeltaPhi(np.array([0]), np.array([1])).dtype == np.float64


def test_delta_r_is_the_distance_in_eta_and_phi():
    squared = vecops.DeltaR2(ETA, 0.0, PHI, 0.0)
    wanted = [
        (e * e + _delta_phi(p, 0.0) ** 2)
        for e, p in zip(ETA.content.tolist(), PHI.content.tolist())
    ]
    assert np.allclose(squared.content, wanted)
    assert np.allclose(vecops.DeltaR(ETA, 0.0, PHI, 0.0).content, np.sqrt(wanted))
    assert np.allclose(vecops.DeltaR(np.array([1.0]), 0.0, np.array([0.0]), 0.0), [1.0])


def _four(pt: float, eta: float, phi: float, mass: float) -> tuple[float, ...]:
    x, y, z = pt * math.cos(phi), pt * math.sin(phi), pt * math.sinh(eta)
    return x, y, z, math.sqrt(x * x + y * y + z * z + mass * mass)


def _mass(vectors: list[tuple[float, ...]]) -> float:
    x, y, z, e = (sum(each[i] for each in vectors) for i in range(4))
    return math.sqrt(max(e * e - x * x - y * y - z * z, 0.0))


def test_invariant_mass_adds_up_every_particle_of_an_entry():
    got = vecops.InvariantMass(PT, ETA, PHI, MASS)
    assert got.dtype == np.float32
    columns = [rows(each) for each in (PT, ETA, PHI, MASS)]
    wanted = [_mass([_four(*p) for p in zip(*entry)]) for entry in zip(*columns)]
    assert np.allclose(got, wanted, rtol=1e-5, atol=1e-2)  # one muon: float cancels
    with pytest.raises(TypeError, match="collection of particles"):
        vecops.InvariantMass(np.ones(2), np.ones(2), np.ones(2), np.ones(2))


def test_invariant_masses_pair_each_particle_with_its_partner():
    got = vecops.InvariantMasses(PT, ETA, PHI, MASS, PT, ETA, PHI, MASS)
    one = [_four(*p) for p in zip(PT.content, ETA.content, PHI.content, MASS.content)]
    assert np.allclose(got.content, [_mass([v, v]) for v in one], atol=1e-2)


def test_fixed_size_rows_and_slices_are_collections_too():
    square = np.arange(6.0).reshape(3, 2)
    assert vecops.Sum(square).tolist() == [1.0, 5.0, 9.0]
    sliced = PT[1:]
    assert vecops.Sum(sliced).tolist() == [0.0, 33.0, 62.5]
    assert rows(vecops.Map(np.add, sliced, np.array([1.0, 2.0, 3.0]))) == [
        [],
        [35.0],
        [8.0, 53.0, 10.5],
    ]


def test_numbers_where_collections_are_wanted_are_refused_by_name():
    with pytest.raises(TypeError, match="Sum takes a collection per entry"):
        vecops.Sum(np.ones(3))
    with pytest.raises(ValueError, match="not the same size"):
        vecops.Map(np.add, PT, ETA[:3])
    with pytest.raises(ValueError, match="not the same size"):
        vecops.Map(np.add, PT, Jagged(np.zeros(4), [0, 1, 2, 3, 4]))


def test_kernels_cope_with_nothing_at_all():
    empty = np.zeros(0, np.float32)
    offsets = np.array([0, 0], np.int64)
    picked, _, bad = kernels.taken(empty, offsets, np.array([0]), np.array([0, 1]))
    assert bad.tolist() == [True] and picked.tolist() == [0.0]
    values, _, bad = kernels.first_n(empty, offsets, 2, default=5)
    assert values.tolist() == [5.0, 5.0] and not bad.any()


def test_rows_that_start_part_way_into_their_values_and_rows_of_nothing():
    shifted = Jagged(np.arange(6.0), [2, 3, 6])
    assert vecops.Sum(shifted).tolist() == [2.0, 12.0]
    assert vecops.Sum(Jagged(np.zeros(0), [0, 0, 0])).tolist() == [0.0, 0.0]
