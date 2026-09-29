"""libc++'s ``std::sort``: equal keys in the order libc++ 20 leaves them, as ROCCurve sorts.

The expected orders were printed by a C++ program sorting the same keys with
``std::sort`` against libc++ 20's headers, the ones ROOT 6.40 was built with.
"""

from __future__ import annotations

import numpy as np

from xrdroot.tmva.libcxxsort import _Sorter, order


def _keys(count: int, seed: int, levels: int) -> list[float]:
    """The C program's keys: a linear congruential stream, folded into ``levels`` values."""
    state, made = seed, []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0xFFFFFFFF
        made.append(float((state >> 16) % levels))
    return made


def test_equal_keys_come_out_in_the_order_libcxx_puts_them_in():
    assert list(order(_keys(12, 1, 3))) == [1, 2, 3, 4, 9, 5, 10, 0, 6, 7, 8, 11]


def test_a_longer_range_is_partitioned_as_libcxx_partitions_it():
    wanted = [
        0,
        6,
        12,
        15,
        20,
        23,
        25,
        35,
        33,
        1,
        4,
        7,
        10,
        21,
        24,
        30,
        27,
        39,
        26,
        22,
        18,
        29,
        17,
        32,
        8,
        3,
        38,
        2,
        5,
        11,
        13,
        14,
        16,
        19,
        28,
        31,
        34,
        36,
        37,
        9,
    ]
    assert list(order(_keys(40, 2, 4))) == wanted


def test_keys_that_never_repeat_are_sorted_stably_without_the_port():
    keys = np.array([3.0, 1.0, 2.0])
    assert list(order(keys)) == [1, 2, 0]
    assert list(order(np.array([5.0]))) == [0]


def test_every_size_and_shape_of_range_ends_sorted():
    for count in (2, 3, 4, 5, 6, 30, 129, 700):
        for levels in (1, 2, 9):
            keys = np.array(_keys(count, count, levels))
            found = order(keys)
            assert sorted(found) == list(range(count))
            assert np.all(np.diff(keys[found]) >= 0)


def test_a_sort_out_of_depth_falls_back_to_a_full_sort_of_its_range():
    made = _Sorter([2.0, 1.0, 2.0, 0.0] * 10)
    made.introsort(0, 40, 0)
    assert made.k == sorted(made.k)


def test_a_nearly_sorted_range_is_finished_by_the_incomplete_insertion_sort():
    keys = [0, 0, 2, 1, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    keys += [3] * 8 + [4] * 9 + [5] * 5
    found = order(np.array(keys, dtype=float))
    assert np.all(np.diff(np.array(keys)[found]) >= 0)


def test_two_keys_out_of_order_are_swapped():
    made = _Sorter([2.0, 1.0])
    assert made.small(0, 2) and made.k == [1.0, 2.0]


def test_a_partition_that_moved_nothing_but_leaves_both_sides_unsorted_recurses():
    keys = [20.0, *range(19, 0, -1), 20.0, *range(39, 21, -1), 40.0]
    made = _Sorter(keys)
    made.introsort(0, len(keys), 10)
    assert made.k == sorted(keys)
