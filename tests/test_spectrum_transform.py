"""``TSpectrumTransform`` and ``TSpectrum2Transform``: every transform, as ROOT 6.40 computes it.

The references are ROOT's own numbers, printed as hex floats by a macro run
through ROOT 6.40.04 on macOS x86-64 (``tests/data/spectrum*-transform-6.40.txt``).
On that machine each is ROOT's to the last bit. Elsewhere the C library's
``sin`` and ``cos`` may round a twiddle factor the other way in the last
place; each coefficient passes through at most eight butterflies of numbers
of order ten, so it may move by a few units in the 13th digit, and a
coefficient that cancels to nothing by as much absolutely: ``1e-12`` either
way holds it.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import numpy as np
import pytest

from refmachine import roots
from xrdroot.pyroot import TSpectrum2Transform, TSpectrumTransform

DATA = Path(__file__).parent / "data"


def _cases(name: str) -> list[tuple[str, int, int, int, list[float]]]:
    """Each line of a reference file: the operation, kind, degree, direction and ROOT's numbers."""
    found = []
    for line in (DATA / name).read_text().splitlines():
        if line.startswith("#"):
            continue
        op, kind, degree, direction, *numbers = line.split()
        values = [float.fromhex(number) for number in numbers]
        found.append((op, int(kind), int(degree), int(direction), values))
    return found


ONE = _cases("spectrum-transform-6.40.txt")
TWO = _cases("spectrum2-transform-6.40.txt")
SOURCE = np.array([((i * 7) % 11) + 0.25 * i + 1 for i in range(64)])
SOURCE2 = [[((i * 7 + j * 3) % 11) + 0.25 * i + 0.5 * j + 1 for j in range(16)] for i in range(4)]


def _close(expected: list[float]) -> object:
    return roots(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("case", ONE, ids=lambda case: " ".join(map(str, case[:4])))
def test_every_1d_transform_filter_and_enhancement_writes_roots_numbers(case):
    op, kind, degree, direction, expected = case
    t = TSpectrumTransform(16)
    t.SetTransformType(kind, degree)
    t.SetDirection(direction)
    dest = np.zeros(len(expected))
    if op == "T":
        t.Transform(SOURCE, dest)
    elif op == "Q":
        t.Transform(SOURCE, dest)
        t.Transform(SOURCE, dest)
    else:
        t.SetRegion(4, 11)
        t.SetFilterCoeff(0.5)
        t.SetEnhanceCoeff(2.0)
        (t.FilterZonal if op == "F" else t.Enhance)(SOURCE, dest)
    assert dest.tolist() == _close(expected)


@pytest.mark.parametrize("case", TWO, ids=lambda case: " ".join(map(str, case[:4])))
def test_every_2d_transform_filter_and_enhancement_writes_roots_numbers(case):
    op, kind, degree, direction, expected = case
    t = TSpectrum2Transform(4, 8)
    t.SetTransformType(kind, degree)
    t.SetDirection(direction)
    dest = np.zeros((4, len(expected) // 4))
    if op == "T":
        t.Transform(SOURCE2, dest)
    else:
        t.SetRegion(1, 2, 2, 6)
        t.SetFilterCoeff(0.5)
        t.SetEnhanceCoeff(2.0)
        (t.FilterZonal if op == "F" else t.Enhance)(SOURCE2, dest)
    assert dest.ravel().tolist() == _close(expected)


def test_a_transform_writes_into_an_array_array_or_a_list_as_through_a_pointer():
    t = TSpectrumTransform(16)
    t.SetTransformType(t.kTransformHaar, 0)
    into_array, into_list = array("d", [0.0] * 16), [0.0] * 16
    t.Transform(array("d", SOURCE[:16]), into_array)
    t.Transform(list(SOURCE[:16]), into_list)
    haar = next(case[4] for case in ONE if case[:4] == ("T", 0, 0, 0))
    assert list(into_array) == list(into_list) == _close(haar[:16])


def test_rows_of_a_2d_destination_are_filled_in_place_as_rows_of_pointers():
    t = TSpectrum2Transform(4, 8)
    t.SetTransformType(t.kTransformWalsh, 0)
    rows = [np.zeros(8) for _ in range(4)]
    t.Transform([np.array(row[:8]) for row in SOURCE2], rows)
    walsh = next(case[4] for case in TWO if case[:4] == ("T", 1, 0, 0))
    assert np.concatenate(rows).tolist() == _close(
        [v for i in range(4) for v in walsh[16 * i : 16 * i + 8]]
    )
