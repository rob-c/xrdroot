"""What the tests share that is not a fixture."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot import Jagged


def plain(value: Any) -> Any:
    """Arrays made lists all the way down, so that nested results compare with ``==``.

    NumPy compares element by element, which is what an array is for and not
    what an assertion about a dict holding one wants; this is the same value
    with every array and every run of rows turned into the lists they hold.
    """
    if isinstance(value, (np.ndarray, Jagged)):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(plain(item) for item in value)
    return value


#: The seed ROOT's ``gRandom`` - a ``TRandom3`` - starts from unless told otherwise.
ROOT_SEED = 4357
#: What ``TRandom3::Rndm`` multiplies a 32-bit draw by: one over 2**32.
ROOT_SCALE = 2.3283064365386963e-10


def root_uniforms(count: int, seed: int = ROOT_SEED) -> np.ndarray:
    """The first ``count`` numbers ROOT's ``gRandom->Rndm()`` gives, to the bit.

    ``TRandom3`` is the Mersenne Twister seeded the way ``init_genrand``
    seeds it, which is NumPy's legacy seeding of the same generator, and a
    draw is the tempered 32-bit word over 2**32. ROOT throws a zero away and
    draws again; none of the draws the tests take is zero, which this checks
    rather than assumes. This is what lets a test fill a histogram with the
    very entries a ROOT macro filled one with, and compare the two bit for bit.
    """
    generator = np.random.MT19937()
    generator._legacy_seeding(seed)
    words = generator.random_raw(count).astype(np.uint32)
    assert np.all(words != 0)
    return words.astype(np.float64) * ROOT_SCALE


def root_rannor(count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` pairs from ``TRandom::Rannor(Float_t&, Float_t&)``, as floats.

    Box-Muller on two uniform draws, rounded to single precision the way the
    ``Float_t`` overload rounds them.
    """
    drawn = root_uniforms(2 * count).reshape(count, 2)
    angle = drawn[:, 1] * 6.28318530717958623
    radius = np.sqrt(-2 * np.log(drawn[:, 0]))
    return (radius * np.sin(angle)).astype(np.float32), (radius * np.cos(angle)).astype(np.float32)
