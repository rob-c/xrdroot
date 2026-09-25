"""What the tests share that is not a fixture."""

from __future__ import annotations

from typing import Any

import numpy as np

from xrdroot import Jagged, TRandom3


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


def root_uniforms(count: int, seed: int = ROOT_SEED) -> np.ndarray:
    """The first ``count`` numbers ROOT's ``gRandom->Rndm()`` gives, to the bit.

    This is what lets a test fill a histogram with the very entries a ROOT
    macro filled one with, and compare the two bit for bit.
    """
    return TRandom3(seed).rndm(count)


def root_rannor(count: int) -> tuple[np.ndarray, np.ndarray]:
    """``count`` pairs from ``TRandom::Rannor(Float_t&, Float_t&)``, as floats.

    The ``Double_t`` pairs rounded to single precision, the way the
    ``Float_t`` overload rounds them.
    """
    a, b = TRandom3(ROOT_SEED).rannor(count)
    return a.astype(np.float32), b.astype(np.float32)
