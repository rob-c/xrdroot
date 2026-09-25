"""``TRandom3``: the Mersenne Twister, as ROOT runs it, and ROOT's ``gRandom``.

ROOT's ``TRandom3`` is MT19937 exactly as Matsumoto and Nishimura published
it: 624 words of state twisted all at once every 624 draws, each word
tempered on the way out, and a non-zero seed spread over the state by the
reference ``init_genrand``. ``Rndm()`` is the tempered word over 2**32, drawn
again if it came out zero, so that the result is never 0 and never 1.

NumPy carries the same generator, word for word, as
:class:`numpy.random.MT19937`, and its state is ROOT's state: the 624 words
after the last twist and the position of the next one to temper, which is
``fMt`` and ``fCount624``. So the words here are made by NumPy's compiled
twist, from a state set to ROOT's and read back as ROOT's, and only the
seeding - which NumPy does its own way - and the zero rule are ROOT's code
written out. Ten million draws take a few tenths of a second.

A zero seed is ROOT's request for an unrepeatable stream: ROOT fills the
state from a ``TRandom2`` seeded from a UUID and throws the first ten draws
away, and so does this.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .tausworthe import TRandom2
from .trandom import UINT, TRandom, _member, nonzero

__all__ = ["TRandom3", "init_genrand"]

#: ``TRandom3::Rndm``'s scale: 1/2**32, written as ROOT writes it.
SCALE = 2.3283064365386963e-10
#: The words of state.
N = 624


def init_genrand(seed: int) -> Any:
    """The state ``TRandom3::SetSeed`` makes of a non-zero seed, by Knuth's multiplier."""
    words = [seed % UINT]
    for i in range(1, N):
        last = words[-1]
        words.append((1812433253 * (last ^ (last >> 30)) + i) % UINT)
    return np.array(words, dtype=np.uint32)


class TRandom3(TRandom):
    """ROOT's ``TRandom3``: the Mersenne Twister, 2**19937 - 1 draws before it repeats.

    It is ROOT's recommended generator and the one ``gRandom`` is, at ROOT's
    default seed of 4357. ``get_seed`` answers what ROOT's does, which is the
    next word of state rather than the seed: 4357 only until the first draw.
    """

    #: The seed ROOT's constructor, and so ``gRandom``, starts from.
    DEFAULT_SEED = 4357

    def __init__(self, seed: int = DEFAULT_SEED) -> None:
        super().__init__(seed)

    def _engine_init(self) -> None:
        self._twister = np.random.MT19937(0)

    def _engine_seed(self, seed: int) -> None:
        if seed:
            self._load(init_genrand(seed), N)
            return
        words = TRandom2(0).rndm(N) * 4294967296.0
        self._load(words.astype(np.uint32), N)
        self._engine_fill(10)

    def _engine_fill(self, count: int) -> Any:
        return nonzero(self._twister.random_raw, count).astype(np.float64) * SCALE

    def _engine_export(self) -> dict[str, Any]:
        inner = self._twister.state["state"]
        return {"fMt": np.array(inner["key"], dtype=np.uint32), "fCount624": int(inner["pos"])}

    def _engine_import(self, state: dict[str, Any]) -> None:
        words = np.asarray(state.get("fMt", ()))
        if words.shape != (N,) or not np.all((words >= 0) & (words < UINT)):
            raise ValueError("A TRandom3 state's fMt is 624 unsigned 32-bit words.")
        self._load(words.astype(np.uint32), _member(state, "fCount624", 0, N + 1))

    def _engine_get_seed(self) -> int:
        state = self._engine_export()
        return int(state["fMt"][state["fCount624"] % N])

    def _load(self, words: Any, position: int) -> None:
        """Set the twister to ROOT's ``fMt`` and ``fCount624``."""
        state = {"key": words, "pos": position}
        self._twister.state = {"bit_generator": "MT19937", "state": state}
