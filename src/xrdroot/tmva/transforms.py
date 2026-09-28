"""TMVA's variable transformations: ``I``, ``N``, ``D``, ``P``, ``G`` and ``U``, and chains of them.

A method's ``VarTransform=N`` - or the Factory's ``Transformations=I;D;P;G,D``
- names transformations applied to the variables before a method sees them:
the identity, a normalisation onto [-1, 1], decorrelation by the inverse
square root of the covariance matrix, the principal components, and the
Gaussianisation (or, ``U``, the flattening) of each variable through its
cumulative distribution. As in TMVA, each is worked out for every class on
its own and for all of them together, and applied with whichever the caller
asks for - all of them, unless a class is named (``D_Signal``), and the
projective likelihood asks for each class in turn. The arithmetic is TMVA's,
down to its single precision: an event's values are ``Float_t`` after
every step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import eigen
from .dataset import Events, as_float

__all__ = ["Decorrelate", "Identity", "Normalize", "PCA", "Transform", "make_transform"]


def weighted_covariance(values: Any, weights: Any) -> Any:
    """``Tools::CalcCovarianceMatrices``: the weighted covariance, divided by the sum of weights."""
    total = float(np.sum(weights))
    first = weights @ values
    second = (values * weights[:, None]).T @ values
    return second / total - np.outer(first, first) / (total * total)


def samples(events: Events, nclasses: int) -> list[Events]:
    """Each class's events, then all of them together - or, with one class, just that one."""
    if nclasses <= 1:
        return [events]
    return [events.of_class(number) for number in range(nclasses)] + [events]


class Transform:
    """One transformation: worked out once from training events, then applied to any."""

    #: TMVA's letter and name for the transformation, as directories and weight files name it.
    letter = "I"
    name = "Id"
    xml_name = "Id"
    #: Does it transform the targets too, as ``N`` does by default?
    targets = False

    def __init__(self, reference: int = -1) -> None:
        #: The class whose parameters it is applied with by default, ``-1`` for all classes'.
        self.reference = reference
        self.params: list[Any] = []

    def prepare(self, events: Events, nclasses: int) -> None:
        """Work out the parameters for each class and for all of them."""
        self.params = [self.fit(sample) for sample in samples(events, nclasses)]

    def fit(self, events: Events) -> Any:
        """The parameters from one sample of events."""
        return None

    def which(self, cls: int | None) -> Any:
        """The parameters for class ``cls``: all classes' for ``-1`` or a class it has none for."""
        index = self.reference if cls is None else cls
        return self.params[index] if 0 <= index < len(self.params) else self.params[-1]

    def apply(self, values: Any, cls: int | None = None) -> Any:
        """``values`` - a row per event - transformed with class ``cls``'s parameters."""
        return values

    def announce(self) -> str | None:
        """What TMVA says when it starts preparing this transformation, if anything."""
        return None


class Identity(Transform):
    """``VariableIdentityTransform``."""

    def prepare(self, events: Events, nclasses: int) -> None:
        self.params = [None]


class Normalize(Transform):
    """``VariableNormalizeTransform``: each value mapped from [min, max] onto [-1, 1]."""

    letter, name, xml_name, targets = "N", "Norm", "Normalize", True

    def fit(self, events: Events) -> Any:
        values = np.asarray(np.column_stack([events.values, events.targets]), dtype=np.float32)
        return values.min(axis=0), values.max(axis=0)

    def _map(self, values: Any, low: Any, high: Any) -> Any:
        values = np.asarray(values, dtype=np.float32)
        scale = (1.0 / (high - low).astype(np.float64)).astype(np.float32)
        return ((values - low) * scale * np.float32(2) - np.float32(1)).astype(np.float64)

    def apply(self, values: Any, cls: int | None = None) -> Any:
        low, high = self.which(cls)
        n = np.shape(values)[1]
        return self._map(values, low[:n], high[:n])

    def apply_targets(self, targets: Any, nvar: int, cls: int | None = None) -> Any:
        low, high = self.which(cls)
        return self._map(targets, low[nvar:], high[nvar:])

    def inverse_targets(self, targets: Any, nvar: int, cls: int | None = None) -> Any:
        low, high = self.which(cls)
        low64 = low[nvar:].astype(np.float64)
        high64 = high[nvar:].astype(np.float64)
        return as_float((np.asarray(targets) + 1.0) * (high64 - low64) / 2.0 + low64)


class Decorrelate(Transform):
    """``VariableDecorrTransform``: the values times the inverse square root of their covariance."""

    letter, name, xml_name = "D", "Deco", "Decorrelation"

    def announce(self) -> str:
        return "Preparing the Decorrelation transformation..."

    def fit(self, events: Events) -> Any:
        return eigen.inverse_square_root(weighted_covariance(events.values, events.weights))

    def apply(self, values: Any, cls: int | None = None) -> Any:
        return as_float(np.asarray(values) @ self.which(cls).T)


class PCA(Transform):
    """``VariablePCATransform``: the values on ``TPrincipal``'s principal components."""

    letter, name, xml_name = "P", "PCA", "PCA"

    def announce(self) -> str:
        return "Preparing the Principle Component (PCA) transformation..."

    def fit(self, events: Events) -> Any:
        values = events.values
        mean = values.mean(axis=0)
        centred = values - mean
        _, vectors = eigen.symmetric(centred.T @ centred / len(values))
        return mean, vectors

    def apply(self, values: Any, cls: int | None = None) -> Any:
        mean, vectors = self.which(cls)
        return as_float((np.asarray(values) - mean) @ vectors)


#: Every name TMVA knows a transformation by, and which it is.
NAMES = {
    "I": "Identity",
    "Ident": "Identity",
    "Identity": "Identity",
    "N": "Normalize",
    "Norm": "Normalize",
    "Normalise": "Normalize",
    "Normalize": "Normalize",
    "D": "Decorrelate",
    "Deco": "Decorrelate",
    "Decorrelate": "Decorrelate",
    "P": "PCA",
    "PCA": "PCA",
    "G": "Gauss",
    "Gauss": "Gauss",
    "U": "Uniform",
    "Uniform": "Uniform",
}


def make_transform(letter: str, reference: int = -1) -> Transform:
    """The transformation TMVA names by ``letter`` - ``N``, ``Norm``, ``Decorrelate``..."""
    from . import gauss

    kinds: dict[str, type[Transform]] = {
        "Identity": Identity,
        "Normalize": Normalize,
        "Decorrelate": Decorrelate,
        "PCA": PCA,
        "Gauss": gauss.Gauss,
        "Uniform": gauss.Uniform,
    }
    kind = NAMES.get(letter)
    if kind is None:
        from .log import Logger

        raise Logger("Factory").fatal(f"<ProcessOptions> Variable transform '{letter}' unknown.")
    return kinds[kind](reference)
