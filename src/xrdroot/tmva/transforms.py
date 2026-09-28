"""TMVA's variable transformations: ``I``, ``N``, ``D``, ``P``, ``G`` and ``U``, and chains of them.

A method's ``VarTransform=N`` - or the Factory's ``Transformations=I;D;P;G,D``
- names transformations applied to the variables before a method sees them:
the identity, a normalisation onto [-1, 1], decorrelation by the inverse
square root of the covariance matrix, the principal components, and the
Gaussianisation (or, ``U``, the flattening) of each variable through its
cumulative distribution. Each is computed from the training events of every
class together, unless ``_Signal`` or another class is named, and a chain
``G,D`` applies them in order - each prepared on what the one before made.
The arithmetic is TMVA's, down to its single precision: an event's values
are ``Float_t`` after every step.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import eigen
from .dataset import Events, as_float

__all__ = [
    "Decorrelate",
    "Identity",
    "Normalize",
    "PCA",
    "Transform",
    "make_transform",
]


def _weighted_covariance(values: Any, weights: Any) -> Any:
    """``Tools::CalcCovarianceMatrices``: the weighted covariance, divided by the sum of weights."""
    total = float(np.sum(weights))
    first = weights @ values
    second = (values * weights[:, None]).T @ values
    return second / total - np.outer(first, first) / (total * total)


class Transform:
    """One transformation: prepared once from training events, then applied to any."""

    #: TMVA's letter and its name for the transformation, as directories are named.
    letter = "I"
    name = "Id"
    #: Does it transform the targets too, as ``N`` and ``U`` do by default?
    targets = False

    def __init__(self, reference: int = -1) -> None:
        #: The class whose events it is computed from, ``-1`` for all of them.
        self.reference = reference

    def prepare(self, events: Events) -> None:
        """Work out what the transformation needs from ``events``."""

    def apply(self, values: Any) -> Any:
        """``values`` - a row per event - transformed."""
        return values

    def inverse(self, values: Any) -> Any:
        """The transformation undone, for the targets a regression gives back."""
        return values

    def _chosen(self, events: Events) -> Events:
        return events if self.reference < 0 else events.of_class(self.reference)

    def announce(self) -> str | None:
        """What TMVA says when it starts preparing this transformation, if anything."""
        return None


class Identity(Transform):
    """``VariableIdentityTransform``."""


class Normalize(Transform):
    """``VariableNormalizeTransform``: each value mapped from [min, max] onto [-1, 1]."""

    letter, name, targets = "N", "Norm", True

    def prepare(self, events: Events) -> None:
        chosen = self._chosen(events)
        values = np.asarray(chosen.values, dtype=np.float32)
        self.minimum = values.min(axis=0)
        self.maximum = values.max(axis=0)
        if chosen.targets.shape[1]:
            targets = np.asarray(chosen.targets, dtype=np.float32)
            self.target_min, self.target_max = targets.min(axis=0), targets.max(axis=0)

    def _map(self, values: Any, low: Any, high: Any) -> Any:
        values = np.asarray(values, dtype=np.float32)
        scale = (1.0 / (high - low).astype(np.float64)).astype(np.float32)
        return ((values - low) * scale * np.float32(2) - np.float32(1)).astype(np.float64)

    def apply(self, values: Any) -> Any:
        return self._map(values, self.minimum, self.maximum)

    def apply_targets(self, targets: Any) -> Any:
        return self._map(targets, self.target_min, self.target_max)

    def inverse(self, targets: Any) -> Any:
        low, high = self.target_min.astype(np.float64), self.target_max.astype(np.float64)
        return as_float((np.asarray(targets) + 1.0) * (high - low) / 2.0 + low)


class Decorrelate(Transform):
    """``VariableDecorrTransform``: the values times the inverse square root of their covariance."""

    letter, name = "D", "Deco"

    def announce(self) -> str:
        return "Preparing the Decorrelation transformation..."

    def prepare(self, events: Events) -> None:
        chosen = self._chosen(events)
        covariance = _weighted_covariance(chosen.values, chosen.weights)
        self.matrix = eigen.inverse_square_root(covariance)

    def apply(self, values: Any) -> Any:
        return as_float(np.asarray(values) @ self.matrix.T)


class PCA(Transform):
    """``VariablePCATransform``: the values on ``TPrincipal``'s principal components."""

    letter, name = "P", "PCA"

    def announce(self) -> str:
        return "Preparing the Principle Component (PCA) transformation..."

    def prepare(self, events: Events) -> None:
        values = self._chosen(events).values
        self.mean = values.mean(axis=0)
        centred = values - self.mean
        covariance = centred.T @ centred / len(values)
        _, self.vectors = eigen.symmetric(covariance)

    def apply(self, values: Any) -> Any:
        return as_float((np.asarray(values) - self.mean) @ self.vectors)


def make_transform(letter: str, reference: int = -1) -> Transform:
    """The transformation TMVA names by ``letter`` - ``N``, ``Norm``, ``Decorrelate``..."""
    from .gauss import Gauss, Uniform

    kinds: dict[str, type[Transform]] = {
        "I": Identity,
        "Ident": Identity,
        "Identity": Identity,
        "N": Normalize,
        "Norm": Normalize,
        "Normalise": Normalize,
        "Normalize": Normalize,
        "D": Decorrelate,
        "Deco": Decorrelate,
        "Decorrelate": Decorrelate,
        "P": PCA,
        "PCA": PCA,
        "G": Gauss,
        "Gauss": Gauss,
        "U": Uniform,
        "Uniform": Uniform,
    }
    kind = kinds.get(letter)
    if kind is None:
        from .log import Logger

        raise Logger("Factory").fatal(f"<ProcessOptions> Variable transform '{letter}' unknown.")
    return kind(reference)
