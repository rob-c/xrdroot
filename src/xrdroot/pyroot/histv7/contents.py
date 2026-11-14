"""What a bin of ``RHistEngine<T>`` holds, and how it is counted into, for each ``T``.

A bin is a Python number - an integer for C++'s integral types, a float for
``double``, and a float rounded to single precision after every change for
``float``, so it holds what C++'s ``float`` would - or an
``RBinWithError``, a sum of weights and of their squares kept in place.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ...errors import UnsupportedFeatureError
from .stats import ATOMIC

__all__ = ["RBinWithError", "Kind", "kind_of", "kind_name"]


class RBinWithError:
    """``RBinWithError``: a bin's sum of weights and sum of their squares."""

    __slots__ = ("fSum", "fSum2")

    def __init__(self, fSum: float = 0.0, fSum2: float = 0.0) -> None:
        self.fSum, self.fSum2 = float(fSum), float(fSum2)

    def __float__(self) -> float:
        return self.fSum

    def __iadd__(self, other: Any) -> RBinWithError:
        """``+= w`` adds a weight and its square; ``+= other`` another bin's sums."""
        if isinstance(other, RBinWithError):
            self.fSum += other.fSum
            self.fSum2 += other.fSum2
        else:
            w = float(other)
            self.fSum += w
            self.fSum2 += w * w
        return self

    def __imul__(self, factor: Any) -> RBinWithError:
        factor = float(factor)
        self.fSum *= factor
        self.fSum2 *= factor * factor
        return self

    def AtomicInc(self) -> None:
        self.AtomicAdd(1.0)

    def AtomicAdd(self, other: Any) -> None:
        with ATOMIC:
            self += other

    def __repr__(self) -> str:
        return f"RBinWithError({self.fSum}, {self.fSum2})"


def _single(value: Any) -> float:
    """``float``: a value rounded to single precision, as a C++ ``float`` keeps it."""
    return float(np.float32(value))


def _whole(value: Any) -> int:
    """An integral type: the whole part, as C++ converts a floating-point value."""
    return int(value)


class Kind:
    """A number bin's arithmetic: every result kept as ``cast`` makes it."""

    #: Whether a bin is a mutable object, changed where it is.
    mutable = False

    def __init__(self, name: str, cast: Callable[[Any], Any]) -> None:
        self.name, self.cast = name, cast
        self.integral = cast is _whole

    def zero(self) -> Any:
        return self.cast(0)

    def convert(self, value: Any) -> Any:
        """``static_cast<T>(value)``: an ``RBinWithError`` converts as its sum does."""
        return self.cast(float(value) if isinstance(value, RBinWithError) else value)

    def inc(self, bin: Any) -> Any:
        return self.cast(bin + 1)

    def add(self, bin: Any, weight: float) -> Any:
        return self.cast(bin + weight)

    def combine(self, bin: Any, other: Any) -> Any:
        return self.cast(bin + other)

    def scale(self, bin: Any, factor: float) -> Any:
        return self.cast(bin * factor)

    def copy(self, bin: Any) -> Any:
        return bin


class ErrorKind(Kind):
    """``RBinWithError`` bins, each changed where it is."""

    mutable = True

    def __init__(self) -> None:
        super().__init__("RBinWithError", RBinWithError)
        self.integral = False

    def zero(self) -> Any:
        return RBinWithError()

    def convert(self, value: Any) -> Any:
        if isinstance(value, RBinWithError):
            return RBinWithError(value.fSum, value.fSum2)
        raise ValueError("It is not possible to convert to RBinWithError: the individual "
                         "weights are lost once a bin is filled.")  # fmt: skip

    def inc(self, bin: Any) -> Any:
        bin.fSum += 1
        bin.fSum2 += 1
        return bin

    def add(self, bin: Any, weight: float) -> Any:
        bin += weight
        return bin

    def combine(self, bin: Any, other: Any) -> Any:
        bin += other
        return bin

    def scale(self, bin: Any, factor: float) -> Any:
        bin *= factor
        return bin

    def copy(self, bin: Any) -> Any:
        return RBinWithError(bin.fSum, bin.fSum2)


#: The integral types a bin can be: each kept as a Python integer.
INTEGRAL = ("char", "signed char", "unsigned char", "short", "unsigned short", "int",
            "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
            "Char_t", "Short_t", "Int_t", "Long_t", "Long64_t")  # fmt: skip
#: Each bin content type ROOT's histograms are made with, by its C++ name.
KINDS: dict[str, Kind] = {
    **{name: Kind(name, _whole) for name in INTEGRAL},
    "float": Kind("float", _single),
    "Float_t": Kind("float", _single),
    "double": Kind("double", float),
    "Double_t": Kind("double", float),
    "RBinWithError": ErrorKind(),
}


def kind_name(given: Any) -> str:
    """A bin content type's C++ name, from the name or the class the template was given."""
    name = given.__name__ if isinstance(given, type) else str(given)
    return name.replace("ROOT::Experimental::", "").replace("Experimental.", "").strip()


def kind_of(name: str) -> Kind:
    """The arithmetic of the bins of type ``name``."""
    found = KINDS.get(name)
    if found is None:
        raise UnsupportedFeatureError(f"A ROOT 7 histogram of bins of type {name} is not "
                                      "supported: xrdroot's histograms hold the integral types, "
                                      "float, double and RBinWithError.")  # fmt: skip
    return found
