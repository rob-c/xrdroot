"""The leaves of a model read from a file: ``RooRealVar``, ``RooConstVar``, ``RooCategory``.

A variable's range is its binning's - ``RooUniformBinning`` or ``RooBinning``
- and its other named ranges and binnings are in its shared properties; its
value, error and constness are members and attributes like any other.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .build import Builder, maker, name_of, title_of
from .stream import Streamed

__all__: list[str] = []

#: ``RooNumber::infinity()``, as a range with no end is written.
INFINITY = 1.0e30


def _end(value: float) -> float:
    return float("inf") if value >= INFINITY else (float("-inf") if value <= -INFINITY else value)


def binning(record: Any) -> Any:
    """The engine's binning of a streamed one: uniform, variable, or a bare range."""
    from ..binning import RooBinning, RooUniformBinning

    if record is None:
        return None
    if record.cls == "RooUniformBinning":
        return RooUniformBinning(_end(record.get("_xlo")), _end(record.get("_xhi")),
                                 int(record.get("_nbins")), name_of(record))  # fmt: skip
    if record.cls == "RooBinning":
        edges = [float(e) for e in record.get("_array") or record.get("_boundaries") or ()]
        made = RooBinning(_end(record.get("_xlo")), _end(record.get("_xhi")), name_of(record))
        for edge in edges:
            made.addBoundary(edge)
        return made
    if record.cls == "RooRangeBinning":
        low, high = record.get("_range")
        from ..binning import RooRangeBinning

        return RooRangeBinning(_end(low), _end(high), name_of(record))
    raise UnsupportedFeatureError(f"a variable here is binned by a {record.cls}, a binning this "
                                  "RooFit engine does not read from a file")  # fmt: skip


@maker("RooRealVar")
def _real_var(builder: Builder, record: Streamed) -> Any:
    from ..variables import RooRealVar

    found = binning(record.get("_binning"))
    low, high = (found.lowBound(), found.highBound()) if found is not None else (
        float("-inf"), float("inf"))  # fmt: skip
    made = RooRealVar(name_of(record), title_of(record), float(record.get("_value")), low, high)
    if found is not None and found.numBins() and hasattr(found, "isUniform"):
        made.setBinning(found)
    made.setError(float(record.get("_error")))  # -1 for none, as RooRealVar keeps it
    made.setAsymError(float(record.get("_asymErrLo")), float(record.get("_asymErrHi")))
    shared = record.get("_sharedProp")
    for name, other in ((shared.get("_altBinning") if shared is not None else None) or {}).items():
        made.setBinning(binning(other), str(name))
    return made


@maker("RooConstVar")
def _const_var(builder: Builder, record: Streamed) -> Any:
    from ..variables import RooConstVar

    return RooConstVar(name_of(record), title_of(record), float(record.get("_value")))


@maker("RooCategory")
def _category(builder: Builder, record: Streamed) -> Any:
    from ..categories import RooCategory

    made = RooCategory(name_of(record), title_of(record))
    states = record.get("_stateNames") or {}
    for label in record.get("_insertionOrder") or sorted(states, key=lambda label: states[label]):
        made.defineType(str(label), int(states[label]))
    if states:
        made.setIndex(int(record.get("_currentIndex")))
    return made
