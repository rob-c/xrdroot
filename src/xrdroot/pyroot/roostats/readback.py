"""RooStats' results read from a file, made into the engine's own objects again.

A file writes a ``HypoTestInverterResult`` as its members: the values it
scanned, the test at each of them, and the sampling distributions each test
was made from. Read member by member that is a dictionary, and every number a
macro asks the result for - the limits, the expected ones, a point's ``CLs`` -
is worked out from those members by :mod:`xrdroot.roostats`, so this puts them
where that code expects them and nothing else.

The p-values are the ones the file stored, not ones reckoned again from the
distributions: ROOT reads a result the same way, and a file written by another
build must answer with its own arithmetic rather than this session's.

The scanned parameter is the one member not carried over. Its record holds a
``RooRefArray``, which ROOT 6.21 wrote without describing it - so the reader
meets the parameter as the name of its class and no numbers - and the only
thing the result asks of it is its range, which is consulted only where the
curve has to be extrapolated past the last point scanned. A file whose limit
falls between its points, as every file a scan writes does, reads exactly.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ...roostats.hypotest import HypoTestResult
from ...roostats.inverterresult import HypoTestInverterResult
from ...roostats.sampling import SamplingDistribution
from ..core.wrapping import register_members

__all__ = ["hypotest_from", "inverter_result_from", "sampling_distribution_from"]

#: ``RooStats::HypoTestInverterResult``'s interval base, and the ``TNamed`` under it: the
#: two records a name and a title are written in.
INTERVAL = "RooStats::SimpleInterval"
CONF_INTERVAL = "RooStats::ConfInterval"


def _record(members: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """The record ``key`` holds, or an empty one for a member written as a null pointer."""
    found = members.get(key)
    return found if isinstance(found, Mapping) else {}


def _named(members: Mapping[str, Any]) -> tuple[str, str]:
    """The name and title a ``TNamed`` base carries."""
    named = _record(members, "TNamed")
    return str(named.get("fName", "")), str(named.get("fTitle", ""))


def _held(members: Any, key: str) -> list[Any]:
    """The objects or numbers a container member holds, or none for one never made."""
    found = members.get(key)
    return list(found) if found is not None else []


def _distributions(record: Any) -> Any:
    """``record`` as a sampling distribution - or ``None``, which is a pointer never made."""
    return None if not isinstance(record, Mapping) else sampling_distribution_from(record)


def sampling_distribution_from(members: Mapping[str, Any]) -> SamplingDistribution:
    """The ``SamplingDistribution`` whose members these are: its values, their weights, and
    the name of the variable the test statistic was computed for."""
    name, title = _named(members)
    made = SamplingDistribution(name, title, _held(members, "fSamplingDist"),
                                _held(members, "fSampleWeights"))  # fmt: skip
    made._var_name = str(members.get("fVarName", ""))
    return made


def hypotest_from(members: Mapping[str, Any]) -> HypoTestResult:
    """The ``HypoTestResult`` whose members these are: its two p-values and their errors, the
    data value they were counted beyond, and the distributions they came from."""
    name, title = _named(members)
    made = HypoTestResult(name, members.get("fNullPValue", math.nan),
                          members.get("fAlternatePValue", math.nan))  # fmt: skip
    made._title = title
    made._null_error = float(members.get("fNullPValueError", 0.0))
    made._alt_error = float(members.get("fAlternatePValueError", 0.0))
    made._data_value = float(members.get("fTestStatisticData", math.nan))
    made._right_tail = bool(members.get("fPValueIsRightTail", True))
    made._background_is_alt = bool(members.get("fBackgroundIsAlt", False))
    made._null = _distributions(members.get("fNullDistr"))
    made._alt = _distributions(members.get("fAltDistr"))
    return made


def inverter_result_from(members: Mapping[str, Any]) -> HypoTestInverterResult:
    """The ``HypoTestInverterResult`` whose members these are: the scan's values, the test at
    each, and the settings its limits are found with - kept as -1 for the errors, as ROOT
    keeps them, so one the file never computed is found by a fit when it is asked for."""
    interval = _record(members, INTERVAL)
    name, title = _named(_record(interval, CONF_INTERVAL))
    made = HypoTestInverterResult(name, None, 0.0)
    made._title = title
    made._cl = float(interval.get("fConfidenceLevel", 0.95))
    made._lower = float(interval.get("fLowerLimit", math.nan))
    made._upper = float(interval.get("fUpperLimit", math.nan))
    made._x = [float(value) for value in _held(members, "fXValues")]
    made._results = [hypotest_from(record) for record in _held(members, "fYObjects")]
    made._use_cls = bool(members.get("fUseCLs", False))
    made._two_sided = bool(members.get("fIsTwoSided", False))
    made._interpolate = [bool(members.get("fInterpolateLowerLimit", True)),
                         bool(members.get("fInterpolateUpperLimit", True))]  # fmt: skip
    made._fitted = [bool(members.get("fFittedLowerLimit", False)),
                    bool(members.get("fFittedUpperLimit", False))]  # fmt: skip
    made._interpolation = int(members.get("fInterpolOption", HypoTestInverterResult.kLinear))
    made._errors = [float(members.get("fLowerLimitError", -1.0)),
                    float(members.get("fUpperLimitError", -1.0))]  # fmt: skip
    made._cleanup = float(members.get("fCLsCleanupThreshold", 0.005))
    made._expected = []
    return made


register_members("RooStats::HypoTestInverterResult", inverter_result_from)
register_members("RooStats::HypoTestResult", hypotest_from)
register_members("RooStats::SamplingDistribution", sampling_distribution_from)
