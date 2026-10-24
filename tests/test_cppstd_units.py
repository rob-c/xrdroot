"""Duration literals share their original units while ROOT pad lengths stay adapted."""

from __future__ import annotations

import datetime

import pytest

from cppstdsupport import isolated


@pytest.mark.parametrize(("suffix", "value", "expected"), [
    ("ns", 1500, datetime.timedelta(microseconds=2)),
    ("us", 25, datetime.timedelta(microseconds=25)),
    ("ms", -2.5, datetime.timedelta(milliseconds=-2.5)),
    ("s", 1.25, datetime.timedelta(seconds=1.25)),
    ("min", 2, datetime.timedelta(minutes=2)),
    ("h", 0.5, datetime.timedelta(minutes=30)),
])
def test_duration_literals_keep_all_units_and_timedelta_rounding(suffix, value, expected) -> None:
    from xrdroot.cint.runtime import units as legacy
    from xrdroot.cppstd import units

    assert legacy.DURATIONS is units.DURATIONS
    assert legacy.duration_literal is units.duration_literal
    assert units.duration_literal(suffix, value) == expected
    assert legacy.user_literal(suffix, value) == expected


def test_root_pad_literal_dispatch_remains_in_the_interpreter_adapter(monkeypatch) -> None:
    from types import SimpleNamespace

    from xrdroot.cint.runtime import units

    seen = []
    token = object()

    def normal(value):
        seen.append(value)
        return token

    monkeypatch.setattr(units, "ROOT", SimpleNamespace(RPadLength=SimpleNamespace(Normal=normal)))
    assert units.user_literal("_normal", 0.25) is token
    assert seen == [0.25]
    with pytest.raises(KeyError, match="_unknown"):
        units.user_literal("_unknown", 1)


def test_duration_literals_work_without_root_or_the_interpreter() -> None:
    isolated("""
import datetime
from xrdroot.cppstd.units import DURATIONS, duration_literal
assert set(DURATIONS) == {'ns', 'us', 'ms', 's', 'min', 'h'}
assert duration_literal('ms', 250) == datetime.timedelta(seconds=0.25)
assert duration_literal('ns', 1500) == datetime.timedelta(microseconds=2)
assert duration_literal('_normal', 0.25) is None
""")
