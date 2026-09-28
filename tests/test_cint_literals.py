"""User-defined literals the library defines: ROOT 7's ``0.1_normal``, std::chrono's ``100us``."""

from __future__ import annotations

import datetime
import types

import pytest

from xrdroot.cint import translate
from xrdroot.cint.runtime import ROOT, user_literal


def test_a_pad_length_literal_is_the_library_operator_called_on_its_number() -> None:
    text = translate("void t() { auto p = RPadPos(0.1_normal, 20_px); auto u = .5_user; }", "t.C")
    assert "ROOT.RPadPos(user_literal('_normal', 0.1), user_literal('_px', 20))" in text
    assert "u = user_literal('_user', 0.5)" in text


def test_a_pad_length_is_made_by_roots_class_when_the_literal_is_evaluated() -> None:
    lengths = types.SimpleNamespace(
        Normal=lambda v: ("normal", v), Pixel=lambda v: ("px", v), User=lambda v: ("user", v)
    )
    with ROOT.bind(types.SimpleNamespace(RPadLength=lengths)):
        assert user_literal("_normal", 0.25) == ("normal", 0.25)
        assert user_literal("_px", 20) == ("px", 20)
        assert user_literal("_user", 80) == ("user", 80)


@pytest.mark.parametrize(
    ("suffix", "value", "duration"),
    [
        ("ns", 1500, datetime.timedelta(microseconds=1.5)),
        ("us", 100, datetime.timedelta(microseconds=100)),
        ("ms", 20, datetime.timedelta(milliseconds=20)),
        ("s", 2, datetime.timedelta(seconds=2)),
        ("min", 3, datetime.timedelta(minutes=3)),
        ("h", 1, datetime.timedelta(hours=1)),
    ],
)
def test_a_chrono_literal_is_a_duration(
    suffix: str, value: int, duration: datetime.timedelta
) -> None:
    assert user_literal(suffix, value) == duration


def test_a_chrono_literal_is_translated_as_its_duration() -> None:
    text = translate("void t() { auto d = 100us; auto e = 1.5s; }", "t.C")
    assert "d = user_literal('us', 100)" in text and "e = user_literal('s', 1.5)" in text
