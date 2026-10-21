"""Fresh ROOT 6.40.04 pools distinguish disabled from an enabled pool of one."""

import sys
from types import SimpleNamespace

import pytest

from xrdroot import RDataFrame
from xrdroot.rdf import parallel
from xrdroot.rdf.frame import (
    DisableImplicitMT,
    EnableImplicitMT,
    GetThreadPoolSize,
    IsImplicitMTEnabled,
)


@pytest.fixture(autouse=True)
def configured_machine(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 12)
    monkeypatch.delenv("ROOT_MAX_THREADS", raising=False)
    DisableImplicitMT()
    yield
    DisableImplicitMT()


def test_a_disabled_pool_and_an_enabled_pool_of_one_have_distinct_states():
    assert (IsImplicitMTEnabled(), GetThreadPoolSize(), RDataFrame(1).GetNSlots()) == (False, 0, 1)
    EnableImplicitMT(1)
    assert (IsImplicitMTEnabled(), GetThreadPoolSize(), RDataFrame(1).GetNSlots()) == (True, 1, 1)
    DisableImplicitMT()
    assert (IsImplicitMTEnabled(), GetThreadPoolSize(), RDataFrame(1).GetNSlots()) == (False, 0, 1)


@pytest.mark.parametrize("requested,expected", [(0, 4), (8, 4), (2, 2), (64, 4), (-1, 4)])
def test_the_configured_cap_bounds_default_and_explicit_requests(monkeypatch, requested, expected):
    monkeypatch.setenv("ROOT_MAX_THREADS", "4")
    EnableImplicitMT(requested)
    assert (GetThreadPoolSize(), RDataFrame(1).GetNSlots()) == (expected, expected)
    assert RDataFrame(1, workers=6).GetNSlots() == 6


@pytest.mark.parametrize("requested,expected", [(0, 12), (8, 8), (2, 2), (64, 12), (-1, 12)])
def test_hardware_bounds_the_request_when_no_root_cap_is_set(requested, expected):
    EnableImplicitMT(requested)
    assert GetThreadPoolSize() == expected


@pytest.mark.parametrize(
    "value,expected",
    [("4junk", 4), ("0x4", 4), ("010", 8), ("3.5", 3), ("  +3 trailing", 3), ("1", 1)],
)
def test_root_consumes_the_same_c_integer_prefix_as_its_native_arena(
    monkeypatch, capsys, value, expected
):
    monkeypatch.setenv("ROOT_MAX_THREADS", value)
    EnableImplicitMT()
    assert GetThreadPoolSize() == expected
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "value,reason",
    [
        ("", "cannot parse number"),
        ("garbage", "cannot parse number"),
        ("0", "must be >= 1, but set to 0"),
        ("-2", "must be >= 1, but set to -2"),
        ("08", "must be >= 1, but set to 0"),
    ],
)
def test_invalid_caps_are_diagnosed_and_ignored(monkeypatch, capsys, value, reason):
    monkeypatch.setenv("ROOT_MAX_THREADS", value)
    EnableImplicitMT()
    assert GetThreadPoolSize() == 12
    assert reason in capsys.readouterr().err


def test_the_first_enable_keeps_its_pool_until_disabled(monkeypatch):
    monkeypatch.setenv("ROOT_MAX_THREADS", "4")
    EnableImplicitMT(1)
    monkeypatch.setenv("ROOT_MAX_THREADS", "2")
    EnableImplicitMT(8)
    assert (IsImplicitMTEnabled(), GetThreadPoolSize()) == (True, 1)
    DisableImplicitMT()
    EnableImplicitMT(8)
    assert GetThreadPoolSize() == 2


def test_an_unreported_hardware_count_still_provides_one_worker(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: None)
    EnableImplicitMT()
    assert (IsImplicitMTEnabled(), GetThreadPoolSize()) == (True, 1)


@pytest.mark.parametrize("level,visible", [(2999, True), (3000, True), (3001, False)])
def test_cap_errors_obey_the_loaded_root_error_threshold(monkeypatch, capsys, level, visible):
    monkeypatch.setitem(sys.modules, "xrdroot.pyroot", SimpleNamespace(gErrorIgnoreLevel=level))
    monkeypatch.setenv("ROOT_MAX_THREADS", "invalid")
    EnableImplicitMT()
    assert GetThreadPoolSize() == 12
    assert bool(capsys.readouterr().err) is visible


def test_cap_errors_do_not_import_the_root_namespace(monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "xrdroot.pyroot", raising=False)
    monkeypatch.setenv("ROOT_MAX_THREADS", "invalid")
    EnableImplicitMT()
    assert "cannot parse number" in capsys.readouterr().err
    assert "xrdroot.pyroot" not in sys.modules
