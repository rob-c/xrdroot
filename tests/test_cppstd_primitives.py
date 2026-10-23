"""Compatibility paths preserve arithmetic and address type identity."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize("name", ["arith", "cells"])
def test_cppstd_primitives_share_the_interpreters_public_objects(name: str) -> None:
    generic = importlib.import_module(f"xrdroot.cppstd.{name}")
    legacy = importlib.import_module(f"xrdroot.cint.runtime.{name}")
    assert legacy.__all__ is generic.__all__
    for exported in generic.__all__:
        assert getattr(legacy, exported) is getattr(generic, exported), exported


def test_a_cell_made_at_either_path_is_the_same_address_type() -> None:
    from xrdroot.cint.runtime.cells import Cell as LegacyCell
    from xrdroot.cppstd.cells import Cell

    held = Cell(7, "int")
    assert isinstance(held, LegacyCell)
    held[0] = 11
    assert held.value == 11
    assert held.ctype == "int"
