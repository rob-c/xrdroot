"""The live ROOT namespace supplies RooFit's names for functions bound by C++ macros."""

from __future__ import annotations

from typing import Any

from ...roofit.bindings import REGISTERED


def registered_name(func: Any) -> tuple[str, tuple[str, ...]] | None:
    """Keep name lookup live so a replaced namespace member has the same binding behavior."""
    import xrdroot.pyroot as ROOT

    for (space, name), arguments in REGISTERED.items():
        if getattr(getattr(ROOT, space), name, None) is func:
            return f"{'ROOT::Math' if space == 'Math' else space}::{name}", arguments
    return None
