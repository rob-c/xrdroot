"""The functions a histogram or graph carries: its fits, in ``fFunctions``.

ROOT hangs a fit on what was fitted - ``TH1::Fit`` adds the function to the
histogram's list of functions, and it is written and read back with it.
The list is the members' own, so a function added to it is one written
with the histogram.
"""

from __future__ import annotations

from typing import Any

__all__ = ["listed"]


def listed(core: dict[str, Any]) -> list[Any]:
    """``GetListOfFunctions``: the list ``core`` keeps its functions in, made if it had none."""
    held = core.get("fFunctions")
    if not isinstance(held, list):
        held = core["fFunctions"] = list(held or ())
    return held
