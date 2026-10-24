"""The interactive application's public namespace follows live facade exports."""

from __future__ import annotations

from typing import Any

__all__ = ["preloaded"]


def preloaded() -> dict[str, Any]:
    """The names a macro and the shell start with: all of :mod:`xrdroot`, and NumPy as ``np``."""
    import numpy

    import xrdroot

    names: dict[str, Any] = {name: getattr(xrdroot, name) for name in xrdroot.__all__}
    names["np"] = numpy
    names["xrdroot"] = xrdroot
    return names

