"""References ROOT printed on one machine: to the bit there, to a stated tolerance elsewhere.

ROOT 6.40.04 printed the RooFit references on macOS x86-64, so its
``std::exp``, ``std::cos``, ``std::erfc`` were Apple's x86-64 libm. xrdroot
calls the same C library functions ROOT calls (:mod:`xrdroot.random.libm`),
so on such a machine its numbers are ROOT's to the last bit, and the tests
say so. Other C libraries round some values differently in the last place -
Apple's x86-64 ``cos(0.8999999999999999)`` is ``0.6216099682706646``, where
glibc's, correctly rounded, is ``...645`` - and ROOT itself prints those
other bits there. Minuit2 is compiled code too: built for arm64, its
compiler fuses multiplies and adds that x86-64 rounds apart, so iminuit's
MIGRAD takes other steps from the same likelihood. So off that machine a
reference is held to a tolerance, each justified where it is used by what
lies between the library call and the number: a sum that cancels, an
adaptive rule that cuts a region more or fewer, a minimiser that stops a
step earlier or later.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any

import pytest

__all__ = ["ROOTS_MACHINE", "roots"]

#: Whether this is the kind of machine the references were printed on - which
#: ``XRDROOT_TEST_OTHER_MACHINE=1`` denies, to try the tolerances where it is.
ROOTS_MACHINE = (
    sys.platform == "darwin"
    and platform.machine() == "x86_64"
    and not os.environ.get("XRDROOT_TEST_OTHER_MACHINE")
)


def roots(expected: Any, **tolerance: float) -> Any:
    """``expected`` itself on ROOT's machine, else ``pytest.approx(expected, **tolerance)``."""
    return expected if ROOTS_MACHINE else pytest.approx(expected, **tolerance)
