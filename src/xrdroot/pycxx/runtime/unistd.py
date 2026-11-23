"""POSIX's ``<unistd.h>``, as a translation calls it, from :mod:`cppstd.unistd`.

A C++ program that includes ``<unistd.h>`` calls ``symlink`` in the global namespace, so the
plain runtime has it beside C's other library calls.
"""

from __future__ import annotations

from ...cppstd.unistd import *  # noqa: F403
from ...cppstd.unistd import __all__ as __all__
