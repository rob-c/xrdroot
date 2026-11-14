"""C's ``<cstdio>`` files, and ``<cstdlib>``'s ``rand``, as a translation calls them, from
:mod:`cppstd`."""

from __future__ import annotations

from ...cppstd.cstdio import *  # noqa: F403
from ...cppstd.cstdio import __all__ as _files
from ...cppstd.stdlib import RAND_MAX as RAND_MAX
from ...cppstd.stdlib import rand as rand
from ...cppstd.stdlib import srand as srand

__all__ = [*_files, "rand", "srand", "RAND_MAX"]
