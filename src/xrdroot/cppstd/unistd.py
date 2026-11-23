"""``<unistd.h>``: the POSIX calls a C++ program makes that this kit has.

ROOT's own unit tests include ``<unistd.h>`` and then call it: ``symlink("./realfile1",
"symlink_realfile1")`` in ``core/clingutils/test/TClingUtilsTests.cxx`` is how that test makes
the symlinks it asks ``GetRealPath`` about. Each answers as C does - 0 for what worked, -1 for
what did not - because the tests compare the answer with a number.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["symlink"]


def symlink(target: Any, link: Any) -> int:
    """``symlink(target, link)``: ``link`` made pointing at ``target`` - 0 when it was made, and
    -1 when it could not be, as C answers."""
    try:
        os.symlink(str(target), str(link))
    except OSError:
        return -1
    return 0
