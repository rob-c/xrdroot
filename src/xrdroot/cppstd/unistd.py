"""``<unistd.h>``: the POSIX calls a C++ program makes that this kit has.

ROOT's own unit tests include ``<unistd.h>`` and then call it: ``symlink("./realfile1",
"symlink_realfile1")`` in ``core/clingutils/test/TClingUtilsTests.cxx`` is how that test makes
the symlinks it asks ``GetRealPath`` about. Each answers as C does - 0 for what worked, -1 for
what did not - because the tests compare the answer with a number.
"""

from __future__ import annotations

import os
import time
from typing import Any

__all__ = ["symlink", "getpid", "usleep", "fork", "waitpid", "mkstemp", "mkstemp_name"]


def symlink(target: Any, link: Any) -> int:
    """``symlink(target, link)``: ``link`` made pointing at ``target`` - 0 when it was made, and
    -1 when it could not be, as C answers."""
    try:
        os.symlink(str(target), str(link))
    except OSError:
        return -1
    return 0


def getpid() -> int:
    """``getpid()``: this process's id, as C answers."""
    return os.getpid()


def usleep(microseconds: Any) -> int:
    """``usleep(us)``: the caller asleep for that many microseconds; 0, as C answers."""
    time.sleep(int(microseconds) / 1e6)
    return 0


def fork() -> int:
    """``fork()``: the child's id in the parent, ``0`` in the child, ``-1`` when it failed.

    A child here is a real process of the operating system, as C's is: the translation that
    forked runs twice, and only the parent's side keeps the program's own bookkeeping.
    """
    pid = os.fork()
    return int(pid)


def waitpid(pid: int, status: Any, options: int = 0) -> int:
    """``waitpid(pid, &status, options)``: the id waited for, with the status written through."""
    found, told = os.waitpid(int(pid), int(options))
    status.value = told
    return int(found)


def mkstemp(template: Any) -> int:
    """``mkstemp(template)``: ``template``'s ``XXXXXX`` made unique, the open file's descriptor.

    C writes the name back into the buffer it was given, which for a translated ``char *`` is
    the ``std::string``'s own text, so a caller that passes a string sees the new name only
    through the return of this call's ``name`` - hence :func:`mkstemp_name` beside it.
    """
    return int(_mkstemp(template)[0])


def mkstemp_name(template: Any) -> str:
    """The name ``mkstemp`` made of ``template``, for a caller that needs it back."""
    return str(_mkstemp(template)[1])


def _mkstemp(template: Any) -> tuple[int, str]:
    """The descriptor and the name a ``XXXXXX`` template became."""
    import tempfile

    text = str(template)
    head, _, tail = text.partition("XXXXXX")
    directory = os.path.dirname(head) or None
    made, name = tempfile.mkstemp(prefix=os.path.basename(head), suffix=tail, dir=directory)
    return made, name
