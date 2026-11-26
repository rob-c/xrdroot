"""``<unistd.h>``: the POSIX calls ROOT's own tests make, and the typedefs they declare.

A C++ test includes ``<unistd.h>`` and calls it, so the translation's runtime holds them:
``getpid``, ``usleep``, ``fork``/``waitpid``, ``mkstemp`` - and ``pid_t`` is a type the
translator knows, which it was not.
"""

from __future__ import annotations

import os

import pytest

from xrdroot.cppstd import unistd
from xrdroot.pycxx.runtime import Cell


def test_getpid_is_this_process_id() -> None:
    assert unistd.getpid() == os.getpid()


def test_usleep_sleeps_and_answers_zero() -> None:
    assert unistd.usleep(1000) == 0


def test_mkstemp_makes_the_template_unique_and_gives_its_descriptor(tmp_path) -> None:
    template = str(tmp_path / "made-XXXXXX")
    name = unistd.mkstemp_name(template)
    assert name.startswith(str(tmp_path / "made-")) and "XXXXXX" not in name
    made = unistd.mkstemp(str(tmp_path / "later-XXXXXX"))
    assert made >= 0
    os.close(made)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX's")
def test_a_forked_child_is_waited_for_and_its_status_written_through() -> None:
    pid = unistd.fork()
    if pid == 0:  # the child
        os._exit(0)
    status = Cell(0, "int")
    assert unistd.waitpid(pid, status, 0) == pid
    assert status.value == 0  # exit(0)


def test_pid_t_is_the_int_the_translator_now_knows(capsys: pytest.CaptureFixture[str]) -> None:
    from cintfake import fake
    from xrdroot.pycint.execute import run_source

    run_source('void t() { pid_t p = 41; printf("%d\\n", int(p) + 1); }', "t.C", root=fake())
    assert capsys.readouterr().out == "42\n"
