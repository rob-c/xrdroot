"""``<unistd.h>``: the POSIX calls ROOT's own tests make, and the typedefs they declare.

A C++ test includes ``<unistd.h>`` and calls it, so the translation's runtime holds them:
``getpid``, ``usleep``, ``fork``/``waitpid``, ``mkstemp`` - and ``pid_t`` is a type the
translator knows, which it was not.
"""

from __future__ import annotations

import os

import pytest

from xrdroot.cppstd import arith, unistd
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


def test_marking_this_process_a_child_is_what_fork_does_in_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag ``fork`` sets in its child, reached here instead of there."""
    monkeypatch.setattr(unistd, "IN_CHILD", False)
    unistd.mark_child()
    assert unistd.IN_CHILD is True


def test_fork_marks_the_child_it_returns_zero_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent gets a pid and no flag; the child - a zero - is marked, without forking."""
    monkeypatch.setattr(unistd, "IN_CHILD", False)
    monkeypatch.setattr(os, "fork", lambda: 0)
    assert unistd.fork() == 0
    assert unistd.IN_CHILD is True
    monkeypatch.setattr(unistd, "IN_CHILD", False)
    monkeypatch.setattr(os, "fork", lambda: 4321)
    assert unistd.fork() == 4321 and unistd.IN_CHILD is False


def test_a_childs_exit_ends_the_process_rather_than_unwinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C's ``exit`` in a ``fork``'s child ends that process; ``SystemExit`` would unwind.

    The flag is given as an argument because coverage cannot follow a child that exits, so
    the two child-only lines are exercised here, with the ``_exit`` they take recorded.
    """
    ended: list[int] = []
    monkeypatch.setattr(os, "_exit", ended.append)
    unistd.exit_process(3, child=True)  # returns: the child is over
    assert ended == [3]


def test_a_macro_that_is_not_a_child_still_stops_with_its_status() -> None:
    with pytest.raises(SystemExit) as caught:
        arith.c_exit(7)  # not a child: the macro stops, as it always did
    assert caught.value.code == 7


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX's")
def test_a_real_forked_child_leaves_with_the_status_it_gave() -> None:
    """The whole idiom, end to end: ``fork``; the child ``exit``s; the parent waits for 3."""
    pid = unistd.fork()
    if pid == 0:  # the child
        arith.c_exit(3)
        os._exit(4)  # reached only if exit unwound instead of ending the process
    status = Cell(0, "int")
    unistd.waitpid(pid, status, 0)
    assert os.waitstatus_to_exitcode(status.value) == 3


def test_the_stdlib_exit_macros_are_their_numbers(capsys: pytest.CaptureFixture[str]) -> None:
    from cintfake import fake
    from xrdroot.pycint.execute import run_source

    run_source(
        'void t() { printf("%d %d\\n", EXIT_SUCCESS, EXIT_FAILURE); }', "t.C", root=fake()
    )
    assert capsys.readouterr().out == "0 1\n"
