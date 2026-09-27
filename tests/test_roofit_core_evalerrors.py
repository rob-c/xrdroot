"""RooFit's evaluation-error log: what a failed likelihood evaluation prints, and in what order.

The layout of :func:`evalerrors.text` is ``RooAbsReal::printEvalErrors``:
each object's description, then its failures indented five spaces, cut
off after ``most`` lines with a count of the rest.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from xrdroot.roofit import evalerrors


@pytest.fixture(autouse=True)
def fresh_log() -> Iterator[None]:
    """Each test starts with an empty log that is collecting, and leaves it off."""
    evalerrors.clear()
    evalerrors.collecting(True)
    yield
    evalerrors.collecting(False)
    evalerrors.clear()


def test_nothing_is_collected_unless_a_likelihood_is_being_evaluated() -> None:
    """Outside a fit a failure is not logged: only Minuit's evaluations are reported."""
    evalerrors.collecting(False)
    assert evalerrors.active() is False
    evalerrors.record("g", lambda: "g", "bad", lambda: "x=1")
    assert evalerrors.count() == 0


def test_a_failure_recorded_no_times_is_not_logged() -> None:
    """A batch in which nothing failed records zero failures, which adds no entry."""
    evalerrors.record("g", lambda: "g", "bad", lambda: "x=1", times=0)
    assert evalerrors.count() == 0
    assert evalerrors.text(10) == ""


def test_the_description_is_made_once_and_nothing_is_collected_while_it_is_made() -> None:
    """Describing an object evaluates it, which must not log a second failure."""
    made: list[bool] = []

    def origin() -> str:
        made.append(evalerrors.active())
        evalerrors.record("inner", lambda: "inner", "nested", lambda: "")
        return "RooGaussian::g[ x=x ]"

    evalerrors.record("g", origin, "p.d.f value is Not-a-Number", lambda: "x=1", times=2)
    evalerrors.record("g", origin, "p.d.f value is Not-a-Number", lambda: "x=2")
    assert made == [False]
    assert evalerrors.active() is True
    assert evalerrors.count() == 3
    assert evalerrors.text(10) == (
        "RooGaussian::g[ x=x ]\n"
        "     p.d.f value is Not-a-Number @ x=1\n"
        "     p.d.f value is Not-a-Number @ x=1\n"
        "     p.d.f value is Not-a-Number @ x=2\n"
    )


def test_the_likelihoods_own_density_is_listed_first() -> None:
    """RooFit lists the top-level density before the components that failed before it."""
    evalerrors.record("a", lambda: "a", "first", lambda: "")
    evalerrors.record("top", lambda: "top", "second", lambda: "", top=True)
    evalerrors.record("b", lambda: "b", "third", lambda: "")
    assert evalerrors.text(0) == "top has 1 errors\na has 1 errors\nb has 1 errors\n"


def test_an_object_with_many_failures_is_cut_off_after_most_lines() -> None:
    """``printEvalErrors(os, 1)`` prints three lines, then counts all but one as left, as ROOT."""
    evalerrors.record("g", lambda: "g", "bad", lambda: "x", times=5)
    assert evalerrors.text(1) == (
        "g\n     bad @ x\n     bad @ x\n     bad @ x\n    ... (remaining 4 messages suppressed)\n"
    )
