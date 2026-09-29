"""MathCore's BrentRootFinder: a grid to bracket the root, then Brent's method on ``|f|``."""

from __future__ import annotations

from typing import Any

from xrdroot.numerics.rootfinder import brent_root_finder


def test_a_root_is_bracketed_on_the_grid_and_polished() -> None:
    found, root = brent_root_finder(lambda x: x * x - 2.0, 0.0, 3.0)
    assert found and abs(root - 2.0**0.5) < 1e-8


def test_a_root_at_a_grid_point_is_taken_as_it_is() -> None:
    assert brent_root_finder(lambda x: x - 1.0, 1.0, 3.0) == (True, 1.0)
    assert brent_root_finder(lambda x: x - 2.0, 0.0, 4.0, npx=5) == (True, 2.0)


def test_no_root_on_the_grid_is_said_as_mathcore_says(capfd: Any) -> None:
    assert brent_root_finder(lambda x: x * x + 1.0, 0.0, 1.0) == (False, 0.0)
    err = capfd.readouterr().err
    assert "Info in <ROOT::Math::BrentMethods::MinimStep>: xmin = 0 xmax = 1 npts = 100" in err
    assert "Error in <ROOT::Math::BrentRootFinder>: Interval does not contain a root" in err


def test_a_search_that_never_converges_gives_up_after_eleven(capfd: Any) -> None:
    found, _ = brent_root_finder(lambda x: x - 0.123456789, 0.0, 1.0, max_iter=0,
                                 abs_tol=0.0, rel_tol=0.0)  # fmt: skip
    assert not found
    assert "Error in <ROOT::Math::BrentRootFinder::Solve>: Search didn't converge" in (
        capfd.readouterr().err)  # fmt: skip
