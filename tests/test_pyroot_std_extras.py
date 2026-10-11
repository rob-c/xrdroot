"""``std::numeric_limits``, the smart pointers, ``std::remove`` and ``emplace_back`` of a class.

The limits are C++'s on a 64-bit machine: ``numeric_limits<int>::min()`` is
-2147483648 (df036's sentinel), ``<double>::min()`` the least positive
normal double. ``emplace_back(px, py, pz, e)`` of a vector of
``ROOT::Math::XYZTVector`` makes the four-vector in place (df002).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import xrdroot.pyroot as ROOT


def test_numeric_limits_are_cpps() -> None:
    limits = ROOT.std.numeric_limits
    assert (limits["int"].min(), limits["int"].max()) == (-(2**31), 2**31 - 1)
    assert limits[int].lowest() == -(2**31) and limits["unsigned int"].max() == 2**32 - 1
    assert limits["double"].min() == np.finfo(np.float64).tiny
    assert limits[float].lowest() == -np.finfo(np.float64).max
    assert limits["float"].epsilon() == float(np.finfo(np.float32).eps)
    assert limits["Long64_t"].epsilon() == 0 and limits["double"].max() > 1e308
    assert (limits["bool"].min(), limits["bool"].max(), limits["bool"].lowest()) == (
        False,
        True,
        False,
    )
    assert limits["double"].infinity() == math.inf
    assert math.isnan(limits["double"].quiet_NaN())
    with pytest.raises(TypeError, match="arithmetic type"):
        limits["string"]


def test_a_smart_pointer_is_the_object_it_holds() -> None:
    held = ROOT.TNamed("n", "t")
    assert ROOT.std.unique_ptr["TNamed"](held) is held
    assert ROOT.std.shared_ptr(held) is held and ROOT.std.unique_ptr() is None
    assert repr(ROOT.std.unique_ptr) == "<std.unique_ptr template>"


def test_remove_deletes_a_file_as_cs_does(tmp_path: Path) -> None:
    path = tmp_path / "gone.txt"
    path.write_text("x")
    assert ROOT.std.remove(str(path)) == 0 and not path.exists()
    assert ROOT.std.remove(str(path)) == -1


def test_emplace_back_makes_a_class_of_roots_in_place() -> None:
    tracks = ROOT.std.vector["ROOT::Math::XYZTVector"]()
    tracks.emplace_back(1.0, 2.0, 2.0, 5.0)
    tracks.emplace_back(tracks[0])
    assert tracks.size() == 2 and tracks[1].E() == 5.0
    assert tracks[0].P() == 3.0
    pairs = ROOT.std.vector["std::pair<int,double>"]()
    pairs.emplace_back(1, 2.5)
    assert (pairs[0].first, pairs[0].second) == (1, 2.5)
    unknown = ROOT.std.vector["NoSuchClass"]()
    unknown.emplace_back("as given")
    assert unknown[0] == "as given"


def test_ofstream_is_the_translators_stream(tmp_path: Path) -> None:
    from xrdroot.cint.runtime.streams import ofstream

    assert ROOT.std.ofstream is ofstream
