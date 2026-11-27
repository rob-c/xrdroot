"""An array object is written as the class ROOT knows, not as anonymous numbers."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot
import xrdroot.pyroot as ROOT
from xrdroot.errors import FormatError, UnsupportedFeatureError


@pytest.mark.parametrize(
    ("name", "values", "expected"),
    [
        ("TArrayD", [1.0, 2.5], [1.0, 2.5]),
        ("TArrayF", [1, 2, 3], [1.0, 2.0, 3.0]),
        ("TArrayI", [4, 5], [4, 5]),
        ("TArrayC", [1, 2], [1, 2]),
    ],
)
def test_a_tarray_is_a_key_that_reads_back_as_itself(tmp_path, name, values, expected):
    """The key carries the ``TArray`` record, so the class survives the file."""
    path = tmp_path / f"{name}.root"
    with xrdroot.create(str(path)) as handle:
        handle["a"] = getattr(ROOT, name)(values)
    with xrdroot.open_root(str(path)) as handle:
        read = np.asarray(handle["a"])
        assert [float(value) for value in read] == [float(v) for v in expected]


def test_every_array_class_names_itself_as_root_does():
    """ROOT asks ``ClassName()`` of anything it writes, so every array answers it."""
    for name in ("TArrayC", "TArrayS", "TArrayI", "TArrayF", "TArrayD", "TArrayL"):
        assert getattr(ROOT, name)().ClassName() == name


def test_a_plain_object_is_not_mistaken_for_an_array(tmp_path):
    """The writer's array path needs a class it knows *and* values it can take."""
    with pytest.raises((FormatError, TypeError, UnsupportedFeatureError)):
        with xrdroot.create(str(tmp_path / "b.root")) as handle:
            handle["a"] = ROOT.TNamed("x", "a named thing")
