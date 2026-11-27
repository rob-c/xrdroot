"""Every array class names itself, the way ROOT asks of anything it writes."""

from __future__ import annotations

import pytest

import xrdroot.pyroot as ROOT
from xrdroot.errors import UnsupportedFeatureError


@pytest.mark.parametrize("name", ["TArrayC", "TArrayS", "TArrayI", "TArrayF", "TArrayD", "TArrayL"])
def test_an_array_names_itself_as_root_does(name):
    """``TObject.ClassName`` reaches for a class dictionary the arrays never fill in."""
    assert getattr(ROOT, name)().ClassName() == name


def test_a_symbolic_array_is_not_mistaken_for_one_of_roots_arrays(tmp_path):
    """The writer's array path needs a class it knows *and* values it can take."""
    import xrdroot

    with pytest.raises(UnsupportedFeatureError, match="a TNamed is not something this writer"):
        with xrdroot.create(str(tmp_path / "b.root")) as handle:
            handle["a"] = ROOT.TNamed("x", "a named thing")
