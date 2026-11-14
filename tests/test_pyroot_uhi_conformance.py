"""UHI's own indexing suite, ``uhi.testing.indexing``, run on ROOT's histograms and tags."""

from __future__ import annotations

import numpy as np
import pytest

import xrdroot.pyroot as ROOT
from xrdroot.pyroot._pythonization._uhi.indexing import _temporarily_disable_add_directory

indexing = pytest.importorskip("uhi.testing.indexing")


def _same(first, second, msg=None):
    """Two histograms are the same when UHI's protocols see the same in both."""
    assert type(first) is type(second), msg
    for read in ("values", "counts", "variances"):
        np.testing.assert_array_equal(getattr(first, read)(), getattr(second, read)())
    assert first.kind == second.kind and first.axes == second.axes, msg


class _Rooted:
    tag = ROOT.uhi
    cls: type

    def setUp(self):
        self.addTypeEqualityFunc(self.cls, _same)

    @classmethod
    def make_histogram(cls):
        with _temporarily_disable_add_directory():
            return cls.cls(cls.get_uhi())


class TestIndexing1D(_Rooted, indexing.Indexing1D):
    cls = ROOT.TH1D


class TestIndexing2D(_Rooted, indexing.Indexing2D):
    cls = ROOT.TH2D


class TestIndexing3D(_Rooted, indexing.Indexing3D):
    cls = ROOT.TH3D
