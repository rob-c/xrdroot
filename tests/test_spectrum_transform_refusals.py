"""``TSpectrumTransform`` and ``TSpectrum2Transform``: what ROOT refuses, and what it does anyway.

The messages are ROOT 6.40's, word for word - ``TSpectrum2Transform`` names
itself ``TSpectrumTransform`` in them, as ROOT's does - and a setting ROOT
refuses is left as it was. What ROOT would do only by reading or writing
memory it never set is refused here in a sentence instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from xrdroot.pyroot import TSpectrum2Transform, TSpectrumTransform


def _errors(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().err.splitlines()


def test_a_1d_transform_refuses_lengths_and_settings_with_roots_messages(capsys):
    TSpectrumTransform(0)
    TSpectrumTransform(12)
    t = TSpectrumTransform(16)
    t.SetTransformType(13, 0)
    t.SetTransformType(6, 0)
    t.SetTransformType(6, 5)
    t.SetRegion(-1, 3)
    t.SetRegion(3, 16)
    t.SetDirection(2)
    where = "Error in <TSpectrumTransform::TSpectrumTransform>: "
    assert _errors(capsys) == [
        where + "Invalid length, must be > than 0",
        where + "Invalid length, must be power of 2",
        where + "Invalid type of transform",
        where + "Invalid degree of mixed transform",
        where + "Invalid degree of mixed transform",
        where + "Wrong range",
        where + "Wrong range",
        where + "Wrong direction",
    ]


def test_a_2d_transform_refuses_as_roots_does_naming_itself_the_1d_class(capsys):
    TSpectrum2Transform(0, 4)
    TSpectrum2Transform(8, 12)
    TSpectrum2Transform(8)
    u = TSpectrum2Transform(8, 16)
    u.SetTransformType(6, 4)
    u.SetTransformType(99, 0)
    u.SetRegion(0, 8, 0, 3)
    u.SetRegion(0, 7, 5, 3)
    u.SetDirection(-1)
    where = "Error in <TSpectrum2Transform::TSpectrumTransform>: "
    assert _errors(capsys) == [
        where + "Invalid length, must be > than 0",
        where + "Invalid length, must be power of 2",
        where + "Invalid length, must be > than 0",
        where + "Invalid degree of mixed transform",
        where + "Invalid type of transform",
        where + "Wrong range",
        where + "Wrong range",
        where + "Wrong direction",
    ]


def test_the_1d_class_is_named_as_roots_and_the_2d_one_is_a_plain_tobject():
    t, u = TSpectrumTransform(16), TSpectrum2Transform(8, 16)
    assert (t.GetName(), t.GetTitle(), TSpectrumTransform().GetName()) == (
        "SpectrumTransform", "Miroslav Morhac transformer", "",
    )  # fmt: skip
    assert (u.GetName(), u.GetTitle()) == (
        "TSpectrum2Transform",
        "Spectrum2 Transformer, it calculates classic orthogonal 2D transforms",
    )
    assert (t.kTransformSinHaar, t.kTransformInverse, u.kTransformCosWalsh) == (12, 1, 9)


def test_a_refused_setting_is_left_as_it_was(capsys):
    t = TSpectrumTransform(16)
    t.SetTransformType(t.kTransformHaar, 0)
    t.SetTransformType(t.kTransformWalshHaar, 9)
    t.SetDirection(5)
    haar = np.zeros(16)
    t.Transform(np.arange(16.0), haar)
    capsys.readouterr()
    assert haar[:2].tolist() == [30.0, -16.0]


def test_a_transform_whose_constructor_refused_its_length_refuses_to_run(capsys):
    with pytest.raises(ValueError, match="no length to transform"):
        TSpectrumTransform(12).Transform(np.zeros(12), np.zeros(12))
    with pytest.raises(ValueError, match="TSpectrum2Transform::Enhance has no length"):
        TSpectrum2Transform().Enhance([np.zeros(4)], [np.zeros(4)])
    capsys.readouterr()


def test_a_source_shorter_than_roots_read_of_it_is_refused():
    t = TSpectrumTransform(16)
    t.SetDirection(t.kTransformInverse)
    t.SetTransformType(t.kTransformFourier, 0)
    with pytest.raises(ValueError, match="reads 32 numbers from its source, which holds only 16"):
        t.Transform(np.zeros(16), np.zeros(32))


def test_a_cosine_transform_doubles_its_size_so_a_second_one_reads_twice_as_much():
    t = TSpectrumTransform(8)
    t.Transform(np.arange(8.0), np.zeros(8))
    t.SetRegion(0, 15)
    with pytest.raises(ValueError, match="TSpectrumTransform::Transform reads 16 numbers"):
        t.Transform(np.arange(8.0), np.zeros(16))


def test_a_cosine_mixed_transform_raises_its_degree_until_root_would_divide_by_zero():
    t = TSpectrumTransform(8)
    t.SetTransformType(t.kTransformCosWalsh, 3)
    t.FilterZonal(np.arange(8.0), np.zeros(8))
    with pytest.raises(ValueError, match="raise this cosine or sine mixed transform's degree to 5"):
        t.Enhance(np.arange(8.0), np.zeros(8))


def test_a_filter_leaving_nothing_scales_nothing_and_a_2d_one_writes_nothing():
    t = TSpectrumTransform(8)
    t.SetTransformType(t.kTransformWalsh, 0)
    dest = np.full(8, 7.0)
    t.FilterZonal(np.zeros(8), dest)
    assert dest.tolist() == [0.0] * 8
    u = TSpectrum2Transform(4, 4)
    u.SetTransformType(u.kTransformHaar, 0)
    rows = [np.full(4, 7.0) for _ in range(4)]
    u.FilterZonal([np.zeros(4)] * 4, rows)
    u.Enhance([np.zeros(4)] * 4, rows)
    assert np.concatenate(rows).tolist() == [7.0] * 16
