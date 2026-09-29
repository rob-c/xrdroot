"""Small input histograms for the HistFactory tests, written to a ROOT file by xrdroot's writer.

``hf001``'s example in two bins: data, a signal, two backgrounds, a relative statistical
error histogram and a pair of shape variations - and, for the tests of shape systematics
and of more dimensions, a relative error histogram and a two- and three-dimensional
channel. Each test writes its own file in its ``tmp_path``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

#: One-dimensional histograms over ``[1, 2]`` in two bins: name, bin contents.
ONE_D = {
    "data": [122.0, 112.0],
    "signal": [20.0, 10.0],
    "background1": [100.0, 0.0],
    "background2": [0.0, 100.0],
    "background1_statUncert": [0.05, 0.05],
    "signal_low": [18.0, 9.0],
    "signal_high": [23.0, 11.0],
    "shape_err": [0.125, 0.25],
    "negative": [5.0, -1.0],
}

#: Histograms binned at 0, 1 and 3, the x axis titled: name, bin contents.
VARIABLE = {
    "vdata": [30.0, 45.0],
    "vsig": [10.0, 20.0],
    "vsig_lo": [8.0, 18.0],
    "vsig_hi": [12.0, 23.0],
    "vbkg": [20.0, 25.0],
    "vbkg_lo": [18.0, 24.0],
    "vbkg_hi": [21.0, 27.0],
    "verr": [0.125, 0.25],
}


def _filled(hist: Any, values: list[float]) -> Any:
    """A 1D histogram of ``values``, each with its Poisson error."""
    for i, value in enumerate(values, 1):
        hist.SetBinContent(i, value)
        hist.SetBinError(i, abs(value) ** 0.5)
    return hist


def _write_moved(handle: Any) -> None:
    """``dir/signal``: a histogram in a directory of the file."""
    from xrdroot.pyroot.core import TH1F

    handle.mkdir("dir").cd()
    moved = TH1F("signal", "", 2, 1, 2)
    moved.SetBinContent(1, 20.0)
    moved.SetBinContent(2, 10.0)
    moved.Write()
    handle.cd()


def _write_2d() -> None:
    from xrdroot.pyroot.core import TH2F

    for key, values in (("data2d", [10.0, 12.0, 14.0, 16.0]), ("sig2d", [2.0, 3.0, 4.0, 5.0]),
                        ("bkg2d", [8.0, 9.0, 10.0, 11.0])):  # fmt: skip
        square = TH2F(key, "", 2, 0, 2, 2, 0, 2)
        for i, value in enumerate(values):
            square.SetBinContent(i % 2 + 1, i // 2 + 1, value)
            square.SetBinError(i % 2 + 1, i // 2 + 1, value**0.5)
        square.Write()


def _write_3d() -> None:
    from xrdroot.pyroot.core import TH3F

    for key, scale in (("data3d", 3.0), ("sig3d", 1.0)):
        cube = TH3F(key, "", 2, 0, 2, 1, 0, 1, 2, 0, 2)
        for i in range(4):
            cube.SetBinContent(i % 2 + 1, 1, i // 2 + 1, scale * (i + 1))
        cube.Write()


def write_inputs(directory: pathlib.Path, name: str = "input.root") -> str:
    """The histograms of :data:`ONE_D` and :data:`VARIABLE` - and a 2D and a 3D channel's - in
    ``directory/name``."""
    from xrdroot.pyroot.core import TH1F, TFile

    path = str(directory / name)
    handle = TFile.Open(path, "RECREATE")
    for key, values in ONE_D.items():
        _filled(TH1F(key, key.replace("_", " "), 2, 1, 2), values).Write()
    _write_moved(handle)
    _write_2d()
    for key, values in VARIABLE.items():
        uneven = TH1F(key, "", 2, np.array([0.0, 1.0, 3.0]))
        uneven.GetXaxis().SetTitle("m [GeV]")
        _filled(uneven, values).Write()
    _write_3d()
    handle.Close()
    return path


def histogram(values: list[float], name: str = "h") -> Any:
    """A two-bin histogram over ``[1, 2]`` in no directory."""
    from xrdroot.pyroot.core import TH1F

    hist = TH1F(name, name, len(values), 1, 2)
    hist.SetDirectory(0)
    for i, value in enumerate(values, 1):
        hist.SetBinContent(i, value)
    return hist
