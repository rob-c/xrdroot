"""``TMVA::Experimental``'s reader and scaler: ``RReader``, ``Compute`` and ``RStandardScaler``.

``RReader(weightfile)`` is a :class:`~.reader.Reader` that declares the
file's own variables and answers a vector of values, or an ``RTensor`` of
events a row each; ``Compute<N, T>(model)`` makes of it the function an
``RDataFrame::Define`` calls. ``RStandardScaler`` learns each column's
mean and standard deviation from a tensor and scales tensors by them, and
saves and reads them as TMVA does, in a ROOT file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .method import CLASSIFICATION, MULTICLASS, REGRESSION
from .tensor import RTensor
from .tools import CxxVector
from .weightfile import read_method
from .xmlfile import load

__all__ = ["Compute", "RReader", "RStandardScaler"]


class RReader:
    """``RReader(path)``: the method in a weight file, with the file's variables."""

    def __init__(self, path: Any) -> None:
        from .dataset import Events

        self._events = Events
        self.path = str(path)
        root = load(self.path)
        self.names = [str(item.get("Expression")) for item in root.find("Variables")]
        from .log import CONFIG

        # TMVA's RReader makes its reader "Silent".
        silent, CONFIG.silent = CONFIG.silent, True
        try:
            self.method = read_method(self.path)
        finally:
            CONFIG.silent = silent

    def GetVariableNames(self) -> CxxVector:
        return CxxVector(self.names)

    def _answer(self, values: Any) -> Any:
        rows = np.asarray(values, dtype=np.float32).astype(np.float64).reshape(-1, len(self.names))
        n = len(rows)
        dsi = self.method.dsi
        events = self._events(
            rows,
            np.zeros((n, max(dsi.GetNTargets(), 1))),
            np.zeros((n, len(dsi.spectators))),
            np.zeros(n, dtype=np.int64),
            np.ones(n),
        )
        found = np.asarray(self.method.mva(events), dtype=np.float64)
        if self.method.analysis == CLASSIFICATION:
            return found.reshape(n, 1)
        return found.reshape(n, -1)

    def Compute(self, values: Any) -> Any:
        """One event's outputs as a vector, or every row of a tensor's as a tensor."""
        if isinstance(values, RTensor):
            found = self._answer(values.array).astype(np.float32)
            return RTensor["float"].wrap(found[:, 0] if found.shape[1] == 1 else found)
        return CxxVector(float(v) for v in self._answer(list(values))[0].astype(np.float32))


class _Compute:
    """``Compute<N, T>(model)``: a function of ``N`` columns giving the model's first output."""

    def __getitem__(self, _: Any) -> _Compute:
        return self

    def __call__(self, model: RReader) -> Any:
        def compute(*columns: Any) -> Any:
            table = np.column_stack([np.asarray(c, dtype=np.float64) for c in columns])
            return model._answer(table)[:, 0].astype(np.float32)

        return compute


class RStandardScaler:
    """``RStandardScaler<T>()``, or ``(title, filename)`` for one saved before."""

    dtype: Any = np.float32

    def __class_getitem__(cls, kind: Any) -> type[RStandardScaler]:
        dtype = RTensor[kind].dtype
        return type(f"RStandardScaler<{kind}>", (RStandardScaler,), {"dtype": dtype})

    def __init__(self, title: Any = None, filename: Any = None) -> None:
        self.means = np.zeros(0, dtype=self.dtype)
        self.stds = np.zeros(0, dtype=self.dtype)
        if title is not None and filename is not None:
            import xrdroot

            with xrdroot.open_root(str(filename)) as source:
                found = source[str(title)].arrays(["means", "stds"])
            self.means = np.asarray(found["means"], dtype=self.dtype)
            self.stds = np.asarray(found["stds"], dtype=self.dtype)

    def Fit(self, x: RTensor) -> None:
        """Each column's mean, and its standard deviation with ``n - 1``, in ``T``'s precision."""
        table = np.asarray(x.array)
        if table.ndim != 2:
            raise RuntimeError("Can only fit to input tensor of rank 2.")
        rows = table.shape[0]
        sums = np.zeros(table.shape[1], dtype=self.dtype)
        for row in table:
            sums = (sums + row).astype(self.dtype)
        self.means = (sums / self.dtype(rows)).astype(self.dtype)
        squares = np.zeros(table.shape[1], dtype=self.dtype)
        for row in table:
            squares = (squares + (row - self.means) * (row - self.means)).astype(self.dtype)
        self.stds = np.sqrt(squares / self.dtype(rows - 1)).astype(self.dtype)

    def Compute(self, x: Any) -> Any:
        if isinstance(x, RTensor):
            table = np.asarray(x.array)
            if table.ndim != 2:
                raise RuntimeError("Can only compute output for input tensor of rank 2.")
            scaled = ((table - self.means) / self.stds).astype(self.dtype)
            return type(x).wrap(scaled, x.layout)
        values = np.asarray(list(x), dtype=self.dtype)
        return CxxVector(float(v) for v in ((values - self.means) / self.stds).astype(self.dtype))

    def GetMeans(self) -> CxxVector:
        return CxxVector(float(v) for v in self.means)

    def GetStds(self) -> CxxVector:
        return CxxVector(float(v) for v in self.stds)

    def Save(self, title: Any, filename: Any) -> None:
        """Written as a tree called ``title`` of the means and deviations, which ``(title, file)``
        reads back; TMVA writes the scaler object itself, which only TMVA can read."""
        import os

        import xrdroot

        opener = xrdroot.update if os.path.exists(str(filename)) else xrdroot.create
        with opener(str(filename)) as out:
            out.tree(str(title), {"means": self.means.dtype, "stds": self.stds.dtype}).extend(
                {"means": self.means, "stds": self.stds}
            )


#: ``TMVA::Experimental::Compute``, templated on the number of inputs and their type.
Compute = _Compute()
#: The analyses a reader answers for, as ``RReader`` names them.
ANALYSES = (CLASSIFICATION, REGRESSION, MULTICLASS)
