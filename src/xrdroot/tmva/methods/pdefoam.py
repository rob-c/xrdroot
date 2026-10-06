"""``PDEFoam``: the variable space cut into cells by TMVA's foam, each cell's value its events'.

With one foam (``SigBgSeparate=F``) a cell's value is the signal fraction
of the training weight in it; with two, the signal and background foams'
densities give it; for several classes each class has its foam and the
outputs are their softmax; for a regression a cell's value is its events'
mean target, an empty cell taking its neighbours'. The foams are grown by
:mod:`xrdroot.tmva.foam`, TMVA's algorithm draw for draw, and written as
TMVA writes them to ``<weights>_foams.root`` - as a tree of cells per foam,
which xrdroot reads back; TMVA's own foam files, which are ``PDEFoam``
objects, xrdroot cannot read. The decision-tree cell splitting
(``DTLogic``), the kernels and multi-target regression it does not have.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..dataset import Events
from ..foam import Density, Foam, ranges
from ..foamcells import Cells
from ..log import color
from ..method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from ..xmlfile import Node, children, number

__all__ = ["MethodPDEFoam"]

f32 = np.float32


class MethodPDEFoam(Method):
    """``TMVA::MethodPDEFoam``."""

    type_name = "PDEFoam"
    analyses = frozenset({CLASSIFICATION, REGRESSION, MULTICLASS})
    defaults: ClassVar[dict[str, Any]] = {
        "SigBgSeparate": False,
        "TailCut": 0.001,
        "VolFrac": 1.0 / 15.0,
        "nActiveCells": 500,
        "nSampl": 2000,
        "nBin": 5,
        "Compress": True,
        "MultiTargetRegression": False,
        "Nmin": 100,
        "MaxDepth": 0,
        "FillFoamWithOrigWeights": False,
        "UseYesNoCell": False,
        "DTLogic": "None",
        "Kernel": "None",
        "TargetSelection": "Mean",
    }

    def process_options(self) -> None:
        for name, allowed in (
            ("DTLogic", "None"),
            ("Kernel", "None"),
            ("MultiTargetRegression", False),
        ):
            if self.opt(name) != allowed:
                raise self.log.refuse(
                    f"{name}={self.opt(name)} is an option of TMVA's PDEFoam xrdroot does not "
                    f"have; it grows its foams with {name}={allowed} only"
                )
        if self.analysis == REGRESSION and self.dsi.GetNTargets() > 1:
            raise self.log.refuse(
                "PDEFoam's multi-target regression xrdroot does not have; it regresses one target"
            )
        self.separate = bool(self.opt("SigBgSeparate")) and self.analysis == CLASSIFICATION
        self.foams: list[Cells] = []

    def booked(self) -> None:
        """A regression's ``ProcessOptions`` counts the targets, which builds the data set."""
        if self.analysis == REGRESSION and self.loader is not None:
            self.loader.dataset()

    # -- training -------------------------------------------------------------------------

    def _grow(self, name: str, events: Events, kind: str, marks: Any, say: str) -> Foam:
        values = np.asarray(events.values, dtype=np.float64)
        weights = np.asarray(events.weights, dtype=np.float32).astype(np.float64)
        box = ((self.xmax - self.xmin).astype(f32) * f32(self.opt("VolFrac"))).astype(np.float64)
        density = Density(values, weights, box, kind, marks)
        foam = Foam(name, self.xmin, self.xmax, density, self.option_values)
        foam.log = self.log
        self.log.info(say)
        foam.grow()
        self.log.info("Elapsed time: 0 sec                                 ")
        return foam

    def _fill(self, foam: Cells, events: Events, second: Any) -> None:
        """``FillFoamCells``: each event's weight added to its cell, and ``second`` beside it."""
        cells = foam.find(foam.to_unit(events.values, self.xmin, self.xmax))
        weights = np.asarray(events.weights, dtype=np.float32).astype(np.float64)
        np.add.at(foam.elements[:, 0], cells, weights)
        np.add.at(foam.elements[:, 1], cells, weights * second)

    def train(self, events: Events) -> None:
        self.xmin, self.xmax = ranges(events.values, self.opt("TailCut"))
        signal = events.classes == self.dsi.GetSignalClassIndex()
        if self.analysis == REGRESSION:
            target = np.asarray(events.targets[:, 0], dtype=np.float64)
            foam = self._grow(
                "MonoTargetRegressionFoam",
                events,
                "target",
                target,
                "Build mono target regression foam",
            )
            self._fill(foam, events, target)
            _finalise_target(foam)
            self.foams = [foam]
        elif self.analysis == MULTICLASS:
            self.foams = []
            for number_ in range(self.dsi.GetNClasses()):
                mine = events.classes == number_
                foam = self._grow(
                    f"MultiClassFoam{number_}",
                    events,
                    "discriminant",
                    mine,
                    f"Build up multiclass foam {number_}",
                )
                self._discriminate(foam, events, mine)
                self.foams.append(foam)
        else:
            self._classification(events, signal)

    def _classification(self, events: Events, signal: Any) -> None:
        if self.dsi.normalization != "EQUALNUMEVENTS":
            self.log.header(
                f"NormMode={self.dsi.normalization} chosen. Note that only "
                "NormMode=EqualNumEvents ensures that Discriminant values correspond to signal "
                "probabilities."
            )
        if not self.separate:
            foam = self._grow(
                "DiscrFoam", events, "discriminant", signal, "Build up discriminator foam"
            )
            self._discriminate(foam, events, signal)
            self.foams = [foam]
            return
        self.foams = []
        for name, mask in (("SignalFoam", signal), ("BgFoam", ~signal)):
            part = events.take(mask)
            foam = self._grow(name, part, "event", None, f"Build up {name}")
            weights = np.asarray(part.weights, dtype=np.float32).astype(np.float64)
            self._fill(foam, part, weights)
            self.foams.append(foam)

    def _discriminate(self, foam: Cells, events: Events, mine: Any) -> None:
        """The discriminant foam's cells filled - its class and the rest - and finalised."""
        cells = foam.find(foam.to_unit(events.values, self.xmin, self.xmax))
        weights = np.asarray(events.weights, dtype=np.float32).astype(np.float64)
        np.add.at(foam.elements[:, 0], cells[mine], weights[mine])
        np.add.at(foam.elements[:, 1], cells[~mine], weights[~mine])
        _finalise_discriminant(foam)

    # -- evaluation -----------------------------------------------------------------------

    def _values(self, foam: Cells, values: Any) -> tuple[Any, Any]:
        cells = foam.find(foam.to_unit(values, self.xmin, self.xmax))
        return cells, foam.elements[cells].astype(f32).astype(np.float64)

    def evaluate(self, values: Any) -> Any:
        values = np.asarray(values, dtype=np.float64)
        if self.analysis == REGRESSION:
            return self.handler.inverse_targets(self._regression(values)[:, None])
        if self.analysis == MULTICLASS:
            found = np.stack([self._values(foam, values)[1][:, 0] for foam in self.foams], 1)
            shifted = (found[:, None, :] - found[:, :, None]).astype(f32)
            norm = (np.exp(shifted).sum(axis=2) - 1.0).astype(f32)
            return (1.0 / (1.0 + norm)).astype(f32)
        if self.separate:
            density = [self._densities(foam, values) for foam in self.foams]
            total = density[0] + density[1]
            output = np.where(total > 0, density[0] / np.where(total > 0, total, 1), 0.5)
        else:
            output = self._values(self.foams[0], values)[1][:, 0]
        if self.opt("UseYesNoCell"):
            return np.where(output < 0.5, -1.0, 1.0)
        return output

    def error(self, events: Events) -> Any:
        """``CalculateMVAError``: the foam's own error of a discriminant, or one worked out
        from the counts of separate signal and background foams, as Carli and Koblitz give it.
        """
        values = self.handler.apply(events).values
        if not self.separate:
            return self._values(self.foams[0], values)[1][:, 1]
        signal, background = (self._densities(foam, values) for foam in self.foams[:2])
        error_s = np.where(signal == 0, 1.0, np.sqrt(np.abs(signal)))
        error_b = np.where(background == 0, 1.0, np.sqrt(np.abs(background)))
        total = np.where(signal + background == 0, 1.0, signal + background) ** 2
        found = np.hypot(background / total * error_s, signal / total * error_b)
        return np.where((signal > 1e-10) | (background > 1e-10), found, 1.0)

    def _densities(self, foam: Cells, values: Any) -> Any:
        """``GetCellValue(kValue)`` of a foam of events: the cell's count over its volume."""
        cells, found = self._values(foam, values)
        volume = np.array([foam.volume(int(c)) for c in cells])
        return np.where(volume > 2.220446049250313e-16, found[:, 0] / volume, 0)

    def _regression(self, values: Any) -> Any:
        foam = self.foams[0]
        units = foam.to_unit(values, self.xmin, self.xmax)
        cells = foam.find(units)
        output = foam.elements[cells, 0].astype(f32).astype(np.float64)
        for row in np.flatnonzero(foam.elements[cells, 1] == -1):
            output[row] = _neighbours(foam, units[row], int(cells[row]))
        return output

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        importance = np.zeros(self.dsi.GetNVariables(), dtype=np.float32)
        for foam in self.foams:
            cuts = foam.cut_counts().astype(np.float32)
            total = cuts.sum()
            share = cuts / total if total > 0 else np.zeros_like(cuts)
            importance += (share / np.float32(len(self.foams))).astype(np.float32)
        return "Variable Importance", [
            (v.label, float(i)) for v, i in zip(self.dsi.variables, importance, strict=False)
        ]

    # -- the weight file ------------------------------------------------------------------

    @property
    def foam_file(self) -> str:
        return (self.source or self.weight_file).replace(".xml", "_foams.root")

    def add_weights(self, node: Node) -> None:
        from .pdefoamio import write_foams

        opt = self.opt
        weights = node.add(
            "Weights",
            SigBgSeparated=int(bool(opt("SigBgSeparate"))),
            Frac=number(f32(opt("TailCut"))),
            DiscrErrCut=number(-1.0),
            VolFrac=number(f32(opt("VolFrac"))),
            nCells=2 * int(opt("nActiveCells")) - 1,
            nSampl=opt("nSampl"),
            nBin=opt("nBin"),
            EvPerBin=10000,
            Compress=int(bool(opt("Compress"))),
            DoRegression=int(self.analysis == REGRESSION),
            CutNmin=int(int(opt("Nmin")) > 0),
            Nmin=opt("Nmin"),
            CutRMSmin=0,
            RMSmin=number(0.0),
            Kernel=0,
            TargetSelection=int(opt("TargetSelection") != "Mean"),
            FillFoamWithOrigWeights=int(bool(opt("FillFoamWithOrigWeights"))),
            UseYesNoCell=int(bool(opt("UseYesNoCell"))),
        )
        for tag, bound in (("Xmin", self.xmin), ("Xmax", self.xmax)):
            for index, value in enumerate(bound):
                weights.add(tag, Index=index, Value=number(value))
        for name in self._names():
            self.log.info(f"writing foam {name} to file")
        write_foams(self.foam_file, self.foams, self._names())
        self.log.info(
            f"Foams written to file: {color('lightblue')}{self.foam_file}{color('reset')}"
        )

    def _names(self) -> list[str]:
        if self.analysis == REGRESSION:
            return ["MonoTargetRegressionFoam"]
        if self.analysis == MULTICLASS:
            return [f"MultiClassFoam{i}" for i in range(self.dsi.GetNClasses())]
        return ["SignalFoam", "BgFoam"] if self.separate else ["DiscrFoam"]

    def read_weights(self, node: Any) -> None:
        from .pdefoamio import read_foams

        self.separate = node.get("SigBgSeparated") == "1"
        bounds = {tag: np.zeros(self.dsi.GetNVariables()) for tag in ("Xmin", "Xmax")}
        for tag, bound in bounds.items():
            for item in children(node, tag):
                bound[int(str(item.get("Index")))] = float(np.float32(item.get("Value")))
        self.xmin, self.xmax = bounds["Xmin"], bounds["Xmax"]
        self.foams = read_foams(self.foam_file, self._names(), self.dsi.GetNVariables(), self.log)
        self.log.info(f"Read foams from file: {color('lightblue')}{self.foam_file}{color('reset')}")


def _finalise_discriminant(foam: Cells) -> None:
    """``PDEFoamDiscriminant::Finalize``: each active cell's signal fraction and its error."""
    active = np.flatnonzero(foam.status[: foam.last + 1] == 1)
    n_sig = np.maximum(foam.elements[active, 0], 0.0)
    n_bg = np.maximum(foam.elements[active, 1], 0.0)
    total = n_sig + n_bg
    safe = np.where(total > 0, total, 1.0)
    error = np.sqrt((n_sig / safe**2) ** 2 * n_bg + (n_bg / safe**2) ** 2 * n_sig)
    foam.elements[active, 0] = np.where(total > 0, n_sig / safe, 0.5)
    foam.elements[active, 1] = np.where(total > 0, error, 1.0)


def _finalise_target(foam: Cells) -> None:
    """``PDEFoamTarget::Finalize``: each active cell's mean target, or -1 for an empty cell."""
    active = np.flatnonzero(foam.status[: foam.last + 1] == 1)
    count, target = foam.elements[active, 0], foam.elements[active, 1]
    safe = np.where(count > 0, count, 1.0)
    foam.elements[active, 0] = np.where(count > 0, target / safe, 0.0)
    foam.elements[active, 1] = np.where(count > 0, target / np.sqrt(safe), -1.0)


def _neighbours(foam: Cells, point: Any, cell: int) -> float:
    """``GetAverageNeighborsValue``: the mean of the defined cells just past each face."""
    posi, size = foam.hcub(cell)
    result, norm = f32(0.0), f32(0.0)
    for dim in range(foam.dim):
        for edge in (posi[dim] - 1.0e-6, posi[dim] + size[dim] + 1.0e-6):
            moved = np.array(point, dtype=np.float32)
            moved[dim] = edge
            found = int(foam.find(moved[None, :].astype(np.float64))[0])
            if foam.elements[found, 1] != -1:
                result, norm = f32(result + f32(foam.elements[found, 0])), f32(norm + 1)
    return float(result / norm) if norm > 0 else 0.0
