"""``PDERS``: the densities of signal and background counted in a box around each event.

The box is centred on the event, ``DeltaFrac`` times each variable's
average RMS wide - or its range (``MinMax``), or ``DeltaFrac`` itself
(``Unscaled``) - and with ``VolumeRangeMode=Adaptive`` grown or shrunk,
by TMVA's own secant steps in single precision, until it holds between
``NEventsMin`` and ``NEventsMax`` events' weight. The training events in
it are weighed by a kernel of their distance from the event, and the
output is the signal's share of the two densities, each class's weight
normalised; for a regression, the kernel-weighted mean target. The boxes
of a batch of events are all worked out together.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..dataset import Events
from ..method import CLASSIFICATION, REGRESSION, Method
from ..searchtree import TreeEvents, read_tree_xml, tree_xml
from ..xmlfile import Node, children
from .pderskernels import KERNELS, kernel_values

__all__ = ["MethodPDERS"]

#: How many events have their boxes found at once.
CHUNK = 256
#: The volume modes xrdroot has.
MODES = ("Unscaled", "MinMax", "RMS", "Adaptive")

f32 = np.float32


def _scaled(lower: Any, upper: Any, factor: Any) -> tuple[Any, Any]:
    """``Volume::ScaleInterval``: each side scaled about the centre by ``factor``."""
    factor = np.asarray(factor, dtype=np.float64)[:, None]
    return (
        0.5 * (lower * (1.0 + factor) + upper * (1.0 - factor)),
        0.5 * (lower * (1.0 - factor) + upper * (1.0 + factor)),
    )


class MethodPDERS(Method):
    """``TMVA::MethodPDERS``."""

    type_name = "PDERS"
    analyses = frozenset({CLASSIFICATION, REGRESSION})
    defaults: ClassVar[dict[str, Any]] = {
        "VolumeRangeMode": "Adaptive",
        "KernelEstimator": "Box",
        "DeltaFrac": 3.0,
        "NEventsMin": 100.0,
        "NEventsMax": 200.0,
        "MaxVIterations": 150.0,
        "InitialScale": 0.99,
        "GaussSigma": 0.1,
        "NormTree": False,
    }

    def process_options(self) -> None:
        self.mode = str(self.opt("VolumeRangeMode"))
        self.kernel = str(self.opt("KernelEstimator"))
        for what, value, known in (
            ("VolumeRangeMode", self.mode, MODES),
            ("KernelEstimator", self.kernel, KERNELS),
        ):
            if value not in known:
                raise self.log.refuse(
                    f"{what}={value} is an option of TMVA's PDERS xrdroot does not have; it has "
                    + ", ".join(known)
                )
        if self.opt("IgnoreNegWeightsInTraining"):
            raise self.log.fatal(
                "Mechanism to ignore events with negative weights in training not yet available "
                'for method: PDERS --> please remove "IgnoreNegWeightsInTraining" option from '
                "booking string."
            )

    # -- training -------------------------------------------------------------------------

    def train(self, events: Events) -> None:
        targets = events.targets if self.analysis == REGRESSION else None
        weights = np.asarray(events.weights, dtype=np.float32).astype(np.float64)
        self.tree = TreeEvents(
            np.asarray(events.values, dtype=np.float64), events.classes, weights, targets
        )
        self._prepare()

    def _prepare(self) -> None:
        """``CalcAverages`` and ``SetVolumeElement``: the class scales and the box's size."""
        tree = self.tree
        signal = tree.classes == 0
        self.scales = [
            f32(1.0 / s) if s > 0 else f32(1.0)
            for s in (tree.weights[signal].sum(), tree.weights[~signal].sum())
        ]
        rms = [self._rms(signal)]
        if self.analysis != REGRESSION:
            rms.append(self._rms(~signal))
        average = f32(0.5) * (rms[0] + rms[1]) if len(rms) == 2 else rms[0]
        frac = f32(self.opt("DeltaFrac"))
        if self.mode in ("RMS", "Adaptive"):
            self.delta = (average * frac).astype(f32)
        elif self.mode == "MinMax":
            spans = [info.maximum - info.minimum for info in self.dsi.variables]
            self.delta = (np.asarray(spans) * float(frac)).astype(f32)
        else:
            self.delta = np.full(tree.values.shape[1], frac, dtype=f32)

    def _rms(self, mask: Any) -> Any:
        """``BinarySearchTree::RMS`` of one class: its sums in double, its moments in single."""
        values, weights = self.tree.values[mask], self.tree.weights[mask]
        total = weights.sum()
        if total == 0:
            return np.zeros(values.shape[1], dtype=f32)
        squares = (values.astype(f32) * values.astype(f32)).astype(np.float64)
        mean = (weights @ values / total).astype(f32)
        return np.sqrt(weights @ squares / total - (mean * mean).astype(np.float64)).astype(f32)

    # -- the box --------------------------------------------------------------------------

    def _count(self, lower: Any, upper: Any) -> Any:
        """``SearchVolume``: the weight of the training events in each box, in single precision."""
        return (self._inside(lower, upper) @ self.tree.weights).astype(f32)

    def _inside(self, lower: Any, upper: Any) -> Any:
        points = self.tree.values[None, :, :]
        return np.all((lower[:, None, :] < points) & (upper[:, None, :] >= points), axis=2)

    def _adapt(self, lower: Any, upper: Any) -> tuple[Any, Any]:
        """``GetSample``'s adaptive volume: the box of each event, scaled until it holds enough."""
        low, high = f32(self.opt("NEventsMin")), f32(self.opt("NEventsMax"))
        count = self._count(lower, upper)
        while np.any(few := count < low):
            lower[few], upper[few] = _scaled(lower[few], upper[few], np.full(few.sum(), 1.15))
            count[few] = self._count(lower[few], upper[few])
        n_old, n_new = count, count.copy()
        expected = f32(0.5 * float(low + high))
        scale_n = np.full(len(count), f32(self.opt("InitialScale")), dtype=f32)
        scale, best_scale, best = scale_n.copy(), scale_n.copy(), n_new.copy()
        for _ in range(1, int(f32(self.opt("MaxVIterations")))):
            active = (n_new < low) | (n_new > high)
            if not active.any():
                break
            a = np.flatnonzero(active)
            n_new[a] = self._count(*_scaled(lower[a], upper[a], scale[a]))
            moved, stepped = (n_new[a] > 1) & (n_new[a] - n_old[a] != 0), scale_n[a] - f32(1.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                secant = stepped / (n_new[a] - n_old[a]) * (expected - n_new[a])
            fallback = (scale[a].astype(np.float64) - 0.01).astype(f32)
            step = np.where(stepped != 0, scale[a] + secant, fallback)
            scale[a] = np.where(moved, step, scale[a] + f32(0.5))
            scale_n[a] = scale[a]
            better = (np.abs(n_new[a] - expected) < np.abs(best[a] - expected)) & (
                (n_new[a] >= low) | (best[a] < n_new[a])
            )
            best[a[better]], best_scale[a[better]] = n_new[a[better]], scale[a[better]]
        return _scaled(lower, upper, best_scale)

    def _boxes(self, values: Any) -> tuple[Any, Any]:
        lower = values - self.delta.astype(np.float64) * (1.0 - 0.5)
        upper = values + self.delta.astype(np.float64) * 0.5
        return self._adapt(lower, upper) if self.mode == "Adaptive" else (lower, upper)

    # -- evaluation -----------------------------------------------------------------------

    def evaluate(self, values: Any) -> Any:
        values = np.asarray(values, dtype=np.float64)
        found = [
            self._estimate(values[start : start + CHUNK]) for start in range(0, len(values), CHUNK)
        ]
        output = np.concatenate(found) if found else np.zeros(0)
        if self.analysis == REGRESSION:
            return self.handler.inverse_targets(output[:, None])
        return output

    def _estimate(self, values: Any) -> Any:
        """``CKernelEstimate`` (or ``RKernelEstimate``) of a batch of events."""
        lower, upper = self._boxes(values.copy())
        inside = self._inside(lower, upper)
        norm = 2.0 / (upper - lower)
        offsets = norm[:, None, :] * (self.tree.values[None, :, :] - values[:, None, :])
        distance = np.sqrt((offsets * offsets).sum(axis=2) / values.shape[1])
        if self.kernel != "Box":
            inside &= distance <= 1
        weight = (
            np.where(
                inside,
                kernel_values(self.kernel, distance, self.opt("GaussSigma"), values.shape[1]),
                0.0,
            )
            * self.tree.weights
        )
        if self.analysis == REGRESSION:
            total = weight.sum(axis=1)
            mean = weight @ self.tree.targets[:, 0]
            return np.where(total != 0, mean / np.where(total != 0, total, 1), 0.0)
        signal = self.tree.classes == 0
        pdf_s = np.maximum(weight[:, signal].sum(axis=1), 0.0)
        pdf_b = np.maximum(weight[:, ~signal].sum(axis=1), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (pdf_b * float(self.scales[1]) / (pdf_s * float(self.scales[0]))).astype(f32)
            output = 1.0 / (ratio.astype(np.float64) + 1.0)
        output = np.where(pdf_s < 1e-20, 0.0, output)
        output = np.where(pdf_b < 1e-20, 1.0, output)
        return np.where((pdf_s < 1e-20) & (pdf_b < 1e-20), 0.5, output)

    # -- the weight file ------------------------------------------------------------------

    def add_weights(self, node: Node) -> None:
        tree_xml(node.add("Weights"), self.tree, bool(self.opt("NormTree")))

    def read_weights(self, node: Any) -> None:
        # TMVA does not process a method's options when it reads its weight file, so the
        # volume mode and kernel are then the defaults ``Init`` set - whatever was booked.
        self.mode, self.kernel = "Adaptive", "Box"
        self.tree = read_tree_xml(children(node, "BinaryTree")[0])
        self._prepare()
        self.log.info(f"signal and background scales: {self.scales[0]:g} {self.scales[1]:g}")
