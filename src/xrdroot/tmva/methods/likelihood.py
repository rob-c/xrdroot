"""``Likelihood``: TMVA's projective likelihood, from smoothed and splined reference histograms.

Each variable's signal and background distributions are histogrammed from
the training sample - ``NAvEvtPerBin`` events a bin - smoothed ``NSmooth``
times and splined into a density (:class:`~..pdf.PDF`); an event's output
is the product of its signal densities over the sum of that and the
product of its background densities, and with ``TransformOutput`` that
ratio is put through an inverse sigmoid. As in TMVA, a transformation is
applied with the signal class's parameters for the signal densities and the
background class's for the background ones, so a PCA is each class's own.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .. import hists
from ..dataset import Events
from ..evaluation import separation_of_hists
from ..method import Method
from ..pdf import PDF, pdf_from_xml, settings
from ..xmlfile import Node, children

__all__ = ["MethodLikelihood"]

#: ``fEpsilon``, ``1e3 * DBL_MIN``: the least a density or a likelihood ratio is taken to be.
EPSILON = 1.0e3 * float(np.finfo(np.float64).tiny)


def transform_output(ps: Any, pb: Any, transform: bool) -> Any:
    """``TransformLikelihoodOutput``: ``ps / (ps + pb)``, through ``-log(1/r - 1)/15`` if asked."""
    ps, pb = np.maximum(ps, EPSILON), np.maximum(pb, EPSILON)
    ratio = ps / (ps + pb)
    ratio = np.where(ratio >= 1.0, 1.0 - 1.0e-15, ratio)
    if not transform:
        return ratio
    ratio = np.where(ratio <= 0.0, EPSILON, ratio)
    return -np.log(1.0 / ratio - 1.0) / 15.0


class MethodLikelihood(Method):
    """``TMVA::MethodLikelihood``."""

    type_name = "Likelihood"
    defaults: ClassVar[dict[str, Any]] = {"TransformOutput": False}
    help_text = "Likelihood"

    def process_options(self) -> None:
        self.drop = -1
        self.pdfs: list[tuple[PDF, PDF]] = []

    def _specs(self, index: int) -> tuple[Any, Any]:
        base = settings(self.options)
        return settings(self.options, f"Sig[{index}]", base), settings(
            self.options, f"Bkg[{index}]", base
        )

    def _by_class(self, events: Events) -> tuple[Any, Any]:
        """Each event's values transformed with the signal's, and the background's, parameters."""
        signal = self.dsi.GetSignalClassIndex()
        background = 1 - signal
        return self.handler.apply(events, signal).values, self.handler.apply(
            events, background
        ).values

    def train(self, events: Events) -> None:
        raw = self.raw_train if self.raw_train is not None else events
        if self.opt("IgnoreNegWeightsInTraining"):
            raw = raw.take(raw.weights > 0)
        as_s, as_b = self._by_class(raw)
        low = np.minimum(as_s.min(axis=0), as_b.min(axis=0))
        high = np.maximum(as_s.max(axis=0), as_b.max(axis=0))
        signal = raw.classes == self.dsi.GetSignalClassIndex()
        own = np.where(signal[:, None], as_s, as_b)
        fewest = min(int(signal.sum()), int((~signal).sum()))
        self.log.info("Filling reference histograms")
        filled = [
            self._fill(index, own[:, index], signal, raw.weights, low[index], high[index], fewest)
            for index in range(len(low))
        ]
        self.log.info("Building PDF out of reference histograms")
        self.pdfs = []
        for index, (sig, bgd) in enumerate(filled):
            spec_s, spec_b = self._specs(index)
            self.pdfs.append(
                (
                    PDF(f"{self.name} PDF Sig[{index}]", spec_s).build(sig),
                    PDF(f"{self.name} PDF Bkg[{index}]", spec_b).build(bgd),
                )
            )
        self.reference = filled
        self._train_events = raw

    def _fill(
        self,
        index: int,
        values: Any,
        signal: Any,
        weights: Any,
        low: float,
        high: float,
        fewest: int,
    ) -> tuple[Any, Any]:
        """The signal and background reference histograms of one variable."""
        info = self.dsi.variables[index]
        spec_s, spec_b = self._specs(index)
        stem = f"{self.dsi.name}_{self.name}_{info.label}"
        ranges: list[tuple[int, float, float]]
        if info.vartype == "I":
            start, stop = round(float(low)), round(float(high + 1))
            ranges = [(stop - start, start, stop)] * 2
        else:
            ranges = [(spec.hist_bins(fewest), low, high) for spec in (spec_s, spec_b)]
        clamped = np.where(
            values >= high, high - 1.0e-10, np.where(values < low, low + 1.0e-10, values)
        )
        w = np.asarray(weights, dtype=np.float32).astype(np.float64)
        made = []
        for (nbins, a, b), mask, suffix, label in (
            (ranges[0], signal, "sig", "signal"),
            (ranges[1], ~signal, "bgd", "background"),
        ):
            histogram = hists.book(f"{stem}_{suffix}", f"{stem} {label} training", nbins, a, b)
            histogram.fill(clamped[mask], weight=w[mask])
            made.append(histogram)
        return made[0], made[1]

    def _likelihood(self, as_s: Any, as_b: Any) -> Any:
        ps = np.ones(len(as_s))
        pb = np.ones(len(as_s))
        for index, (pdf_s, pdf_b) in enumerate(self.pdfs):
            if index == self.drop:
                continue
            for values, pdf, product in ((as_s, pdf_s, ps), (as_b, pdf_b, pb)):
                x = values[:, index]
                x = np.where(
                    x >= pdf_s.xmax, pdf_s.xmax - 1.0e-10, np.where(x < pdf_s.xmin, pdf_s.xmin, x)
                )
                product *= np.maximum(pdf.value(x, floor=0.0), EPSILON)
        return transform_output(ps, pb, bool(self.opt("TransformOutput")))

    def mva(self, events: Events) -> Any:
        return self._likelihood(*self._by_class(events))

    def evaluate(self, values: Any) -> Any:
        return self._likelihood(values, values)

    def ranking(self) -> tuple[str, list[tuple[str, float]]]:
        """``CreateRanking``: how much the separation drops without each variable."""
        events = self._train_events
        signal = events.classes == self.dsi.GetSignalClassIndex()
        as_s, as_b = self._by_class(events)
        found, reference = [], 0.0
        for drop in range(-1, len(self.pdfs)):
            self.drop = drop
            values = self._likelihood(as_s, as_b)
            made = [hists.book(f"r{side}_{drop + 1}", "", 80, 0.0, 1.0) for side in "SB"]
            made[0].fill(values[signal], weight=events.weights[signal])
            made[1].fill(values[~signal], weight=events.weights[~signal])
            separation = separation_of_hists(*made)
            if drop == -1:
                reference = separation
            else:
                found.append((self.dsi.variables[drop].internal, reference - separation))
        self.drop = -1
        return "Delta Separation", found

    def monitoring(self, output: Any, directory: str) -> None:
        """The reference histograms, their smoothed copies and densities, as TMVA writes them."""
        output_name = output.GetName()
        self.log.info(f"{output_name}:/{directory}")
        for index, ((pdf_s, pdf_b), (sig, bgd)) in enumerate(zip(self.pdfs, self.reference)):
            for histogram in (sig, bgd, pdf_s.smoothed, pdf_b.smoothed, pdf_s.fine, pdf_b.fine):
                output.write(directory, histogram)
            output.write(directory, self._check(pdf_s, index))
            for histogram in (sig, bgd):
                output.write(
                    directory,
                    hists.renamed(histogram.copy(), histogram.name + "_nice", histogram.title),
                )

    def _check(self, pdf: PDF, index: int) -> Any:
        """``<var>_additional_check``: the signal density sampled in 15000 steps."""
        stem = f"{self.dsi.variables[index].label}_additional_check"
        axis = pdf.fine.axes[0]
        low, high = float(np.float32(axis.low)), float(np.float32(axis.high))
        made = hists.book(stem, stem, 15000, low, high)
        x = (np.arange(15000) + 0.5) * ((high - low) / 15000) + low
        hists.set_bins(made, np.concatenate(([0.0], pdf.value(x), [0.0])), entries=15000)
        return made

    def add_weights(self, node: Node) -> None:
        weights = node.add("Weights", NVariables=len(self.pdfs), NClasses=2)
        for index, pair in enumerate(self.pdfs):
            for cls, pdf in enumerate(pair):
                pdf.add_xml(weights.add("PDFDescriptor", VarIndex=index, ClassIndex=cls))

    def read_weights(self, node: Any) -> None:
        descriptors = children(node, "PDFDescriptor")
        found = [pdf_from_xml(item.find("PDF")) for item in descriptors]
        self.pdfs = [(found[i], found[i + 1]) for i in range(0, len(found), 2)]
