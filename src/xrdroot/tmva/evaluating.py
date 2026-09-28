"""The Factory's ``EvaluateAllMethods`` for classifiers: TMVA's tables, and the trees written."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .efficiency import Efficiencies
from .evaluation import roc_curve, roc_integral, test_classification
from .log import Logger, color
from .method import CLASSIFICATION, REGRESSION
from .output import event_tree
from .plots import plot_variables
from .training import method_directory

if TYPE_CHECKING:
    from .factory import Booked
    from .output import Output

__all__ = ["Evaluating"]

#: The rule TMVA's classification tables are drawn with.
HLINE = "-" * 115
#: The background efficiencies the signal efficiency is quoted at.
EFFICIENCIES = (0.01, 0.10, 0.30)


def _signal(item: Booked, classes: Any) -> Any:
    return classes == item.method.dsi.GetSignalClassIndex()


class Evaluating:
    """``EvaluateAllMethods`` and the ROC queries, for :class:`~.factory.Factory` to inherit."""

    booked: dict[str, list[Booked]]
    output: Output
    log: Logger
    analysis: int
    roc: bool

    def _evaluate_classifier(self, item: Booked) -> dict[str, Any]:
        """``TestClassification`` and every number TMVA quotes for one classifier."""
        method = item.method
        self.log.header(f"Evaluate classifier: {method.name}")
        self.log.info("")
        dataset = item.loader.dataset()
        test = dataset.test
        if method.handler.transforms:
            method.handler.print_stats(method.handler.apply(test))
        method.log.header(
            f"[{method.dsi.name}] : Loop over test events and fill histograms with classifier "
            "response..."
        )
        method.log.info("")
        values = np.asarray(item.test_values, dtype=np.float32).astype(np.float64)
        extra = None
        if method.mva_pdfs is not None:
            method.log.info("Also filling probability and rarity histograms (on request)...")
            extra = {"proba": item.test_proba, "rarity": method.rarity(values)}
        signal = _signal(item, test.classes)
        tested = test_classification(method.testvar, values, signal, test.weights, extra)
        efficiencies = Efficiencies(method.testvar, tested.xmin, tested.xmax, tested.positive)
        top = tested.xmax + 0.00001
        efficiencies.test(values, signal, test.weights, tested.xmin, top)
        train = dataset.train
        train_values = method.mva(train)
        efficiencies.train(
            train_values, _signal(item, train.classes), train.weights, tested.xmin, top
        )
        item.test, item.efficiencies = tested, efficiencies
        self._write_evaluation(item, test)
        found = {
            "sig": tested.significance(),
            "sep": tested.separation(),
            "area": efficiencies.area(),
            "roc": self._roc(item),
            "name": method.name,
        }
        for level in EFFICIENCIES:
            found[f"eff{level}"] = efficiencies.efficiency(level)[0]
            found[f"train{level}"] = efficiencies.training_efficiency(level)
        return found

    def _write_evaluation(self, item: Booked, test: Any) -> None:
        """``WriteEvaluationHistosToFile``: the densities, the results, the variables' plots."""
        method = item.method
        where = method_directory(method)
        for pdf in method.mva_pdfs or ():
            for histogram in (pdf.original, pdf.smoothed, pdf.fine):
                self.output.write(where, histogram)
        for key in (
            "MVA_S",
            "MVA_B",
            "proba_S",
            "proba_B",
            "rarity_S",
            "rarity_B",
            "MVA_HIGHBIN_S",
            "MVA_HIGHBIN_B",
        ):
            if item.test is not None and key in item.test.histograms:
                self.output.write(where, item.test.histograms[key])
        efficiencies = item.efficiencies
        for histogram in efficiencies.histograms.values() if efficiencies else ():
            self.output.write(where, histogram)
        transformed = method.handler.apply(test)
        method.handler.print_stats(transformed)
        note = method.handler.name if method.handler.transforms else ""
        plots = plot_variables(method.dsi, transformed, method.handler.stats, "", note)
        for histogram in plots.histograms:
            self.output.write(where, histogram)
        for histogram in plots.correlations:
            self.output.write(f"{where}/CorrelationPlots", histogram)

    def _roc(self, item: Booked) -> float:
        test = item.loader.dataset().test
        return roc_integral(item.test_values, _signal(item, test.classes), test.weights)

    def _classification_tables(self, dataset: str, rows: list[dict[str, Any]]) -> None:
        """The two tables TMVA ends a classification with, the best method first."""
        ranked = sorted(rows, key=lambda row: -row["area"])
        log = self.log
        log.info("")
        log.info("Evaluation results ranked by best signal efficiency and purity (area)")
        log.info(HLINE)
        log.info("DataSet       MVA                       ")
        log.info("Name:         Method:          ROC-integ")
        for row in ranked:
            value = row["area"] if row["sep"] < 0 or row["sig"] < 0 else row["roc"]
            log.info(f"{dataset:<13} {row['name']:<15}: {value:#1.3f}")
        log.info(HLINE)
        log.info("")
        log.info("Testing efficiency compared to training efficiency (overtraining check)")
        log.info(HLINE)
        log.info(
            "DataSet              MVA              Signal efficiency: from test sample "
            "(from training sample) "
        )
        log.info(
            "Name:                Method:          @B=0.01             @B=0.10            @B=0.30   "
        )
        log.info(HLINE)
        for row in ranked:
            cells = [
                f"{row[f'eff{level}']:#1.3f} ({row[f'train{level}']:#1.3f})"
                for level in EFFICIENCIES
            ]
            log.info(
                f"{dataset:<20} {row['name']:<15}: {cells[0]}       {cells[1]}      {cells[2]}"
            )
        log.info(HLINE)
        log.info("")

    def _write_trees(self, name: str, items: list[Booked]) -> None:
        """The ``TestTree`` and ``TrainTree``: every event, and every method's output for it."""
        if self.output.silent or not items:
            return
        dataset = items[0].loader.dataset()
        dsi = items[0].loader.info
        for tree, events, key in (
            ("TestTree", dataset.test, "test"),
            ("TrainTree", dataset.train, "train"),
        ):
            outputs: dict[str, Any] = {}
            for item in items:
                values = item.test_values if key == "test" else item.train_values
                if values is not None:
                    outputs[item.method.name] = values
                probability = item.extra.get(f"prob_{key}")
                if probability is not None:
                    outputs[f"prob_{item.method.name}"] = probability
            self.output.write_tree(name, tree, event_tree(dsi, events, outputs))
            Logger(f"Dataset:{name}").header(f"Created tree '{tree}' with {len(events)} events")
            Logger(f"Dataset:{name}").info("")

    def EvaluateAllMethods(self) -> None:
        """Evaluate every method on its test sample, print TMVA's tables, write the trees."""
        self.log.header(f"{color('bold')}Evaluate all methods{color('reset')}")
        if not self.booked:
            self.log.info("...nothing found to evaluate")
            return
        for name in sorted(self.booked):
            items = self.booked[name]
            if self.analysis == CLASSIFICATION:
                rows = [self._evaluate_classifier(item) for item in items]
                if self.roc:
                    self._classification_tables(name, rows)
            self._write_trees(name, items)
        self.log.header(f"{color('bold')}Thank you for using TMVA!{color('reset')}")
        self.log.info(
            f"{color('bold')}For citation information, please visit: "
            f"http://tmva.sf.net/citeTMVA.html{color('reset')}"
        )

    # -- ROC queries ----------------------------------------------------------------------

    def _item(self, dataset: Any, title: Any) -> Booked:
        name = dataset.GetName() if hasattr(dataset, "GetName") else str(dataset)
        for item in self.booked.get(name, []):
            if item.method.name == str(title):
                return item
        raise self.log.fatal(f"Method = {title} not found with Dataset = {name} ")

    def _sample(self, item: Booked, iclass: int, kind: int) -> tuple[Any, Any, Any]:
        dataset = item.loader.dataset()
        events = dataset.test if kind == 1 else dataset.train
        values = item.test_values if kind == 1 else item.train_values
        values = np.asarray(values)
        if values.ndim == 2:
            return values[:, iclass], events.classes == iclass, events.weights
        return values, _signal(item, events.classes), events.weights

    def GetROCIntegral(self, dataset: Any, title: Any, iClass: int = 0, type: int = 1) -> float:
        """``Factory::GetROCIntegral``: the area under the method's ROC curve on the test sample."""
        if self.analysis == REGRESSION:
            self.log.error(
                "Can only generate ROC integral for analysis type kClassification. and kMulticlass."
            )
            return 0.0
        return roc_integral(*self._sample(self._item(dataset, title), iClass, type))

    def GetROCCurve(self, dataset: Any, title: Any = None, *args: Any) -> Any:
        """``Factory::GetROCCurve``: the curve as a ``TGraph`` - or, for a loader, every method's."""
        from ..pyroot.core.graphs import TGraph

        if title is None or isinstance(title, bool):
            return self._roc_canvas(dataset)
        iclass = int(args[1]) if len(args) > 1 else 0
        kind = int(args[2]) if len(args) > 2 else 1
        sensitivity, specificity = roc_curve(
            *self._sample(self._item(dataset, title), iclass, kind)
        )
        graph = TGraph(len(sensitivity), np.asarray(sensitivity), np.asarray(specificity))
        if not args or args[0]:
            graph.SetTitle("Signal efficiency vs. Background rejection")
            graph.GetXaxis().SetTitle("Signal efficiency")
            graph.GetYaxis().SetTitle("Background rejection")
        graph.SetName(str(title))
        return graph

    def _roc_canvas(self, dataset: Any) -> Any:
        """``GetROCCurve(dataloader)``: a canvas of every method's ROC curve, with a legend."""
        from ..pyroot.graphics.canvas import TCanvas
        from ..pyroot.graphics.legend import TLegend

        name = dataset.GetName() if hasattr(dataset, "GetName") else str(dataset)
        canvas = TCanvas(name, "ROC Curve", 800, 600)
        legend = TLegend(0.15, 0.15, 0.35, 0.3, "MVA Method")
        colour = 1
        for position, item in enumerate(self.booked.get(name, [])):
            graph = self.GetROCCurve(name, item.method.name)
            graph.SetLineWidth(2)
            graph.SetLineColor(colour)
            colour += 1
            if colour == 5 or colour == 10 or colour == 11:
                colour += 1
            graph.Draw("AL" if position == 0 else "L")
            legend.AddEntry(graph, item.method.name, "l")
        legend.Draw()
        canvas._tmva_legend = legend
        return canvas
