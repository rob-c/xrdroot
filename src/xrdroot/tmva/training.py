"""The Factory's ``TrainAllMethods`` and ``TestAllMethods``, as TMVA runs and describes them."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from . import hists
from .dataset import Events
from .evaluation import norm_hist
from .handler import TransformationHandler
from .helptext import print_help
from .log import Logger, color
from .method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from .pdf import PDF
from .plots import plot_variables
from .ranking import print_ranking
from .weightfile import read_method

if TYPE_CHECKING:
    from .factory import Booked
    from .loader import DataLoader
    from .output import Output

__all__ = ["Training", "evaluate_sample"]

#: The fewest training events a method is trained with.
MIN_TRAINING_EVENTS = 10


def evaluate_sample(method: Method, events: Events, kind: str, dataset: str) -> Any:
    """``GetMvaValues`` over a sample, with the lines TMVA prints about it."""
    header = f"[{dataset}] : Evaluation of {method.name} on {kind} sample ({len(events)} events)"
    method.log.header(header)
    method.log.header(header)
    start = time.perf_counter()
    values = method.mva(events)
    elapsed = time.perf_counter() - start
    for _ in range(2):
        method.log.info(
            f"Elapsed time for evaluation of {len(events)} events: {elapsed:.3g} sec       "
        )
    return values


def method_directory(method: Method) -> str:
    """``dataset/Method_<type>/<name>``: where a method's histograms go."""
    return f"{method.dsi.name}/Method_{method.type_name}/{method.name}"


class Training:
    """``TrainAllMethods`` and ``TestAllMethods``, for :class:`~.factory.Factory` to inherit."""

    booked: dict[str, list[Booked]]
    output: Output
    log: Logger
    transformations: str
    persistence: bool
    analysis: int
    job: str

    # -- the data set's description, once -----------------------------------------------------

    def _write_data_information(self, loader: DataLoader) -> None:
        """``WriteDataInformation``: correlation matrices and the Factory's transformations."""
        name = loader.GetName()
        if self.output.silent or self.output.exists(name):
            return
        self.output.directory(name)
        dataset = loader.dataset()
        self._correlation_hists(loader)
        handlers = []
        for definition in self.transformations.split(";"):
            handler = TransformationHandler(loader.info, "Factory")
            handler.create(definition, self.log)
            handlers.append((definition, handler))
        identity = None
        for definition, handler in handlers:
            transformed = handler.prepare(dataset.train)
            suffix = handler.name or "Id"
            plots = plot_variables(loader.info, transformed, handler.stats, f"_{suffix}", "")
            where = f"{name}/InputVariables_{suffix}"
            for histogram in plots.histograms:
                self.output.write(where, histogram)
            for histogram in plots.correlations:
                self.output.write(f"{where}/CorrelationPlots", histogram)
            if definition.startswith("I"):
                identity = plots
        if identity is not None and identity.separations:
            self.log.info("Ranking input variables (method unspecific)...")
            print_ranking("IdTransformation", "Separation", identity.separations)

    def _correlation_hists(self, loader: DataLoader) -> None:
        from ..hist import Histogram

        info = loader.info
        titles = [variable.title for variable in info.variables]
        pairs = (
            [
                (cls.name, f"CorrelationMatrix{cls.name}", f"Correlation Matrix ({cls.name})")
                for cls in info.classes
            ]
            if self.analysis == MULTICLASS
            else [
                ("Signal", "CorrelationMatrixS", "Correlation Matrix (signal)"),
                ("Background", "CorrelationMatrixB", "Correlation Matrix (background)"),
                ("Regression", "CorrelationMatrix", "Correlation Matrix"),
            ]
        )
        n = len(titles)
        for cls, name, title in pairs:
            matrix = info.correlations.get(cls)
            if matrix is None:
                continue
            made = Histogram.book(
                name, (n, 0.0, float(n)), (n, 0.0, float(n)), title=title, kind="F", labels=None
            )
            contents = np.trunc(np.asarray(matrix, dtype=np.float32) * np.float32(100.0))
            cells = made._cells().reshape(n + 2, n + 2, order="F")
            cells[1:-1, 1:-1] = contents
            made._core["fEntries"] = float(n * n)
            self.output.write(loader.GetName(), made)

    # -- training -------------------------------------------------------------------------

    def _base_directory(self, method: Method) -> str:
        """``BaseDir``: the method's directory, made with its ``TrainingPath`` and ``WeightFileName``."""
        where = method_directory(method)
        if not self.output.silent and not self.output.exists(where):
            self.output.directory(
                f"{method.dsi.name}/Method_{method.type_name}",
                f"Directory for all {method.type_name} methods",
            )
            self.output.directory(where)
            if self.persistence:
                self.output.write(where, os.getcwd(), "TrainingPath")
                self.output.write(where, method.weight_file, "WeightFileName")
        return where

    def _train_one(self, item: Booked) -> None:
        """``MethodBase::TrainMethod``, with the lines TMVA prints for it."""
        method = item.method
        dataset = item.loader.dataset()
        where = self._base_directory(method)
        if method.opt("H"):
            print_help(method.log, method.type_name, method.name)
        train = dataset.train
        transformed = method.handler.prepare(train) if method.handler.transforms else train
        start = time.perf_counter()
        method.n_train = len(train)
        method.train(transformed)
        method.train_time = time.perf_counter() - start
        self.log.info(
            f"Elapsed time for training with {len(train)} events: {method.train_time:.3g} sec         "
        )
        if method.analysis == CLASSIFICATION:
            item.train_values = evaluate_sample(method, train, "training", method.dsi.name)
            if method.has_mva_pdfs():
                self._create_mva_pdfs(item, train, where)
        if self.persistence:
            method.write_weight_file()
        method.monitoring(self.output, where)

    def _create_mva_pdfs(self, item: Booked, train: Events, where: str) -> None:
        """``CreateMVAPdfs``: the output's densities for signal and background, from training."""
        method = item.method
        values = np.asarray(item.train_values, dtype=np.float32).astype(np.float64)
        signal = train.classes == method.dsi.GetSignalClassIndex()
        spec = method.pdf_settings()
        made = []
        for side, mask in (("S", signal), ("B", ~signal)):
            name = f"{method.type_name}_tr_{side}"
            histogram = hists.book(
                name, name, spec.hist_bins(len(values)), values.min(), values.max(), kind="D"
            )
            histogram.sumw2(True)
            histogram.fill(values[mask], weight=train.weights[mask])
            norm_hist(histogram)
            self.output.write(where, histogram)
            made.append(histogram)
        pdfs = (
            PDF(f"{method.name}_PDFSig", spec).build(made[0]),
            PDF(f"{method.name}_PDFBkg", spec).build(made[1]),
        )
        method.mva_pdfs = pdfs
        from .evaluation import separation_of_hists

        method.log.info(
            f"<CreateMVAPdfs> Separation from histogram (PDF): "
            f"{separation_of_hists(made[0], made[1]):1.3f} (0.000)"
        )
        method.log.info(
            f"Dataset[{method.dsi.name}] : Evaluation of {method.name} on training sample"
        )
        item.extra["prob_train"] = np.asarray(method.proba(values, 0.5), dtype=np.float32)
        for pdf in pdfs:
            for histogram in (pdf.original, pdf.smoothed, pdf.fine):
                self.output.write(where, histogram)

    def TrainAllMethods(self) -> None:
        """Train every booked method, rank the variables, and read each back from its weight file."""
        self.log.header(f"{color('bold')}Train all methods{color('reset')}")
        if not self.booked:
            self.log.info("...nothing found to train")
            return
        for name in sorted(self.booked):
            items = self.booked[name]
            for item in items:
                self._check_training(item)
                self._write_data_information(item.loader)
                words = {REGRESSION: "Regression", MULTICLASS: "Multiclass classification"}
                what = words.get(item.method.analysis, "Classification")
                self.log.header(f"Train method: {item.method.name} for {what}")
                self.log.info("")
                self._train_one(item)
                self.log.header("Training finished")
                self.log.info("")
            if self.analysis != REGRESSION:
                self._rank(items)
            if self.persistence:
                self._recreate(items)

    def _check_training(self, item: Booked) -> None:
        info = item.loader.info
        if self.analysis == REGRESSION and info.GetNTargets() < 1:
            raise self.log.fatal("You want to do regression training without specifying a target.")
        if self.analysis != REGRESSION and info.GetNClasses() < 2:
            raise self.log.fatal(
                "You want to do classification training, but specified less than two classes."
            )

    def _rank(self, items: list[Booked]) -> None:
        self.log.info("Ranking input variables (method specific)...")
        for item in items:
            ranking = item.method.ranking()
            if ranking is None:
                self.log.info(f"No variable ranking supplied by classifier: {item.method.name}")
            else:
                print_ranking(item.method.name, *ranking)

    def _recreate(self, items: list[Booked]) -> None:
        """``Destroy and recreate all methods via weight files for testing``."""
        self.log.header("=== Destroy and recreate all methods via weight files for testing ===")
        self.log.info("")
        for item in items:
            old = item.method
            made = read_method(old.weight_file, old.dsi, self.job, old.log)
            made.loader, made.weight_dir = old.loader, old.weight_dir
            made.n_train, made.train_time = old.n_train, old.train_time
            item.method = made

    # -- testing ----------------------------------------------------------------------------

    def TestAllMethods(self) -> None:
        """Every method's output over its loader's test sample."""
        self.log.header(f"{color('bold')}Test all methods{color('reset')}")
        if not self.booked:
            self.log.info("...nothing found to test")
            return
        for name in sorted(self.booked):
            for item in self.booked[name]:
                method = item.method
                words = {REGRESSION: "Regression", MULTICLASS: "Multiclass classification"}
                what = words.get(method.analysis, "Classification")
                self.log.header(f"Test method: {method.name} for {what} performance")
                self.log.info("")
                test = item.loader.dataset().test
                item.test_values = evaluate_sample(method, test, "testing", method.dsi.name)
                if method.analysis == CLASSIFICATION and method.mva_pdfs is not None:
                    values = np.asarray(item.test_values, dtype=np.float64)
                    item.test_proba = np.asarray(method.proba(values, 0.5), dtype=np.float32)
                    item.extra["prob_test"] = item.test_proba
                    method.log.info(
                        f"Dataset[{method.dsi.name}] : Evaluation of {method.name} on testing sample"
                    )
