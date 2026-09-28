"""``TMVA::Experimental::Classification``: the Envelope that trains and tests booked classifiers.

Each method is loaded as its booking says - ``Loading booked method: BDT
BDTG`` - then trained and tested without a word, as TMVA's workers are, and
the envelope ends with each one's ROC integral. With ``Jobs`` above one TMVA
trains the methods in parallel processes and lists them as they come back;
here they are trained one after another and listed fastest first, which is
the order parallel workers would finish in.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .evaluation import roc_curve, roc_integral
from .log import CONFIG, Logger
from .method import CLASSIFICATION, Method
from .options import Options
from .tools import CxxVector

__all__ = ["Classification", "ClassificationResult"]

#: The rule of the envelope's table.
RULE = "--------------------------------------------------- :"


class ClassificationResult:
    """``ClassificationResult``: one method's test outputs, its ROC integral and curve."""

    def __init__(
        self, loader: str, kind: str, title: str, values: Any, signal: Any, weights: Any
    ) -> None:
        self.loader, self.kind, self.title = loader, kind, title
        self.values, self.signal, self.weights = values, signal, weights
        #: The efficiency area a Cuts method's ROC integral is, as its table gives it.
        self.area: float | None = None

    def GetDataLoaderName(self) -> str:
        return self.loader

    def GetMethodName(self) -> str:
        return self.kind

    def GetMethodTitle(self) -> str:
        return self.title

    def IsCutsMethod(self) -> bool:
        return self.kind == "Cuts"

    def GetROCIntegral(self, iClass: int = 0, type: int = 1) -> float:
        if self.area is not None:
            return self.area
        return roc_integral(self.values, self.signal, self.weights)

    def GetROCGraph(self, fLegend: bool = True, iClass: int = 0, type: int = 1) -> Any:
        from ..pyroot.core.graphs import TGraph

        sensitivity, specificity = roc_curve(self.values, self.signal, self.weights)
        graph = TGraph(len(sensitivity), np.asarray(sensitivity), np.asarray(specificity))
        graph.SetName(self.title)
        graph.SetTitle(self.title)
        return graph


class Classification:
    """``Classification(dataloader, options)``: ``Jobs=N`` and the Factory's options."""

    def __init__(self, loader: Any, options: Any = "", *_: Any) -> None:
        self.loader, self.options = loader, Options(options)
        self.jobs = self.options.integer("Jobs", 1)
        CONFIG.use_color = self.options.flag("Color", False)
        CONFIG.silent = self.options.flag("Silent", False)
        self.log = Logger("")
        self.methods: list[tuple[str, str, str]] = []
        self.results: list[ClassificationResult] = []

    def BookMethod(self, method: Any, title: Any, options: Any = "") -> None:
        from .factory import _method_name

        self.methods.append((_method_name(method), str(title), str(options)))

    def _load(self, kind: str, title: str, options: str) -> Method:
        """``GetMethod``: the method booked as its options say, with what its booking prints."""
        from .weightfile import method_class

        self.log.header(f"Loading booked method: {kind} {title}")
        self.log.info("")
        made = method_class(kind)(
            self.loader.GetName(), title, self.loader.info, options, "", CLASSIFICATION
        )
        made.loader = self.loader
        made.setup()
        made.booked()
        return made

    def _run(self, method: Method) -> tuple[float, ClassificationResult]:
        """``TrainMethod`` and ``TestMethod``, silently: the result and the time it took."""
        start = time.perf_counter()
        silent, CONFIG.silent = CONFIG.silent, True
        try:
            dataset = self.loader.dataset()
            train = dataset.train
            method.raw_train = train
            method.train(method.handler.prepare(train) if method.handler.transforms else train)
            test = dataset.test
            if method.type_name == "Cuts":
                method.test_signal_eff = -1.0
            values = np.asarray(method.mva(test), dtype=np.float32)
            special = getattr(method, "classifier_evaluation", None)
            area = special(test, train)[0]["area"] if special is not None else None
        finally:
            CONFIG.silent = silent
        signal = test.classes == self.loader.info.GetSignalClassIndex()
        result = ClassificationResult(
            self.loader.GetName(), method.type_name, method.name, values, signal, test.weights
        )
        result.area = area
        return time.perf_counter() - start, result

    def Evaluate(self) -> None:
        """Every method loaded, trained and tested; each one's ROC integral in TMVA's table."""
        start = time.perf_counter()
        loaded = [self._load(*booked) for booked in self.methods]
        timed = [self._run(method) for method in loaded]
        if self.jobs > 1:
            timed.sort(key=lambda pair: pair[0])
        self.results = [result for _, result in timed]
        log = self.log
        log.info(RULE)
        log.info("DataSet              MVA                            :")
        log.info("Name:                Method/Title:    ROC-integ     :")
        log.info(RULE)
        for result in self.results:
            label = f"{result.GetMethodName()}/{result.GetMethodTitle()}"
            log.info(
                f"{result.GetDataLoaderName():<20} {label:<15}  "
                f"{result.GetROCIntegral():#1.3f}         :"
            )
        log.info(RULE)
        log.info("-----------------------------------------------------")
        log.header("Evaluation done.")
        log.info("")
        log.info(f"Jobs = {self.jobs} Real Time = {time.perf_counter() - start:f} ")
        log.info("-----------------------------------------------------")
        log.info("Evaluation done.")

    def GetResults(self) -> CxxVector:
        return CxxVector(self.results)

    def GetResult(self, name: Any, title: Any) -> ClassificationResult:
        for result in self.results:
            if (result.GetMethodName(), result.GetMethodTitle()) == (str(name), str(title)):
                return result
        raise self.log.fatal(f"Method {name}/{title} not found in the results.")
