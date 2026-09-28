"""``TMVA::Factory``: booking, training, testing and evaluating methods, and writing it all down.

``Factory(job, outputFile, options)`` is TMVA's; ``BookMethod`` makes a
method from its type (``TMVA::Types::kBDT`` or ``"BDT"``), its title and
its option string; ``TrainAllMethods`` trains each on its loader's training
sample and writes its weight file, then - as TMVA does - reads every method
back from that file; ``TestAllMethods`` evaluates the read-back methods on
the test sample; and ``EvaluateAllMethods`` works out and prints TMVA's
tables, and writes the test and training trees. What is printed along the
way is TMVA's own messages, in TMVA's order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .efficiency import Efficiencies
from .evaluating import Evaluating
from .evaluation import ClassifierTest
from .loader import DataLoader
from .log import CONFIG, Logger, color
from .method import CLASSIFICATION, MULTICLASS, REGRESSION, Method
from .options import Options
from .output import Output
from .training import Training
from .types import METHOD_NAMES
from .weightfile import method_class

__all__ = ["Booked", "Factory"]

#: ``AnalysisType``'s values, by the lower-cased word.
ANALYSES = {"classification": CLASSIFICATION, "regression": REGRESSION, "multiclass": MULTICLASS}
#: What ``Train method:`` says each analysis is.
ANALYSIS_WORDS = {
    CLASSIFICATION: "Classification",
    REGRESSION: "Regression",
    MULTICLASS: "Multiclass classification",
}
#: The fewest training events a method is trained with.
MIN_TRAINING_EVENTS = 10
#: The ROOT release, and its date, a Factory's greeting names.
ROOT_VERSION, ROOT_DATE = "6.40.04", "Aug 27, 2026"
#: TMVA's release, and its date.
TMVA_VERSION = "4.2.1, Feb 5, 2015"
#: ``kLogoWelcomeMsg``: the logo a Factory greets with.
LOGO = (
    "",
    "_/_/_/_/_/ _|      _|  _|      _|    _|_|   ",
    "   _/      _|_|  _|_|  _|      _|  _|    _| ",
    "  _/       _|  _|  _|  _|      _|  _|_|_|_| ",
    " _/        _|      _|    _|  _|    _|    _| ",
    "_/         _|      _|      _|      _|    _| ",
    "",
)


@dataclass
class Booked:
    """A booked method, its loader, and what training and testing it left behind."""

    method: Method
    loader: DataLoader
    train_values: Any = None
    test_values: Any = None
    test_proba: Any = None
    test: ClassifierTest | None = None
    efficiencies: Efficiencies | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _method_name(method: Any) -> str:
    """``TMVA::Types::kBDT`` or ``"BDT"`` - the method's type name either way."""
    if isinstance(method, int):
        return METHOD_NAMES[method] if 0 <= method < len(METHOD_NAMES) else str(method)
    return str(method)


class Factory(Training, Evaluating):
    """``TMVA::Factory(jobName, outputFile, options)`` - or ``(jobName, options)``, with no file."""

    def __init__(self, job: Any, *args: Any) -> None:
        target, options = (None, args[0]) if len(args) == 1 else ((args + (None, ""))[:2])
        if isinstance(target, str):
            target, options = None, target
        self.job = str(job)
        self.output = Output(target)
        self.options = Options(options or "")
        CONFIG.use_color = self.options.flag("Color", False)
        CONFIG.silent = self.options.flag("Silent", False)
        CONFIG.draw_progress_bar = self.options.flag("DrawProgressBar", True)
        self.transformations = self.options.text_of("Transformations", "I")
        self.correlations = self.options.flag("Correlations", False)
        self.roc = self.options.flag("ROC", True)
        self.persistence = self.options.flag("ModelPersistence", True)
        word = self.options.text_of("AnalysisType", "Auto").lower()
        self.analysis = ANALYSES.get(word, -1)
        self.log = Logger("Factory")
        self.booked: dict[str, list[Booked]] = {}
        self.last_rows: dict[str, list[dict[str, Any]]] = {}
        if target is None:
            self._greet()

    def _greet(self) -> None:
        """``Greetings``: what a Factory made without a file says first, logo and all."""
        self.log.header(f"You are running ROOT Version: {ROOT_VERSION}, {ROOT_DATE}")
        for line in LOGO:
            self.log.info(line)
        self.log.info(f"___________TMVA Version {TMVA_VERSION}")
        self.log.info("")

    # -- booking --------------------------------------------------------------------------

    def _analysis_for(self, loader: DataLoader) -> int:
        """The analysis type, worked out from the loader's classes when ``AnalysisType=Auto``."""
        if self.analysis >= 0:
            return self.analysis
        info = loader.info
        if (
            info.GetNClasses() == 2
            and info.GetClassInfo("Signal")
            and info.GetClassInfo("Background")
        ):
            self.analysis = CLASSIFICATION
        elif info.GetNClasses() >= 2:
            self.analysis = MULTICLASS
        else:
            raise self.log.fatal(
                f"No analysis type for {info.GetNClasses()} classes and "
                f"{info.GetNTargets()} regression targets."
            )
        return self.analysis

    def BookMethod(
        self, loader: DataLoader, method: Any, title: Any, options: Any = ""
    ) -> Method | None:
        """Book a method of type ``method`` called ``title`` with ``options``."""
        name, title = _method_name(method), str(title)
        analysis = self._analysis_for(loader)
        if self.HasMethod(loader.GetName(), title):
            raise self.log.fatal(
                f"Booking failed since method with title <{title}> already exists in with "
                f"DataSet Name <{loader.GetName()}>  "
            )
        self.log.header(f"Booking method: {color('bold')}{title}{color('reset')}")
        self.log.info("")
        kind = method_class(name)
        weight_dir = f"{loader.GetName()}/weights" if self.persistence else ""
        made = kind(self.job, title, loader.info, str(options), weight_dir, analysis)
        made.loader = loader
        if analysis not in made.analyses:
            self._incapable(made, loader, analysis)
            return None
        made.setup()
        made.booked()
        self.booked.setdefault(loader.GetName(), []).append(Booked(made, loader))
        return made

    def _incapable(self, method: Method, loader: DataLoader, analysis: int) -> None:
        what = {
            REGRESSION: f"regression with {loader.info.GetNTargets()} targets.",
            MULTICLASS: f"multiclass classification with {loader.info.GetNClasses()} classes.",
        }.get(analysis, f"classification with {loader.info.GetNClasses()} classes.")
        self.log.warning(f"Method {method.type_name} is not capable of handling {what}")

    def HasMethod(self, dataset: Any, title: Any) -> bool:
        return self.GetMethod(dataset, title) is not None

    def GetMethod(self, dataset: Any, title: Any) -> Method | None:
        found = self._booked(dataset, title)
        return None if found is None else found.method

    def _booked(self, dataset: Any, title: Any) -> Booked | None:
        name = dataset.GetName() if hasattr(dataset, "GetName") else str(dataset)
        for item in self.booked.get(name, []):
            if item.method.name == str(title):
                return item
        return None

    def SetVerbose(self, verbose: bool = True) -> None:
        self.options = Options(f"{self.options.text}:{'V' if verbose else '!V'}")

    def DeleteAllMethods(self) -> None:
        self.booked = {}
