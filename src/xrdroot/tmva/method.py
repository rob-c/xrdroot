"""``TMVA::MethodBase``: what every method has - its options, transformations, weight file.

A method is booked by name with an option string; it declares its options
with their defaults (:attr:`Method.defaults`), and the string overrides
them. It is trained on the transformed training events, evaluated on
events it transforms the same way, and written to - and read back from - a
weight file in TMVA's ``MethodSetup`` layout: the Factory tests every
method only after reading it back, as TMVA does, so that what the Reader
applies later is exactly what was tested.
"""

from __future__ import annotations

import os
import time
from typing import Any, ClassVar

import numpy as np

from .dataset import DataSetInfo, Events
from .handler import TransformationHandler
from .log import Logger, color
from .options import Options
from .pdf import PDF, PDFSettings, settings
from .variables import VariableInfo
from .xmlfile import Node, number

__all__ = ["CLASSIFICATION", "MULTICLASS", "Method", "REGRESSION"]

#: ``Types::EAnalysisType``.
CLASSIFICATION, REGRESSION, MULTICLASS = 0, 1, 2
#: How each analysis type is written in a weight file.
ANALYSIS_NAMES = {
    CLASSIFICATION: "Classification",
    REGRESSION: "Regression",
    MULTICLASS: "Multiclass",
}
#: The options every method has, and their defaults.
BASE_DEFAULTS: dict[str, Any] = {
    "V": False,
    "VerbosityLevel": "Default",
    "VarTransform": "None",
    "H": False,
    "CreateMVAPdfs": False,
    "IgnoreNegWeightsInTraining": False,
}


def _typed(default: Any, options: Options, name: str) -> Any:
    """Option ``name`` as the type of its default."""
    if isinstance(default, bool):
        return options.flag(name, default)
    if isinstance(default, int):
        return options.integer(name, default)
    if isinstance(default, float):
        return options.number(name, default)
    return options.text_of(name, default)


def _option_text(value: Any) -> str:
    """An option's value as TMVA's weight file writes it: ``True``, ``850``, ``5.000000e-01``."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return format(value, "e")
    return str(value)


class Method:
    """The part of every TMVA method that is not the method's own algorithm."""

    #: ``GetMethodTypeName``: ``"BDT"``, ``"Likelihood"``...
    type_name: ClassVar[str] = ""
    #: The method's own options and their defaults, in the order TMVA declares them.
    defaults: ClassVar[dict[str, Any]] = {}
    #: The analyses it can do.
    analyses: ClassVar[frozenset[int]] = frozenset({CLASSIFICATION})
    #: The help TMVA prints for option ``H``, if any.
    help_text: ClassVar[str] = ""

    def __init__(
        self,
        job: str,
        title: str,
        dsi: DataSetInfo,
        options: Any = "",
        weight_dir: str = "",
        analysis: int = CLASSIFICATION,
    ) -> None:
        self.job, self.name, self.dsi = str(job), str(title), dsi
        self.options = Options(options)
        self.analysis = analysis
        self.log = Logger(self.name)
        self.handler = TransformationHandler(dsi, self.name)
        self.weight_dir = weight_dir
        #: The weight file the method was read from, if it was.
        self.source = ""
        #: Which sample - "training" or "testing" - the method is being evaluated on.
        self.sample_kind = ""
        #: ``BaseDir``: the Factory's making of the method's directory in the output file.
        self.base_directory: Any = None
        #: The Factory's output, for a method that writes as it trains.
        self.output: Any = None
        self.mva_pdfs: tuple[PDF, PDF] | None = None
        self.train_time = 0.0
        self.n_train = 0
        #: The loader the method was booked with, which the Factory sets.
        self.loader: Any = None
        #: The training events before any transformation, for a method that transforms per class.
        self.raw_train: Events | None = None
        self.option_values = {
            name: _typed(d, self.options, name) for name, d in self.all_defaults().items()
        }
        self.process_options()

    @classmethod
    def all_defaults(cls) -> dict[str, Any]:
        return {**BASE_DEFAULTS, **cls.defaults}

    def opt(self, name: str) -> Any:
        """The value of option ``name``: what the string said, or its default."""
        return self.option_values[name]

    def process_options(self) -> None:
        """``ProcessOptions``: whatever a method works out from its options once they are read."""

    # -- names ------------------------------------------------------------------------------

    def GetName(self) -> str:
        return self.name

    def GetMethodName(self) -> str:
        return self.name

    def GetMethodTypeName(self) -> str:
        return self.type_name

    @property
    def testvar(self) -> str:
        """``GetTestvarName``: the name of the method's output in the test tree, ``MVA_<name>``."""
        return f"MVA_{self.name}"

    @property
    def weight_file(self) -> str:
        """``GetWeightFileName``: ``<dataset>/weights/<job>_<name>.weights.xml``."""
        base = f"{self.job}_{self.name}.weights.xml"
        return f"{self.weight_dir.rstrip('/')}/{base}" if self.weight_dir else base

    # -- setup ------------------------------------------------------------------------------

    def setup(self) -> None:
        """``ProcessBaseOptions``: the transformations the method's ``VarTransform`` names."""
        declared = {name.lower(): name for name in self.all_defaults()}
        for name, before in self.options.repeated:
            # ``ParseOptions`` names the option as it was declared, not as it was written.
            spelled = declared.get(name, name)
            self.log.warning(f"Value for option {spelled} was previously set to {before}")
        self.handler.create(str(self.opt("VarTransform")), self.log)

    #: How often ``GetMvaValues`` says which sample it evaluates: twice, but once for a Category.
    evaluation_headers: ClassVar[int] = 2
    #: Does booking the method build the data set, as ``LD``'s ``ProcessOptions`` does?
    needs_data: ClassVar[bool] = False

    def booked(self) -> None:
        """What the method does once booked: build the data set, or say how it will train."""
        if self.needs_data and self.loader is not None:
            self.loader.dataset()

    def monitoring(self, output: Any, directory: str) -> None:
        """``WriteMonitoringHistosToFile``: anything the method writes of its training."""

    def has_mva_pdfs(self) -> bool:
        return bool(self.opt("CreateMVAPdfs"))

    def pdf_settings(self) -> PDFSettings:
        """The ``MVAPdf`` options of the classifier's output densities."""
        base = settings(self.options, "MVAPdf", PDFSettings())
        return settings(self.options, "MVAPdfSig", base)

    # -- the algorithm, which each method provides ----------------------------------------------

    def train(self, events: Events) -> None:
        raise NotImplementedError

    def evaluate(self, values: Any) -> Any:
        """The output for each row of already-transformed ``values``."""
        raise NotImplementedError

    def add_weights(self, node: Node) -> None:
        """``AddWeightsXMLTo``: the method's own ``<Weights>``."""
        raise NotImplementedError

    def read_weights(self, node: Any) -> None:
        """``ReadWeightsFromXML``."""
        raise NotImplementedError

    def ranking(self) -> tuple[str, list[tuple[str, float]]] | None:
        """``CreateRanking``: the ranking's title and each variable's value, or ``None``."""
        return None

    def mva(self, events: Events) -> Any:
        """``GetMvaValue`` of every event: transformed, then evaluated."""
        return self.evaluate(self.handler.apply(events).values)

    def error(self, events: Events) -> Any:
        """``GetMvaValue``'s error of every event, for the methods that give one."""
        return np.full(len(events), -1.0)

    # -- output densities ---------------------------------------------------------------------

    def proba(self, values: Any, signal_fraction: float = 0.5) -> Any:
        """``GetProba``: the chance of being signal, from the output densities."""
        if self.mva_pdfs is None:
            self.log.warning(
                f"Dataset[{self.dsi.name}] : <GetProba> MVA PDFs for Signal and Background "
                "don't exist"
            )
            return np.full(np.shape(values), -1.0)
        p_s, p_b = (pdf.value(values) for pdf in self.mva_pdfs)
        denominator = p_s * signal_fraction + p_b * (1 - signal_fraction)
        return np.where(
            denominator > 0, p_s * signal_fraction / np.where(denominator > 0, denominator, 1), -1
        )

    def rarity(self, values: Any, signal: bool = False) -> Any:
        """``GetRarity``: the integral of the reference density up to each value."""
        if self.mva_pdfs is None:
            self.log.warning(
                f"Dataset[{self.dsi.name}] : <GetRarity> Required MVA PDF for Signal or "
                'Background does not exist: select option "CreateMVAPdfs"'
            )
            return np.zeros(np.shape(values))
        pdf = self.mva_pdfs[0 if signal else 1]
        return np.array([pdf.integral_between(pdf.xmin, float(v)) for v in np.ravel(values)])

    # -- the weight file ------------------------------------------------------------------------

    def _general(self, root: Node) -> None:
        info = root.add("GeneralInfo")
        entries = [
            ("TMVA Release", "4.2.1 [262657]"),
            ("ROOT Release", "6.40/04 [403460]"),
            ("Creator", os.environ.get("USER", "xrdroot")),
            ("Date", time.strftime("%a %b %d %H:%M:%S %Y")),
            ("Host", "xrdroot"),
            ("Dir", os.getcwd()),
            ("Training events", str(self.n_train)),
            ("TrainingTime", number(self.train_time, 8)),
            ("AnalysisType", ANALYSIS_NAMES[self.analysis]),
        ]
        for name, value in entries:
            info.add("Info", name=name, value=value)

    def _options(self, root: Node) -> None:
        node = root.add("Options")
        for name, value in self.option_values.items():
            option = node.add(
                "Option", name=name, modified="Yes" if self.options.given(name) else "No"
            )
            option.text = _option_text(value)

    def _variables(self, root: Node, tag: str, infos: list[VariableInfo], index: str) -> None:
        count = {"Variables": "NVar", "Spectators": "NSpec", "Targets": "NTrgt"}[tag]
        block = root.add(tag, **{count: len(infos)})
        for position, info in enumerate(infos):
            block.add(
                tag[:-1],
                **{index: position},
                Expression=info.expression,
                Label=info.label,
                Title=info.title,
                Unit=info.unit,
                Internal=info.internal,
                Type=info.vartype,
                Min=number(info.minimum, 8),
                Max=number(info.maximum, 8),
            )

    def to_xml(self) -> Node:
        """``WriteStateToXML``: the whole weight file."""
        from .trafoxml import write_transformations

        root = Node("MethodSetup", Method=f"{self.type_name}::{self.name}")
        self._general(root)
        self._options(root)
        self._variables(root, "Variables", self.dsi.variables, "VarIndex")
        self._variables(root, "Spectators", self.dsi.spectators, "SpecIndex")
        classes = root.add("Classes", NClass=self.dsi.GetNClasses())
        for info in self.dsi.classes:
            classes.add("Class", Name=info.name, Index=info.number)
        if self.analysis == REGRESSION:
            self._variables(root, "Targets", self.dsi.targets, "TargetIndex")
        write_transformations(root, self.handler)
        pdfs = root.add("MVAPdfs")
        for pdf in self.mva_pdfs or ():
            pdf.add_xml(pdfs)
        self.add_weights(root)
        return root

    def write_weight_file(self) -> None:
        """``WriteStateToFile``: the weight file written, and said so as TMVA says it."""
        directory = os.path.dirname(self.weight_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.log.info(
            f"Creating xml weight file: {color('lightblue')}{self.weight_file}{color('reset')}"
        )
        self.to_xml().write(self.weight_file)
