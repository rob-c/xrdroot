"""Generation settings are registered when RooFit first creates a numeric sampler."""

from __future__ import annotations

import copy
from itertools import product
from typing import Any

from .categories import RooCategory
from .collections import RooArgSet
from .messages import ERROR, log
from .printing import RooPrintable, kName, kStandard, kTitle, kValue, kVerbose
from .variables import RooRealVar

#: The default starts empty; registering samplers mutates this same instance.
DEFAULT: list[RooNumGenConfig] = []
#: Unknown section lookups share ROOT's empty placeholder.
MISSING = RooArgSet()
#: Sampling, rather than configuration access, initializes the generator factory.
REGISTERED = [False]


def _methods() -> dict[tuple[str, bool, bool], RooCategory]:
    methods = {}
    for dimension, conditional, categories in product(
        ("1D", "2D", "ND"), (False, True), (False, True)
    ):
        name = "method" + dimension + ("Cond" if conditional else "")
        if categories and not conditional:
            name += "Cat"
        title = dimension + " sampling method"
        if conditional:
            title += " for conditional pfs"
            if categories:
                title += " with categories"
        elif categories:
            title += " for pdfs with categories"
        methods[dimension, conditional, categories] = RooCategory(name, title, {"N/A": 0})
    return methods


class RooNumGenConfig(RooPrintable):
    """Choose samplers independently for dimensions, conditions and category variables."""

    def __init__(self, other: RooNumGenConfig | None = None) -> None:
        self._methods = _methods()
        self._sections: list[RooArgSet] = []
        self._capabilities: dict[str, tuple[bool, bool]] = {}
        if other is not None:
            self.__dict__.update(copy.deepcopy(other.__dict__))

    @staticmethod
    def defaultConfig() -> RooNumGenConfig:
        if not DEFAULT:
            DEFAULT.append(RooNumGenConfig())
        return DEFAULT[0]

    def method1D(self, conditional: bool, categories: bool) -> RooCategory:
        return self._methods["1D", bool(conditional), bool(categories)]

    def method2D(self, conditional: bool, categories: bool) -> RooCategory:
        return self._methods["2D", bool(conditional), bool(categories)]

    def methodND(self, conditional: bool, categories: bool) -> RooCategory:
        return self._methods["ND", bool(conditional), bool(categories)]

    def addConfigSection(self, proto: Any, section: RooArgSet) -> bool:
        return self._register(
            str(proto.generatorName()),
            section,
            proto.canSampleConditional(),
            proto.canSampleCategories(),
        )

    def _register(self, name: str, section: RooArgSet, conditional: bool, categories: bool) -> bool:
        for (_, cond, cat), method in self._methods.items():
            if (not cond or conditional) and (not cat or categories):
                method.defineType(name)
        copied = section.snapshot()
        for variable in copied:
            variable.setAttribute("SnapShot_ExtRefClone")
        copied.SetName(name)
        self._sections.append(copied)
        self._capabilities[name] = (conditional, categories)
        return False

    def getConfigSection(self, name: str) -> RooArgSet:
        for section in self._sections:
            if section.GetName() == name:
                return section
        log(
            None,
            ERROR,
            "InputArguments",
            "RooNumGenConfig::getIntegrator: ERROR: "
            f"no configuration stored for integrator '{name}'",
        )
        return MISSING

    def defaultPrintStyle(self, option: Any) -> int:
        return kVerbose if "v" in str(option or "").lower() else kStandard

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        text = "\n"
        for dimension in ("1D", "2D", "ND"):
            text += self._method_text(dimension, indent)
        if verbose:
            text += "\nAvailable sampling methods:\n\n"
            for section in self._sections:
                text += self._section_text(section, indent)
        return text

    def _method_text(self, dimension: str, indent: str) -> str:
        regular = self._methods[dimension, False, False]
        text = f"{indent}{dimension[0]}-D sampling method: {regular.getLabel()}\n"
        for conditional, categories, label in (
            (False, True, "with categories"),
            (True, False, "conditional"),
            (True, True, "conditional with categories"),
        ):
            method = self._methods[dimension, conditional, categories]
            if method.getIndex() != regular.getIndex():
                text += f" ({method.getLabel()} if {label})\n"
        return text + "\n"

    def _section_text(self, section: RooArgSet, indent: str) -> str:
        name = section.GetName()
        text = f"{indent}*** {name} ***\n{indent}Capabilities: "
        for label, enabled in zip(
            ("Conditional", "Categories"), self._capabilities[name], strict=False
        ):
            if enabled:
                text += f"[{label}] "
        text += "\nConfiguration: \n"
        return text + section.printMultiline(kName | kValue | kTitle, False, "") + "\n"


#: ROOT 6.40.04 sampler defaults; these values need no ROOT installation.
PARAMETERS = {
    "RooAcceptReject": (
        ("nTrial0D", "Number of trial samples for cat-only generation", 100.0, 0.0, 1000000000.0),
        ("nTrial1D", "Number of trial samples for 1-dim generation", 1000.0, 0.0, 1000000000.0),
        ("nTrial2D", "Number of trial samples for 2-dim generation", 100000.0, 0.0, 1000000000.0),
        ("nTrial3D", "Number of trial samples for N-dim generation", 10000000.0, 0.0, 1000000000.0),
    ),
    "RooFoamGenerator": (
        ("nSample", "Number of samples per cell", 200.0, 0.0, 1000000.0),
        ("nCell1D", "Number of cells for 1-dim generation", 30.0, 0.0, 1000000.0),
        ("nCell2D", "Number of cells for 2-dim generation", 500.0, 0.0, 1000000.0),
        ("nCell3D", "Number of cells for 3-dim generation", 5000.0, 0.0, 1000000.0),
        ("nCellND", "Number of cells for N-dim generation", 10000.0, 0.0, 1000000.0),
        ("chatLevel", "TFOAM 'chat level' (verbosity)", 0.0, 0.0, 2.0),
    ),
}


def initialize_factory() -> None:
    """The first numerical generator registers its samplers on the existing default."""
    if REGISTERED[0]:
        return
    config = RooNumGenConfig.defaultConfig()
    for name, specs in PARAMETERS.items():
        section = RooArgSet(*[RooRealVar(*spec) for spec in specs])
        conditional = categories = name == "RooAcceptReject"
        config._register(name, section, conditional, categories)
    for (_, conditional, categories), method in config._methods.items():
        method.setLabel("RooAcceptReject" if conditional or categories else "RooFoamGenerator")
    REGISTERED[0] = True
