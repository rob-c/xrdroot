"""Numeric-integration settings are copied independently of the functions using them.

ROOT keeps one mutable global default and optional private copies on individual
functions. An explicitly constructed empty configuration has only N/A methods;
copying the default includes its registered integrators and parameter sections.
"""

from __future__ import annotations

import copy
from typing import Any

from .categories import RooCategory
from .collections import RooArgSet
from .messages import ERROR, log
from .numintdefaults import PLUGINS
from .printing import RooPrintable, g, kName, kStandard, kValue, kVerbose
from .variables import RooRealVar

__all__ = ["RooNumIntConfig"]

#: The global configuration is initialized lazily, after the model classes load.
DEFAULT: list[RooNumIntConfig] = []
#: ROOT returns one shared empty section for an unknown integrator.
MISSING = RooArgSet()
#: Each closed-domain category and its title; open-domain categories follow them.
DIMENSIONS = (("1D", "1D"), ("2D", "2D"), ("ND", "ND"))


def _methods() -> dict[str, RooCategory]:
    methods = {}
    for suffix in ("", "Open"):
        for dimension, title in DIMENSIONS:
            name = f"method{dimension}{suffix}"
            label = f"{title} integration method" + (" in open domain" if suffix else "")
            methods[name] = RooCategory(name, label, {"N/A": 0})
    return methods


def _parameter(spec: Any) -> Any:
    name, title, value = spec
    if not isinstance(value, tuple):
        return RooRealVar(name, title, value)
    states, selected = value
    made = RooCategory(name, title, dict(states))
    made.setIndex(selected)
    return made


def _registered() -> RooNumIntConfig:
    made = RooNumIntConfig()
    for name, capabilities, dependency, parameters in PLUGINS:
        section = RooArgSet(*[_parameter(spec) for spec in parameters])
        made.addConfigSection(name, section, *capabilities)
        made._dependencies[name] = dependency
    for dimension, label in (
        ("1D", "RooIntegrator1D"),
        ("2D", "RooAdaptiveIntegratorND"),
        ("ND", "RooAdaptiveIntegratorND"),
    ):
        made._methods[f"method{dimension}"].setLabel(label)
    made.method1DOpen().setLabel("RooImproperIntegrator1D")
    return made


class RooNumIntConfig(RooPrintable):
    """The precision, method choices and per-integrator parameters of numeric integrals."""

    def __init__(self, other: RooNumIntConfig | None = None) -> None:
        self._eps_abs = self._eps_rel = 1e-7
        self._counter = False
        self._methods = _methods()
        self._sections: list[RooArgSet] = []
        self._capabilities: dict[str, tuple[bool, ...]] = {}
        self._dependencies: dict[str, str] = {}
        if other is not None:
            self.__dict__.update(copy.deepcopy(other.__dict__))

    @staticmethod
    def defaultConfig() -> RooNumIntConfig:
        if not DEFAULT:
            DEFAULT.append(_registered())
        return DEFAULT[0]

    def epsAbs(self) -> float:
        return self._eps_abs

    def epsRel(self) -> float:
        return self._eps_rel

    def _precision(self, which: str, value: float) -> None:
        if value < 0:
            log(
                None,
                ERROR,
                "InputArguments",
                f"RooNumIntConfig::setEps{which}: ERROR: "
                "target absolute precision must be greater or equal than zero",
            )
            return
        setattr(self, "_eps_" + which.lower(), float(value))

    def setEpsAbs(self, value: float) -> None:
        self._precision("Abs", value)

    def setEpsRel(self, value: float) -> None:
        self._precision("Rel", value)

    def printEvalCounter(self) -> bool:
        return self._counter

    def setPrintEvalCounter(self, enabled: bool) -> None:
        self._counter = bool(enabled)

    def method1D(self) -> RooCategory:
        return self._methods["method1D"]

    def method2D(self) -> RooCategory:
        return self._methods["method2D"]

    def methodND(self) -> RooCategory:
        return self._methods["methodND"]

    def method1DOpen(self) -> RooCategory:
        return self._methods["method1DOpen"]

    def method2DOpen(self) -> RooCategory:
        return self._methods["method2DOpen"]

    def methodNDOpen(self) -> RooCategory:
        return self._methods["methodNDOpen"]

    def addConfigSection(
        self, name: str, section: RooArgSet, one: bool, two: bool, many: bool, opened: bool
    ) -> bool:
        capabilities = (one, two, many, opened)
        for (dimension, _), capable in zip(DIMENSIONS, capabilities, strict=False):
            if capable:
                self._methods[f"method{dimension}"].defineType(name)
                if opened:
                    self._methods[f"method{dimension}Open"].defineType(name)
        copied = section.snapshot()
        for parameter in copied:
            parameter.setAttribute("SnapShot_ExtRefClone")
        copied.SetName(name)
        self._sections.append(copied)
        self._capabilities[name] = capabilities
        return False

    def getConfigSection(self, name: str) -> RooArgSet:
        for section in self._sections:
            if section.GetName() == name:
                return section
        log(
            None,
            ERROR,
            "InputArguments",
            "RooNumIntConfig::getConfigSection: ERROR: "
            f"no configuration stored for integrator '{name}'",
        )
        return MISSING

    def defaultPrintStyle(self, option: Any) -> int:
        return kVerbose if "v" in str(option or "").lower() else kStandard

    def printMultiline(self, contents: int, verbose: bool, indent: str) -> str:
        text = f"{indent}Requested precision: {g(self._eps_abs)} absolute, "
        text += f"{g(self._eps_rel)} relative\n\n"
        if self._counter:
            text += f"{indent}Printing of function evaluation counter "
            text += "for each integration enabled\n\n"
        text += "".join(self._method_line(dimension, indent) for dimension, _ in DIMENSIONS)
        if verbose:
            text += "\nAvailable integration methods:\n\n"
            text += "".join(self._section_text(section, indent) for section in self._sections)
        return text

    def _method_line(self, dimension: str, indent: str) -> str:
        closed, opened = (self._methods[f"method{dimension}{suffix}"] for suffix in ("", "Open"))
        text = f"{indent}{dimension[0]}-D integration method: {closed.getLabel()}"
        if closed.getIndex() != opened.getIndex():
            text += f" ({opened.getLabel()} if open-ended)"
        return text + "\n"

    def _section_text(self, section: RooArgSet, indent: str) -> str:
        name = section.GetName()
        labels = ("1-D", "2-D", "N-D", "OpenEnded")
        caps = "".join(
            f"[{label}] "
            for label, enabled in zip(labels, self._capabilities[name], strict=False)
            if enabled
        )
        text = f"{indent}*** {name} ***\n{indent}Capabilities: {caps}\nConfiguration: \n"
        text += section.printMultiline(kName | kValue, False, "")
        dependency = self._dependencies.get(name, "")
        if dependency:
            text += f"{indent}(Depends on '{dependency}')\n"
        return text + "\n"
