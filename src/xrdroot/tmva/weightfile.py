"""Reading a method back from its weight file: ``MethodBase::ReadStateFromFile``.

The ``Method="BDT::BDT"`` attribute says which method it is and what it was
called; the options are read back into an option string, the variables and
classes into a :class:`~.dataset.DataSetInfo` when the reader has none of
its own, and then the transformations, the output densities and the
method's own weights. This reads what the Factory here writes, and TMVA's
own files for the methods whose weights it knows how to read.
"""

from __future__ import annotations

from typing import Any

from .dataset import DataSetInfo
from .log import Logger, color
from .method import ANALYSIS_NAMES, CLASSIFICATION, Method
from .pdf import pdf_from_xml
from .trafoxml import read_transformations
from .variables import VariableInfo
from .xmlfile import children, load

__all__ = ["method_class", "options_text", "read_method", "read_variables"]


def options_text(root: Any) -> str:
    """The ``<Options>`` written back into the option string they came from."""
    parts = []
    for option in (
        children(root.find("Options"), "Option") if root.find("Options") is not None else []
    ):
        name, value = str(option.get("name")), (option.text or "").strip()
        if option.get("size") is not None:
            continue
        if value in ("True", "False"):
            parts.append(name if value == "True" else f"!{name}")
        else:
            parts.append(f"{name}={value}")
    return ":".join(parts)


def read_variables(root: Any, tag: str) -> list[VariableInfo]:
    """The variables, spectators or targets a weight file declares."""
    block = root.find(tag)
    made = []
    for item in children(block) if block is not None else []:
        info = VariableInfo(
            f"{item.get('Label')}:={item.get('Expression')}",
            item.get("Title", ""),
            item.get("Unit", ""),
            str(item.get("Type", "F"))[:1],
        )
        info.internal = str(item.get("Internal", info.internal))
        info.minimum = float(item.get("Min", 0.0))
        info.maximum = float(item.get("Max", 0.0))
        made.append(info)
    return made


def _dataset_info(root: Any, dsi: DataSetInfo | None) -> DataSetInfo:
    """The reader's declarations, or the file's; its classes added if the reader has none."""
    made = dsi or DataSetInfo("Default")
    if not made.variables:
        made.variables = read_variables(root, "Variables")
        made.spectators = read_variables(root, "Spectators")
    if not made.targets:
        made.targets = read_variables(root, "Targets")
    classes = root.find("Classes")
    if not made.classes and classes is not None:
        for item in children(classes, "Class"):
            made.AddClass(str(item.get("Name")))
    return made


def method_class(type_name: str) -> type[Method]:
    """The class of the method a weight file is of, or the refusal of one there is not."""
    from .methods import REGISTRY

    found = REGISTRY.get(type_name)
    if found is None:
        raise Logger("Reader").fatal(
            f"The weight file is of a {type_name} method, which xrdroot's TMVA does not have"
        )
    return found


def _analysis(root: Any) -> int:
    general = root.find("GeneralInfo")
    for item in children(general) if general is not None else []:
        if item.get("name") == "AnalysisType":
            names = {name.lower(): number for number, name in ANALYSIS_NAMES.items()}
            return names.get(str(item.get("value")).lower(), CLASSIFICATION)
    return CLASSIFICATION


def read_method(
    path: str, dsi: DataSetInfo | None = None, job: str = "", log: Logger | None = None
) -> Method:
    """The method a weight file holds, read back ready to evaluate."""
    (log or Logger("MethodBase")).info(
        f"Reading weight file: {color('lightblue')}{path}{color('reset')}"
    )
    root = load(path)
    type_name, _, name = str(root.get("Method")).partition("::")
    kind = method_class(type_name)
    info = _dataset_info(root, dsi)
    method = kind(job, name, info, options_text(root), "", _analysis(root))
    transformations = root.find("Transformations")
    if transformations is not None:
        read_transformations(transformations, method.handler)
    pdfs = children(root.find("MVAPdfs")) if root.find("MVAPdfs") is not None else []
    if len(pdfs) == 2:
        method.mva_pdfs = (pdf_from_xml(pdfs[0]), pdf_from_xml(pdfs[1]))
    method.read_weights(root.find("Weights"))
    return method
