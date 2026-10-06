"""A method's transformations in its weight file: ``<Transformations>``, written and read.

Each transformation is written as TMVA writes it - the normalisation's
ranges, the decorrelation's matrices, the principal components' means and
eigenvectors, the Gaussianisation's cumulative densities - once per class
and once for all classes, and at TMVA's precision, which is what TMVA reads
back to test and apply the method with. Reading takes TMVA's files and these
alike.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import DataSetInfo
from .handler import TransformationHandler
from .log import Logger
from .transforms import PCA, Decorrelate, Identity, Normalize, Transform
from .xmlfile import Node, children, floats, number

__all__ = ["read_transformations", "write_transformations"]


def _selection(parent: Node, dsi: DataSetInfo, with_targets: bool) -> None:
    items = [("Variable", info) for info in dsi.variables]
    if with_targets:
        items += [("Target", info) for info in dsi.targets]
    selection = parent.add("Selection")
    for side in ("Input", "Output"):
        block = selection.add(side, **{f"N{side}s": len(items)})
        for kind, info in items:
            block.add(side, Type=kind, Label=info.label, Expression=info.expression)


def _class_names(dsi: DataSetInfo, count: int) -> list[str]:
    names = [info.name for info in dsi.classes]
    return [*names, "Combined"][:count] if count > 1 else names[:count] or ["Combined"]


def _write_one(parent: Node, transform: Transform, dsi: DataSetInfo) -> None:
    if isinstance(transform, Normalize):
        node = parent.add("Transform", Name="Normalize", UseOffsetOrNot="UseOffset")
        _selection(node, dsi, bool(dsi.targets))
        for index, (low, high) in enumerate(transform.params):
            ranges = node.add("Class", ClassIndex=index).add("Ranges")
            for row, (a, b) in enumerate(zip(low, high, strict=False)):
                ranges.add("Range", Index=row, Min=number(a), Max=number(b))
    elif isinstance(transform, Decorrelate):
        node = parent.add("Transform", Name="Decorrelation")
        _selection(node, dsi, False)
        for matrix in transform.params:
            size = len(matrix)
            node.add("Matrix", Rows=size, Columns=size).block(np.ravel(matrix), 15)
    elif isinstance(transform, PCA):
        _write_pca(parent, transform, dsi)
    elif transform.xml_name in ("Gauss", "Uniform"):
        _write_gauss(parent, transform, dsi)
    else:
        _selection(parent.add("Transform", Name="Id"), dsi, False)


def _write_pca(parent: Node, transform: PCA, dsi: DataSetInfo) -> None:
    node = parent.add("Transform", Name="PCA")
    _selection(node, dsi, False)
    names = _class_names(dsi, len(transform.params))
    for index, (mean, _) in enumerate(transform.params):
        node.add("Statistics", Class=names[index], ClassIndex=index, NRows=len(mean)).block(mean)
    for index, (mean, vectors) in enumerate(transform.params):
        size = len(mean)
        node.add(
            "Eigenvectors", Class=names[index], ClassIndex=index, NRows=size, NCols=size
        ).block(np.ravel(vectors))


def _write_gauss(parent: Node, transform: Transform, dsi: DataSetInfo) -> None:
    flat = "Flat" if transform.xml_name == "Uniform" else "Gauss"
    node = parent.add("Transform", Name="Gauss", FlatOrGauss=flat)
    _selection(node, dsi, False)
    for index in range(dsi.GetNVariables()):
        variable = node.add("Variable", VarIndex=index)
        for cls, pdfs in enumerate(transform.params):
            pdfs[index].add_xml(variable.add(f"CumulativePDF_cls{cls}"))


def write_transformations(parent: Node, handler: TransformationHandler) -> None:
    """``TransformationHandler::AddXMLTo``."""
    node = parent.add("Transformations", NTransformations=len(handler.transforms))
    for transform in handler.transforms:
        _write_one(node, transform, handler.dsi)


def _read_normalize(node: Any) -> Normalize:
    made = Normalize()
    for cls in children(node, "Class"):
        ranges = children(cls.find("Ranges"), "Range")  # type: ignore[arg-type]
        low = np.array([float(r.get("Min", 0)) for r in ranges], dtype=np.float32)
        high = np.array([float(r.get("Max", 0)) for r in ranges], dtype=np.float32)
        made.params.append((low, high))
    return made


def _read_decorrelation(node: Any) -> Decorrelate:
    made = Decorrelate()
    for matrix in children(node, "Matrix"):
        rows, columns = int(matrix.get("Rows", 0)), int(matrix.get("Columns", 0))
        made.params.append(np.array(floats(matrix)).reshape(rows, columns))
    return made


def _read_pca(node: Any) -> PCA:
    made = PCA()
    means = [np.array(floats(item)) for item in children(node, "Statistics")]
    for mean, item in zip(means, children(node, "Eigenvectors"), strict=False):
        rows, columns = int(item.get("NRows", 0)), int(item.get("NCols", 0))
        made.params.append((mean, np.array(floats(item)).reshape(rows, columns)))
    return made


def _read_gauss(node: Any) -> Transform:
    from .gauss import Gauss, Uniform
    from .pdf import pdf_from_xml

    made: Transform = Uniform() if node.get("FlatOrGauss") == "Flat" else Gauss()
    per_class: list[list[Any]] = []
    for variable in children(node, "Variable"):
        for cls, holder in enumerate(item for item in variable if item.tag.startswith("Cumul")):
            if cls >= len(per_class):
                per_class.append([])
            per_class[cls].append(pdf_from_xml(holder.find("PDF"), normalise=False))
    made.params = per_class
    return made


#: How each transformation TMVA writes is read, by the name it writes it under.
READERS = {
    "Normalize": _read_normalize,
    "Decorrelation": _read_decorrelation,
    "PCA": _read_pca,
    "Gauss": _read_gauss,
}


def read_transformations(node: Any, handler: TransformationHandler) -> None:
    """``TransformationHandler::ReadFromXML``: every transformation, ready to apply."""
    for item in children(node, "Transform"):
        name = str(item.get("Name"))
        reader = READERS.get(name)
        if reader is None and name not in ("Id", "Identity"):
            raise Logger("TransformationHandler").fatal(
                f"<ReadFromXML> Variable transform '{name}' unknown."
            )
        transform = Identity() if reader is None else reader(item)
        if reader is None:
            transform.params = [None]
        handler.transforms.append(transform)
