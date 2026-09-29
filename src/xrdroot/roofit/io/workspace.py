"""A ``RooWorkspace`` read from a file, as the engine's :class:`~..workspace.RooWorkspace`.

Its nodes, its datasets and embedded datasets, its snapshots, its named sets
and its generic objects - a ``ModelConfig`` - are made in that order, the
named sets and the ModelConfig pointing at the nodes the workspace holds.
Its code repository is read first: code for classes this engine has not got
is refused by their names, since running it would mean compiling C++.
"""

from __future__ import annotations

from typing import Any

from ...errors import UnsupportedFeatureError
from .build import MAKERS, Builder, maker, name_of, title_of
from .stream import Streamed

__all__ = ["read_workspace"]


def _check_code(record: Any) -> None:
    """Refuse a workspace whose code repository has classes this engine does not."""
    if record is None:
        return
    classes = [one[0] for one in record.get("relations") or ()]
    classes += [one[0] for one in record.get("files") or () if one[0] not in classes]
    missing = [name for name in classes if name not in MAKERS]
    if missing:
        raise UnsupportedFeatureError(
            f"the workspace carries the C++ code of {', '.join(missing)}, classes of its "
            "author's that this engine has not got and will not compile"
        )


@maker("RooWorkspace")
def _workspace(builder: Builder, record: Streamed) -> Any:
    from ..workspace import RooWorkspace

    _check_code(record.get("_classes"))
    made = RooWorkspace(name_of(record), title_of(record))
    builder.made[id(record)] = made  # a ModelConfig inside points back at it
    for node in builder.nodes(record.get("_allOwnedNodes").get("_list")):
        made._nodes.setdefault(node.GetName(), node)
    for data in builder.nodes(record.get("_dataList").get("items")):
        made._data[data.GetName()] = data
    for data in builder.nodes(record.get("_embeddedDataList").get("items")):
        made._embedded[data.GetName()] = data
    for snapshot in record.get("_snapshots").get("items") or ():
        made._snapshots[str(snapshot.get("_name"))] = builder.nodes(snapshot.get("_list"))
    for name, content in (record.get("_namedSets") or {}).items():
        made.defineSet(str(name), builder.nodes(content.get("_list")))
    for generic in record.get("_genObjects").get("items") or ():
        held = generic.get("_list").get("items")[0] if generic.cls == "RooTObjWrap" else generic
        obj = builder.node(held)
        made._generic[obj.GetName()] = obj
    return made


@maker("RooStats::ModelConfig")
def _model_config(builder: Builder, record: Streamed) -> Any:
    """The names of the density and the sets, in the workspace being read."""
    from ...roostats.modelconfig import ModelConfig

    made = ModelConfig(name_of(record), title_of(record))
    for key, member in (("Pdf", "fPdfName"), ("PriorPdf", "fPriorPdfName"),
                        ("ProtoData", "fProtoDataName"), ("Snapshot", "fSnapshotName"),
                        ("POI", "fPOIName"), ("NuisParams", "fNuisParamsName"),
                        ("ConstrainedParams", "fConstrParamsName"),
                        ("Observables", "fObservablesName"),
                        ("ConditionalObservables", "fConditionalObsName"),
                        ("GlobalObservables", "fGlobalObsName"),
                        ("ExternalConstraints", "fExtConstraintsName")):  # fmt: skip
        made._names[key] = str(record.get(member) or "")
    return made


def read_workspace(buf: Any, infos: Any) -> Any:
    """The workspace in ``buf``, the key's bytes, by the file's streamer information."""
    from . import custom, data, models, variables  # noqa: F401 - the makers, by importing them
    from .stream import Reader

    record = Reader(buf, infos).object("RooWorkspace")
    builder = Builder()
    made = builder.node(record)
    for obj in made._generic.values():
        if hasattr(obj, "ReplaceWS"):
            obj.ReplaceWS(made)
    return made
