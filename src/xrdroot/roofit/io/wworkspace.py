"""A workspace key and its ProcessID preserve proxy references in native ROOT files."""

from __future__ import annotations

from typing import Any

from ...buffer import Buffer
from ...errors import UnsupportedFeatureError
from ...file import Key
from .layouts import descriptions
from .processids import SESSION
from .records import linked, named, put, record
from .stream import Streamed
from .wdata import dataset
from .wnodes import Nodes
from .wstream import Writer


class Payload:
    """The generic file writer supplies the key length needed by object references."""

    def __init__(self, value: Streamed, title: str) -> None:
        self.value = value
        self.classname = value.cls
        self.title = title
        self.process_id_needed = True

    def root_payload(self, keylen: int) -> tuple[bytes, tuple[str, ...], dict[Any, bytes]]:
        writer = Writer(keylen)
        writer.object(self.value, self.classname)
        return bytes(writer.buf.data), (), descriptions()


def workspace(value: Any, process_id: int) -> Payload:
    if value._generic or value._embedded:
        raise UnsupportedFeatureError(
            "Writing workspace generic or embedded objects is not implemented. "
            "Save a workspace containing supported model nodes and RooDataSets instead."
        )
    nodes = Nodes(process_id)
    made = record("RooWorkspace")
    named(made, value.GetName(), value.GetTitle())
    objects = list(value._nodes.values())
    put(made, "_allOwnedNodes", nodes.listed(objects, owned=True))
    put(made, "_dataList", linked([dataset(nodes, data) for data in value._data.values()]))
    put(made, "_namedSets", {name: nodes.listed(list(group))
                             for name, group in value._sets.items()})
    put(made, "_snapshots", linked([nodes.listed(snapshot, name, True)
                                   for name, snapshot in value._snapshots.items()]))
    nodes.finalize_links(objects)
    payload = Payload(made, value.GetTitle())
    payload.process_id_needed = bool(nodes.nodes)
    return payload


def write_workspace(directory: Any, name: str, value: Any) -> tuple[str, str] | None:
    with SESSION.lock:
        top = directory._file
        process_id, needed = SESSION.file_index(top)
        payload = workspace(value, process_id)
        directory.write(name, payload)
        if not payload.process_id_needed or not needed:
            return None
        identity = record("TProcessID")
        named(identity, f"ProcessID{process_id}", SESSION.title)
        top.write(f"ProcessID{process_id}", Payload(identity, SESSION.title))
        SESSION.remember(top, process_id)
        return f"ProcessID{process_id}", SESSION.title


def written_key(directory: Any, name: str, classname: str) -> Key:
    """The encoded header retains the actual compression, cycle and committed byte count."""
    here, leaf = directory._place(name)
    keys = (Key(Buffer(raw)) for raw in reversed(here._keys))
    return next(key for key in keys if key.name == leaf and key.classname == classname)


def output(filename: str, recreate: bool) -> Any:
    from ...writer import create
    from ...wupdate import update

    if recreate:
        return create(filename)
    try:
        return update(filename)
    except FileNotFoundError:
        return create(filename)
