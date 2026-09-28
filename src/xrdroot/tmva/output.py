"""The Factory's output file: TMVA's directories, histograms and trees, where TMVA puts them.

``dataset/`` holds the correlation matrices, an ``InputVariables_<trafo>``
directory per Factory transformation, ``Method_<type>/<name>/`` for every
method and the ``TestTree`` and ``TrainTree``; this writes into whatever
the macro opened - a :class:`xrdroot.pyroot.TFile` - through its own
directories, so that the file lists and reads back as any other written
there. A Factory made without a file writes nothing, as TMVA's
"silent file" mode does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dataset import DataSetInfo, Events

__all__ = ["LeafList", "Output", "event_tree"]


class Output:
    """Where the Factory writes: the pyroot file, or nothing at all."""

    def __init__(self, tfile: Any) -> None:
        self.tfile = tfile
        self._made: dict[str, Any] = {}

    @property
    def silent(self) -> bool:
        return self.tfile is None

    def GetName(self) -> str:
        return "" if self.tfile is None else str(self.tfile.GetName())

    def directory(self, path: str, title: str = "") -> Any:
        """The directory at ``path`` - ``dataset/Method_BDT/BDT`` - made, with ``title``, if new."""
        if path in self._made:
            return self._made[path]
        parent_path, _, name = path.rpartition("/")
        parent = self.directory(parent_path) if parent_path else self.tfile
        found = parent.GetDirectory(name)
        if found is None:
            found = parent.mkdir(name, title or name)
        self._made[path] = found
        return found

    def exists(self, path: str) -> bool:
        return path in self._made

    def write(self, path: str, obj: Any, name: str | None = None) -> None:
        """``obj`` - an :class:`xrdroot.Histogram`, a string - written into the directory ``path``."""
        if self.silent:
            return
        directory = self.directory(path)
        text = isinstance(obj, str)
        called = name or obj.name
        title = "Collectable string class" if text else obj.title
        classname = "TObjString" if text else obj.classname
        directory._writable().write(called, obj, title=title)
        directory._note_key(called, title, classname)

    def write_tree(self, path: str, name: str, columns: dict[str, Any], title: str = "") -> None:
        """A tree of ``columns`` written into ``path``, a branch per column."""
        if self.silent:
            return
        from ..wtree import spec_of

        target = self.directory(path)._xrd
        specs = {
            column: ("f", values.leaves)
            if isinstance(values, LeafList)
            else spec_of(column, values)
            for column, values in columns.items()
        }
        data = {k: v.values if isinstance(v, LeafList) else v for k, v in columns.items()}
        target.tree(name, specs, title=title or name).extend(data)


def event_tree(dsi: DataSetInfo, events: Events, outputs: dict[str, Any]) -> dict[str, Any]:
    """The columns of TMVA's ``TestTree`` or ``TrainTree``: the events, then each method's output."""
    names = [info.name for info in dsi.classes]
    columns: dict[str, Any] = {
        "classID": events.classes.astype(np.int32),
        "className": [names[number] for number in events.classes],
    }
    for index, info in enumerate(dsi.variables):
        columns[info.label] = events.values[:, index].astype(np.float32)
    for index, info in enumerate(dsi.targets):
        columns[info.label] = events.targets[:, index].astype(np.float32)
    for index, info in enumerate(dsi.spectators):
        columns[info.label] = events.spectators[:, index].astype(np.float32)
    columns["weight"] = events.weights.astype(np.float32)
    labels = (
        [info.name for info in dsi.classes]
        if len(dsi.classes) > 2
        else [info.label for info in dsi.targets]
    )
    for name in sorted(outputs):
        values = np.asarray(outputs[name], dtype=np.float32)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim == 1:
            columns[name] = values
            continue
        columns[name] = LeafList(values, tuple(labels[: values.shape[1]]))
    return columns


@dataclass
class LeafList:
    """A branch of one ``Float_t`` leaf per class or target, ``Signal/F:bg0/F``, as TMVA writes."""

    values: Any
    leaves: tuple[str, ...]
