"""``RooDataSet::write`` and ``RooDataSet::read``: a dataset as lines of text, and back.

``write(stream)`` puts each event on a line, every variable's value as the
stream writes a number - six digits unless the stream was told otherwise -
and a space after each, as ``RooArgList::writeToStream`` writes it compact;
a category writes its label. ``read(files, variables)`` makes the dataset
``dataset``, titled by the files, of those variables and a ``blindState``
category ROOT adds to it (``Normal`` or ``Blind``, not read from the lines),
one event a line; a line with a value outside its variable's range is left
out and counted, ``#`` lines are comments, and what was read is said as
``RooDataSet::read`` says it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..collections import as_list
from ..messages import INFO, log

__all__ = ["read", "write"]


def write(dataset: Any, stream: Any) -> bool:
    """``write(ofs)``: the events, a line each; ``False``, as ROOT's ``ofs.fail()`` is then."""
    from ...cppstd.streams import endl

    for index in range(dataset.numEntries()):
        for one in dataset.get(index):
            category = getattr(one, "isCategory", lambda: False)()
            stream << (one.getLabel() if category else float(one.getVal())) << " "
        stream << endl
    return False


def _blind_state(variables: list[Any]) -> Any:
    """The variables' ``blindState`` - made, if they have none - with ROOT's two states."""
    from ..categories import RooCategory

    found = next((one for one in variables if one.GetName() == "blindState"), None)
    if found is None:
        found = RooCategory("blindState", "Blinding State")
        variables.append(found)
    for label, number in (("Normal", 0), ("Blind", 1)):
        if not found.hasLabel(label):
            found.defineType(label, number)
    found.setAttribute("Dynamic")
    return found


def _event(line: str, variables: list[Any]) -> list[float] | None:
    """One line's values, for the variables read from lines - ``None`` if one is out of range."""
    fields = line.split()
    values = [float(field) for field in fields[: len(variables)]]
    if len(values) < len(variables):
        return None
    if any(not one.inRange(value) for one, value in zip(variables, values, strict=True)):
        return None
    return values


def _read_lines(path: Path, data: Any, chosen: list[Any], read_from: list[Any], blind: Any) -> int:
    """Each event of one file added to ``data``; how many were out of range."""
    ignored = 0
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    for line in (one for one in lines if not one.lstrip().startswith("#")):
        values = _event(line, read_from)
        if values is None:
            ignored += 1
            continue
        for one, value in zip(read_from, values, strict=True):
            one.setVal(value)
        blind.setIndex(0)
        data.add(chosen)
    return ignored


def read(files: Any, variables: Any, options: Any = "", common_path: Any = "") -> Any:
    """``RooDataSet::read(fileList, varList, verbOpt, commonPath)``: the events of the files."""
    from .dataset import RooDataSet

    chosen = list(as_list(variables))
    blind = _blind_state(chosen)
    read_from = [one for one in chosen if not one.getAttribute("Dynamic")]
    data = RooDataSet("dataset", str(files), chosen)
    ignored = 0
    for name in [part.strip() for part in str(files).split(",") if part.strip()]:
        log(None, INFO, "DataHandling", f"RooDataSet::read: reading file {name}")
        path = Path(str(common_path)) / name if common_path else Path(name)
        ignored += _read_lines(path, data, chosen, read_from, blind)
    if data.numEntries():  # the dataset's own variables hold the last event, as filling left them
        data.get(data.numEntries() - 1)
    log(None, INFO, "DataHandling", f"RooDataSet::read: read {data.numEntries()} events "
        f"(ignored {ignored} out of range events)")  # fmt: skip
    return data
