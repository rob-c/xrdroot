"""What an ``RNTupleReader`` prints: the summary box of ``PrintInfo`` and the JSON of ``Show``.

The box is ROOT's, 80 characters wide: the name and the number of entries,
then a line per field - a member's indented under its parent, numbered
``1.1`` - with each label padded to the longest one's width and three
more. ``Show`` writes an entry as ROOT's JSON does: numbers as C++'s
streams print them, which is ``%g``, strings in quotes, ``true`` and
``false``, a vector's items in brackets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["info_text", "json_text"]

#: How wide ROOT's box is.
WIDTH = 80


def _boxed(text: str) -> str:
    return f"* {text:<{WIDTH - 4}} *\n"


def _rows(fields: list[Any], ids: list[int], prefix: str, depth: int) -> list[tuple[str, str]]:
    """Each field under ``ids`` and its members: its label, and its name and type."""
    rows = []
    for at, field_id in enumerate(ids, start=1):
        field = fields[field_id]
        number = f"{prefix}{at}"
        rows.append((f"{'  ' * depth}Field {number}", f"{field.name} ({field.type})"))
        children = [child for child in field.children if child != field_id]
        rows.extend(_rows(fields, children, f"{number}.", depth + 1))
    return rows


def info_text(ntuple: Any) -> str:
    """``PrintInfo()``'s summary box of an RNTuple's fields."""
    schema = ntuple._store.schema
    rows = _rows(schema.fields, [field.id for field in schema.tops()], "", 0)
    width = max((len(label) for label, _ in rows), default=0) + 3
    stars = "*" * WIDTH + "\n"
    text = "*" * 36 + " NTUPLE " + "*" * 36 + "\n"
    text += _boxed(f"N-Tuple : {ntuple.name}") + _boxed(f"Entries : {len(ntuple)}") + stars
    text += "".join(_boxed(f"{label:<{width}}: {what}") for label, what in rows)
    return text + stars


def _value(value: Any) -> str:
    """One value as ROOT's JSON writes it."""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return "[" + ", ".join(_value(item) for item in value) + "]"


def json_text(values: dict[str, Any]) -> str:
    """``Show(i)``'s JSON of one entry's values, by field name."""
    lines = ",\n".join(f'  "{name}": {_value(value)}' for name, value in values.items())
    return "{\n" + lines + "\n}\n"
