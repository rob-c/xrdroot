"""ROOT's pave text files carry optional attributes for their preceding line."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any

from ..core.messages import Error

#: ROOT reads the last two values through a C float, including its rounding.
_ATTRIBUTES = {"Color": int, "Align": int, "Font": int, "Size": float, "Angle": float}
#: The numeric prefix accepted by ROOT's sscanf conversions.
_NUMBER = {
    int: re.compile(r"^[ \t]*([+-]?\d+)"),
    float: re.compile(r"^[ \t]*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"),
}


def _number(text: str, convert: type, previous: dict[type, float]) -> float:
    matched = _NUMBER[convert].match(text)
    if matched is not None:
        value = convert(float(matched[1]))
        if convert is float:
            value = struct.unpack("f", struct.pack("f", value))[0]
        previous[convert] = value
    return previous[convert]


def _attribute(pave: Any, text: str, previous: dict[type, float]) -> str | None:
    if "+SetText" not in text:
        return text
    if ")" not in text or not pave.members["fLines"]:
        return None
    command = text[8:text.index(")")]
    for name, convert in _ATTRIBUTES.items():
        if name + "(" in command:
            value = _number(command[len(name) + 1:], convert, previous)
            getattr(pave.members["fLines"][-1], "SetText" + name)(value)
            return None
    return text[:text.index(")")]


def read_file(pave: Any, filename: str, option: str, nlines: int, fromline: int) -> None:
    """Text directives do not consume a numbered line in ROOT's reader."""
    if "+" not in option:
        pave.Clear()
    pave.SetTextAlign(12)
    filename = str(filename).removesuffix(";")
    if not filename:
        return
    try:
        content = Path(filename).read_bytes().decode("utf-8", "surrogateescape")
    except OSError:
        Error("TPaveText::ReadFile", "illegal file name %s", filename)
        return
    number = 0
    previous: dict[type, float] = {int: 0, float: 0.0}
    for text in content.split("\n")[:-1]:
        if fromline <= number < fromline + nlines:
            line = _attribute(pave, text, previous)
            if line is None:
                continue
            pave.AddText(line)
        number += 1
