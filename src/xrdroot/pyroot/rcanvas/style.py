"""``RStyle``: attribute values for the drawables a selector picks out, as CSS gives them.

A style is blocks, each a selector and its values: ``line`` picks out
every line, ``.group1`` every drawable of that CSS class, ``#obj7`` the
one of that id. It is written in CSS - ``RStyle::Parse("line {
line_width: 2; }")`` - or built block by block, ``AddBlock(".a").AddInt
("line_style", 4)``; :meth:`RStyle.Eval` says what a drawable is given
for an attribute, the last block that picks it out winning.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = ["RStyle"]

#: One block of the CSS a style is parsed from: its selector, and what is in its braces.
BLOCK = re.compile(r"\s*([^{}\s][^{}]*?)\s*\{([^{}]*)\}")


def _value(text: str) -> Any:
    """A CSS value as the attribute takes it: a boolean, a number, or the text itself."""
    if text in ("true", "false"):
        return text == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            continue
    return text


class _Block:
    """One block of a style: a selector, and the values it gives."""

    def __init__(self, selector: str) -> None:
        self.selector = selector
        self.values: dict[str, Any] = {}

    def _add(self, name: Any, value: Any) -> _Block:
        self.values[str(name)] = value
        return self

    def AddInt(self, name: Any, value: Any) -> _Block:
        return self._add(name, int(value))

    def AddDouble(self, name: Any, value: Any) -> _Block:
        return self._add(name, float(value))

    def AddString(self, name: Any, value: Any) -> _Block:
        return self._add(name, str(value))

    def AddBool(self, name: Any, value: Any) -> _Block:
        return self._add(name, bool(value))

    def picks(self, drawable: Any) -> bool:
        """Does this block's selector pick out ``drawable``, by its type, class or id?"""
        found = self.selector
        if found.startswith("."):
            return bool(drawable.GetCssClass() == found[1:])
        if found.startswith("#"):
            return bool(drawable.GetId() == found[1:])
        return bool(drawable.CSS_TYPE == found)


class RStyle:
    """``RStyle``: blocks of attribute values, each for the drawables its selector picks out."""

    def __init__(self) -> None:
        self.blocks: list[_Block] = []

    def AddBlock(self, selector: Any) -> _Block:
        made = _Block(str(selector).strip())
        self.blocks.append(made)
        return made

    def ParseString(self, css: Any) -> bool:
        """Blocks from CSS added to this style; ``False``, and none added, if it is not CSS."""
        text = str(css)
        if BLOCK.sub("", text).strip():
            return False
        made = []
        for found in BLOCK.finditer(text):
            block = _Block(found.group(1))
            for entry in filter(None, (part.strip() for part in found.group(2).split(";"))):
                name, colon, value = entry.partition(":")
                if not colon:
                    return False
                block._add(name.strip(), _value(value.strip()))
            made.append(block)
        self.blocks.extend(made)
        return True

    @staticmethod
    def Parse(css: Any) -> RStyle | None:
        """``RStyle::Parse``: the style the CSS is, or ``None`` - a null pointer - if it is not."""
        made = RStyle()
        return made if made.ParseString(css) else None

    def Eval(self, name: Any, drawable: Any) -> Any:
        """What the style gives ``drawable`` for the attribute ``name``, or ``None``."""
        found = None
        for block in self.blocks:
            if str(name) in block.values and block.picks(drawable):
                found = block.values[str(name)]
        return found
