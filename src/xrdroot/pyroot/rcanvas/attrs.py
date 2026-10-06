"""ROOT 7's drawing attributes: groups of named values, and a group assigned into as a whole.

A primitive's attributes are groups - ``line``, ``fill``, ``text`` - and a
group may hold groups: ``axis.ticks``, ``text.font``. Each group knows its
members and their defaults, and anything else set on it is an error, as
it is in C++. ROOT's groups take a value of their own as well -
``axis.title = "x"``, ``frame.margins = 0.1_normal``, ``text.font =
RAttrFont::kArial`` - so setting a member that is a group to something
that is not one of its kind hands the value to the group, which keeps it
its own way.
"""

from __future__ import annotations

import copy
import types
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

__all__ = ["Attrs", "RColor", "frozen"]


def frozen(members: dict[str, Callable[[], Any]]) -> Mapping[str, Callable[[], Any]]:
    """A group's table of members, which no instance changes."""
    return types.MappingProxyType(members)


class Attrs:
    """A group of attributes: its members by name, each made from its default."""

    #: Each member's name and what makes its default; a subclass adds its own.
    MEMBERS: ClassVar[Mapping[str, Callable[[], Any]]] = frozen({})

    def __init__(self) -> None:
        for name, made in self._members().items():
            object.__setattr__(self, name, made())

    @classmethod
    def _members(cls) -> dict[str, Callable[[], Any]]:
        found: dict[str, Callable[[], Any]] = {}
        for klass in reversed(cls.__mro__):
            found.update(getattr(klass, "MEMBERS", {}))
        return found

    def __getattr__(self, name: str) -> Any:
        """Only a name that is no member gets here: there is no such attribute."""
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name not in self._members():
            raise AttributeError(f"{type(self).__name__} has no attribute {name!r}; it has "
                                 f"{', '.join(self._members()) or 'none'}")  # fmt: skip
        held = self.__dict__.get(name)
        if isinstance(held, Attrs) and not isinstance(value, type(held)):
            held._assign(value)
        else:
            object.__setattr__(self, name, copy.copy(value) if isinstance(value, Attrs) else value)

    def _assign(self, value: Any) -> None:
        """What a group makes of a value given for it whole; a group that takes none refuses."""
        members = ", ".join(self._members())
        raise TypeError(f"{type(self).__name__} is a group of attributes; set its members - "
                        f"{members} - rather than give it {value!r}")

    def __repr__(self) -> str:
        members = ", ".join(f"{name}={getattr(self, name)!r}" for name in self._members())
        return f"{type(self).__name__}({members})"


#: ROOT 7's named colours, as the red, green and blue each is made of.
NAMED = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 128, 0),
    "blue": (0, 0, 255), "yellow": (255, 255, 0), "magenta": (255, 0, 255),
    "cyan": (0, 255, 255), "orange": (255, 165, 0), "grey": (128, 128, 128),
}  # fmt: skip


class RColor:
    """``RColor``: red, green, blue and how opaque, out of 255 - or a name a browser knows."""

    def __init__(self, red: Any = 0, green: Any = 0, blue: Any = 0, alpha: Any = 255) -> None:
        if isinstance(red, str):
            self.name = red
            red, green, blue = NAMED.get(red.lower(), (0, 0, 0))
        else:
            self.name = ""
        self.rgba = (int(red), int(green), int(blue), int(alpha))

    def GetRed(self) -> int:
        return self.rgba[0]

    def GetGreen(self) -> int:
        return self.rgba[1]

    def GetBlue(self) -> int:
        return self.rgba[2]

    def GetAlpha(self) -> int:
        return self.rgba[3]

    def AsHex(self) -> str:
        return "".join(f"{part:02X}" for part in self.rgba[:3])

    def AsString(self) -> str:
        """The colour as a browser is told it: its name, or ``#RRGGBB``."""
        return self.name or "#" + self.AsHex()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RColor) and other.rgba == self.rgba

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"RColor({self.AsString()})"


for _name in NAMED:
    setattr(RColor, "k" + _name.capitalize(), RColor(_name))
