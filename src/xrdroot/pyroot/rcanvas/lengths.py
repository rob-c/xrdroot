"""ROOT 7's pad lengths: a sum of a fraction of the pad, pixels, and the frame's user units.

``0.1_normal`` is a tenth of the pad, ``20_px`` twenty pixels, ``80_user``
80 on the frame's axis; a length is any sum of them - ``0.5_normal -
5_px`` - scaled by numbers, which is how a macro places a primitive a few
pixels off a fraction of its pad. A position, ``RPadPos``, is two of them;
a bare number given for one is a fraction of the pad.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RPadLength", "RPadPos", "RPadExtent", "length"]


class RPadLength:
    """``RPadLength``: so much of the pad, so many pixels and so many user units, summed."""

    __slots__ = ("normal", "pixel", "user")

    def __init__(self, normal: float = 0.0, pixel: float = 0.0, user: float = 0.0) -> None:
        self.normal, self.pixel, self.user = float(normal), float(pixel), float(user)

    @staticmethod
    def Normal(value: Any) -> RPadLength:
        return RPadLength(normal=value)

    @staticmethod
    def Pixel(value: Any) -> RPadLength:
        return RPadLength(pixel=value)

    @staticmethod
    def User(value: Any) -> RPadLength:
        return RPadLength(user=value)

    def _parts(self) -> tuple[float, float, float]:
        return self.normal, self.pixel, self.user

    def __repr__(self) -> str:
        named = zip(("normal", "px", "user"), self._parts(), strict=True)
        return "RPadLength(" + " + ".join(f"{v:g}_{n}" for n, v in named if v) + ")"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RPadLength) and other._parts() == self._parts()

    __hash__ = None  # type: ignore[assignment]

    def __add__(self, other: Any) -> RPadLength:
        given = length(other)
        return RPadLength(*(a + b for a, b in zip(self._parts(), given._parts(), strict=True)))

    __radd__ = __add__

    def __sub__(self, other: Any) -> RPadLength:
        return self + (-length(other))

    def __rsub__(self, other: Any) -> RPadLength:
        return length(other) - self

    def __neg__(self) -> RPadLength:
        return self * -1.0

    def __mul__(self, factor: Any) -> RPadLength:
        return RPadLength(*(part * float(factor) for part in self._parts()))

    __rmul__ = __mul__

    def __truediv__(self, divisor: Any) -> RPadLength:
        return self * (1.0 / float(divisor))

    def HasNormal(self) -> bool:
        return self.normal != 0

    def HasPixel(self) -> bool:
        return self.pixel != 0

    def HasUser(self) -> bool:
        return self.user != 0

    def GetNormal(self) -> float:
        return self.normal

    def GetPixel(self) -> float:
        return self.pixel

    def GetUser(self) -> float:
        return self.user


def length(value: Any) -> RPadLength:
    """A length, from one or from a number - which is that fraction of the pad."""
    return value if isinstance(value, RPadLength) else RPadLength(normal=value)


class RPadPos:
    """``RPadPos``: a point on a pad, its ``x`` and ``y`` each a length."""

    def __init__(self, x: Any = 0.0, y: Any = 0.0) -> None:
        self.fHoriz, self.fVert = length(x), length(y)

    @property
    def x(self) -> RPadLength:
        return self.fHoriz

    @property
    def y(self) -> RPadLength:
        return self.fVert

    def __repr__(self) -> str:
        return f"RPadPos({self.fHoriz!r}, {self.fVert!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RPadPos) and (other.x, other.y) == (self.x, self.y)

    __hash__ = None  # type: ignore[assignment]


#: ``RPadExtent``: a width and a height, kept as a position is.
RPadExtent = RPadPos


def position(value: Any) -> RPadPos:
    """A position, from one or from the ``{x, y}`` a macro braces."""
    return value if isinstance(value, RPadPos) else RPadPos(*value)
