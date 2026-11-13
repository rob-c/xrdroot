"""What a call returns a reference to, held so a compound assignment evaluates it once.

``v(v.GetUpb()) += 1`` reads the element and writes it back; the
translation evaluates the call's arguments once, into one of these, and
:func:`store_reference` does the reading and writing through ``.value``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from .references import assign_call, assign_into, assign_method

__all__ = ["CallRef", "HeldRef"]


class CallRef:
    """What ``obj(*args)`` - or ``obj.name(*args)`` - returns a reference to, as ``.value``.

    ``v(v.GetUpb()) += 1`` evaluates the call's arguments once; this holds
    them, so the element is read by calling again with the same ones and
    written back the way ``f(i) = v`` writes, through :func:`assign_call` or
    :attr:`assigning` - :func:`assign_method`, unless a dialect's runtime has
    its own.
    """

    __slots__ = ("obj", "args", "name")

    #: How ``obj.name(*args) = v`` is written.
    assigning: ClassVar[Callable[..., Any]] = staticmethod(assign_method)

    def __init__(self, obj: Any, args: tuple[Any, ...], name: str | None = None) -> None:
        self.obj = obj
        self.args = args
        self.name = name

    @property
    def value(self) -> Any:
        called = self.obj if self.name is None else getattr(self.obj, self.name)
        return called(*self.args)

    @value.setter
    def value(self, value: Any) -> None:
        if self.name is None:
            assign_call(self.obj, self.args, value)
        else:
            self.assigning(self.obj, self.name, self.args, value)


class HeldRef:
    """A temporary object - ``Column(m, i++)`` - read as itself and written into."""

    __slots__ = ("obj",)

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    @property
    def value(self) -> Any:
        return self.obj

    @value.setter
    def value(self, value: Any) -> None:
        assign_into(self.obj, value)
