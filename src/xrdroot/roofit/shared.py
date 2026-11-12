"""Named variable binnings share ROOT UUID identity without keeping dead models alive."""

from __future__ import annotations

import uuid
from threading import RLock
from typing import Any
from weakref import WeakValueDictionary

#: TUUID's persistent fields; the final field is the six node bytes.
UUID_FIELDS = ("fTimeLow", "fTimeMid", "fTimeHiAndVersion", "fClockSeqHiAndReserved",
               "fClockSeqLow", "fNode")
#: The null shared-property sentinel must not join otherwise unrelated variables.
NULL_IDENTITY = (0, 0, 0, 0, 0, (0, 0, 0, 0, 0, 0))


def new_identity() -> tuple[Any, ...]:
    low, mid, high, clock_high, clock_low, node = uuid.uuid4().fields
    return low, mid, high, clock_high, clock_low, tuple(node.to_bytes(6, "big"))


class SharedBinnings(dict[str, Any]):
    """A weakly interned property map, allocated by cloning just as RooRealVar::sharedProp() is."""

    __slots__ = ("identity", "__weakref__")

    def __init__(self, values: Any = None, identity: tuple[Any, ...] | None = None) -> None:
        super().__init__(values or {})
        self.identity = identity if identity is not None else (new_identity() if self else None)

    def __setitem__(self, name: str, value: Any) -> None:
        super().__setitem__(name, value)
        self.key()

    def key(self, create: bool = False) -> tuple[Any, ...] | None:
        if self.identity is None and (self or create):
            self.identity = new_identity()
        return self.identity

    def __deepcopy__(self, memo: dict[int, Any]) -> SharedBinnings:
        self.key(create=True)
        memo[id(self)] = self
        return self


#: ROOT keeps only weak references to properties indexed by UUID.
PROPERTIES: WeakValueDictionary[tuple[Any, ...], SharedBinnings] = WeakValueDictionary()
#: A read transaction publishes new identities only after its whole graph is valid.
PROPERTIES_LOCK = RLock()
