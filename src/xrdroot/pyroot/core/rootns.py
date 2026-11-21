"""``ROOT.ROOT``: C++'s ``ROOT::`` namespace - implicit multithreading, and everything else.

``ROOT::EnableImplicitMT()`` is spelt ``ROOT.ROOT.EnableImplicitMT()`` from
Python, and ``ROOT::RDataFrame`` ``ROOT.ROOT.RDataFrame``; the namespace
here has the threading switches - which are xrdroot's worker processes -
and hands any other name to the top of the namespace, where the class of
that name is if the kit has it.
"""

from __future__ import annotations

import sys
from typing import Any

from ...rdf.frame import DisableImplicitMT, EnableImplicitMT, GetThreadPoolSize, IsImplicitMTEnabled

__all__ = [
    "ROOT",
    "EnableImplicitMT",
    "DisableImplicitMT",
    "IsImplicitMTEnabled",
    "GetThreadPoolSize",
    "EnableThreadSafety",
    "TSeqI",
    "TSeqL",
    "TSeqU",
    "TSeqUL",
]


def EnableThreadSafety() -> None:
    """``ROOT::EnableThreadSafety``: nothing to do - xrdroot's objects share no global lock."""


class TSeqI:
    """``ROOT::TSeqI(begin, end, step)``: the integers a ``for`` over it walks, as ``range``."""

    def __init__(self, *bounds: Any) -> None:
        if len(bounds) == 1 and not isinstance(bounds[0], int) and hasattr(bounds[0], "__iter__"):
            # TSeqI(seeds): another sequence, walked as it is (the copy C++ makes)
            self._range = range(len(list(bounds[0])))
        else:
            self._range = range(*(int(bound) for bound in bounds))

    def __iter__(self) -> Any:
        return iter(self._range)

    def __len__(self) -> int:
        return len(self._range)

    def size(self) -> int:
        return len(self._range)

    def front(self) -> int:
        return self._range[0]

    def back(self) -> int:
        return self._range[-1]

    def __getitem__(self, index: int) -> int:
        return self._range[index]

    def value_at(self, index: int) -> int:
        """The value ``index`` places from the start, which C++ works out and does not look
        up: past the last one a ``TSeq``'s value is still ``begin + index * step``."""
        return self._range.start + int(index) * self._range.step if self._range.step else 0

    def begin(self) -> _TSeqIterator:
        """``begin``: where a walk over the sequence starts, as C++'s iterator is."""
        return _TSeqIterator(self, 0)

    def end(self) -> _TSeqIterator:
        """``end``: one past the last value, which a walk compares against and never reads."""
        return _TSeqIterator(self, len(self._range))


class _TSeqIterator:
    """``ROOT::Internal::TSeq<T>::iterator``: where a walk over a ``TSeq`` is.

    It is C++'s random-access iterator: ``it + n`` and ``it - n`` are another place, the
    difference of two is how far apart they are, ``++``/``--`` move it, ``it[n]`` reads a
    value n further on, and ``*it`` - which the runtime reads as the ``value`` it holds -
    is the value it is at.
    """

    #: ``*it`` of it is a value, which the runtime reads as a cell's ``.value`` is.
    _cint_cell = True

    def __init__(self, seq: TSeqI, at: int) -> None:
        self._seq, self._at = seq, int(at)

    @property
    def value(self) -> int:
        """``*it``: the value the walk is at."""
        return self._seq.value_at(self._at)

    def __add__(self, count: Any) -> _TSeqIterator:
        return _TSeqIterator(self._seq, self._at + int(count))

    __radd__ = __add__

    def __sub__(self, other: Any) -> Any:
        if isinstance(other, _TSeqIterator):
            return self._at - other._at  # how far apart two places are
        return _TSeqIterator(self._seq, self._at - int(other))

    def __iadd__(self, count: Any) -> _TSeqIterator:
        self._at += int(count)
        return self

    def __isub__(self, count: Any) -> _TSeqIterator:
        self._at -= int(count)
        return self

    def __getitem__(self, index: int) -> int:
        return self._seq.value_at(self._at + int(index))

    def _place(self, other: Any) -> tuple[int, int] | None:
        """The two places a comparison is between, or ``None`` for something else - which is
        not equal to a place and orders before nothing."""
        if isinstance(other, _TSeqIterator) and other._seq is self._seq:
            return self._at, other._at
        return None

    def __eq__(self, other: object) -> bool:
        places = self._place(other)
        return places[0] == places[1] if places is not None else NotImplemented

    def __ne__(self, other: object) -> bool:
        same = self.__eq__(other)
        return same if same is NotImplemented else not same

    def __ge__(self, other: Any) -> bool:
        places = self._place(other)
        return places[0] >= places[1] if places is not None else NotImplemented

    def __le__(self, other: Any) -> bool:
        places = self._place(other)
        return places[0] <= places[1] if places is not None else NotImplemented

    def __gt__(self, other: Any) -> bool:
        places = self._place(other)
        return places[0] > places[1] if places is not None else NotImplemented

    def __lt__(self, other: Any) -> bool:
        places = self._place(other)
        return places[0] < places[1] if places is not None else NotImplemented

    __hash__ = None  # type: ignore[assignment]  # a place has no hash of its own in C++

    def __repr__(self) -> str:
        return f"<TSeq iterator at {self._at}>"


#: ROOT's names for a sequence of its integer types: ``TSeqI`` is of ``int``, and a walk over
#: one of the others walks numbers as ``TSeqI`` does.
TSeqL = TSeqU = TSeqUL = TSeqI


class _Namespace:
    """``ROOT::``: its own few functions, then whatever the namespace's top has by that name."""

    EnableImplicitMT = staticmethod(EnableImplicitMT)
    DisableImplicitMT = staticmethod(DisableImplicitMT)
    IsImplicitMTEnabled = staticmethod(IsImplicitMTEnabled)
    GetThreadPoolSize = staticmethod(GetThreadPoolSize)
    EnableThreadSafety = staticmethod(EnableThreadSafety)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(sys.modules["xrdroot.pyroot"], name)

    def __repr__(self) -> str:
        return "<namespace ROOT>"


#: ``ROOT.ROOT``.
ROOT = _Namespace()
