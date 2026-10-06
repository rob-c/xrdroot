"""``RNTupleProcessor``: several RNTuples read as one - one after another, or side by side.

A chain, ``CreateChain(specs)``, reads RNTuples one after another, its
entries numbered on across them; a join, ``CreateJoin(primary, auxiliary,
joinFields)``, reads the primary's entries and with each the auxiliary
entry whose join fields hold the same values, its fields named
``auxiliary.field``. ``RequestField<T>(name)`` hands out the holder the
current entry's value is put in as the processor is iterated.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .fields import field_type, typed
from .reading import RNTupleReader, _load

__all__ = ["RNTupleOpenSpec", "RNTupleProcessor"]


class RNTupleOpenSpec:
    """``RNTupleOpenSpec``: an RNTuple's name, and the file it is in."""

    def __init__(self, name: Any, storage: Any) -> None:
        self.fNTupleName, self.fStorage = str(name), str(storage)


def _spec(given: Any) -> RNTupleOpenSpec:
    """An open spec, from one or from its braced ``{name, storage}``."""
    return given if isinstance(given, RNTupleOpenSpec) else RNTupleOpenSpec(*given)


def _reader(spec: RNTupleOpenSpec) -> RNTupleReader:
    return RNTupleReader.Open(spec.fNTupleName, spec.fStorage)


class RNTupleProcessor:
    """The entries of a chain or a join, and the holders of the fields asked for."""

    def __init__(self, readers: list[tuple[str, RNTupleReader]], join: list[str] | None) -> None:
        self._readers, self._join = readers, join
        self._requested: dict[str, Any] = {}
        self._processed, self._number = 0, 0

    @staticmethod
    def CreateChain(specs: Any, model: Any = None) -> RNTupleProcessor:
        """``CreateChain(specs)``: the RNTuples one after another."""
        return RNTupleProcessor([("", _reader(_spec(spec))) for spec in specs], None)

    @staticmethod
    def CreateJoin(primary: Any, auxiliary: Any, fields: Any, *models: Any) -> RNTupleProcessor:
        """``CreateJoin(primary, auxiliary, joinFields)``: each primary entry with the auxiliary
        entry of the same join fields' values - or, with no join fields, of the same number."""
        main, aux = _spec(primary), _spec(auxiliary)
        readers = [("", _reader(main)), (aux.fNTupleName, _reader(aux))]
        return RNTupleProcessor(readers, [str(field) for field in fields])

    @typed
    def RequestField(self, kind: Any, name: Any) -> Any:
        """``RequestField<T>(name)``: the holder each entry's value of the field is put in."""
        held = field_type(kind).holder()
        self._requested[str(name)] = held
        return held

    def GetNEntriesProcessed(self) -> int:
        return self._processed

    def GetCurrentProcessorNumber(self) -> int:
        return self._number

    def __iter__(self) -> Iterator[int]:
        return self._joined() if self._join is not None else self._chained()

    def _chained(self) -> Iterator[int]:
        self._processed = 0
        for number, (_, reader) in enumerate(self._readers):
            self._number = number
            for entry in reader:
                self._fill(reader, "", entry)
                self._processed += 1
                yield self._processed - 1

    def _joined(self) -> Iterator[int]:
        (_, main), (prefix, aux) = self._readers
        assert self._join is not None
        found = _index(aux, self._join)
        self._processed = 0
        for entry in main:
            key = tuple(main.column(field)[entry] for field in self._join) or (entry,)
            self._fill(main, "", entry)
            if key in found:
                self._fill(aux, prefix + ".", found[key])
            self._processed += 1
            yield entry

    def _fill(self, reader: RNTupleReader, prefix: str, entry: int) -> None:
        """The requested fields of one reader - those under ``prefix`` - set from an entry."""
        for name, held in self._requested.items():
            own = name[len(prefix):] if prefix else name
            if (not prefix and "." in name) or (prefix and not name.startswith(prefix)):
                continue
            _load(held, reader.column(own)[entry])


def _index(reader: RNTupleReader, fields: list[str]) -> dict[tuple[Any, ...], int]:
    """Each entry's number by the values of its join fields - by its own number with none."""
    if not fields:
        return {(entry,): entry for entry in reader}
    columns = [reader.column(field) for field in fields]
    return {tuple(column[entry] for column in columns): entry for entry in reader}
