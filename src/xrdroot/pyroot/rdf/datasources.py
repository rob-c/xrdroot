"""``RDF::FromCSV`` and ``RDF::FromSqlite``: the arguments ROOT takes, as the sources read.

``FromCSV(file, readHeaders, delimiter, linesChunkSize, colTypes)`` or
``FromCSV(file, options)`` with an ``RCsvDS::ROptions``; a type letter or a
delimiter may come as a one-letter string or as a C ``char``'s number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...rdf.csvsource import CsvOptions, from_csv
from ...rdf.sqlitesource import from_sqlite
from ...rdf.tabular import Columns
from ..core.messages import Warning

__all__ = ["RCsvDS", "csv_source", "lazy_source", "sqlite_source"]


def _char(value: Any) -> str:
    return chr(value) if isinstance(value, int) else str(value)


def _types(given: Any) -> dict[str, str]:
    """``colTypes``: a map of column names to type letters, or the pairs a braced list gives."""
    pairs = given.items() if hasattr(given, "items") else (given or [])
    return {str(name): _char(letter) for name, letter in pairs}


def csv_source(fileName: Any, readHeaders: Any = True, delimiter: Any = ",",
               linesChunkSize: int = -1, colTypes: Any = None) -> Columns:  # fmt: skip
    """The CSV file's columns, read with ROOT's options; ROOT's warning of any empty integers."""
    if isinstance(readHeaders, CsvOptions):
        options = readHeaders
        options.fDelimiter = _char(options.fDelimiter)
        options.fColumnTypes = _types(options.fColumnTypes)
    else:
        options = CsvOptions(fHeaders=bool(readHeaders), fDelimiter=_char(delimiter),
                             fLinesChunkSize=int(linesChunkSize), fColumnTypes=_types(colTypes))
    source = from_csv(str(fileName), options)
    if source.warning:
        Warning("RCsvDS", "%s", source.warning)
    return source


def sqlite_source(fileName: Any, query: Any) -> Columns:
    """The rows the query gives from the SQLite file."""
    return from_sqlite(str(fileName), str(query))


#: The C++ type of a column of each NumPy kind of number, for a lazy frame's columns.
LAZY_TYPES = {"f": "double", "i": "Long64_t", "u": "ULong64_t", "b": "bool"}


def lazy_source(*pairs: Any) -> Columns:
    """``MakeLazyDataFrame(pair(name, result), ...)``: columns of results ``Take`` booked."""
    columns = {}
    for name, result in pairs:
        taken = result.GetValue() if hasattr(result, "GetValue") else result
        values = taken.data() if callable(getattr(taken, "data", None)) else taken
        columns[str(name)] = np.asarray(values)
    kinds = {name: LAZY_TYPES.get(values.dtype.kind, "std::string")
             for name, values in columns.items()}  # fmt: skip
    return Columns(columns, kinds, "RLazyDS")


class RCsvDS:
    """``ROOT::RDF::RCsvDS``: here, the home of ``ROptions``."""

    ROptions = CsvOptions
