"""``TFITSHDU``: one header-data unit of a FITS file - its records, and an image or a table.

The file is read by :mod:`xrdroot.fitsio`, NumPy alone, named as CFITSIO
names one: ``TFITSHDU("f.fits[1][DATAMAX > 2e-15]")``, or ``TFITSHDU(path,
1)`` and ``TFITSHDU(path, "EVENTS")``. ``Print`` prints what ROOT's prints,
from the same records: ``"F"`` and ``"F+"`` the file's units, ``"T"`` a
table's columns, ``"T+"`` the table itself, and nothing the unit's records.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..errors import FormatError
from ..fitsio import VECTOR, Column, base_path, image, open_unit
from ..fitsio.tables import VECTOR_TYPES
from .core.messages import Error, Info, Warning
from .core.objects import TNamed
from .core.strings import TString
from .fitsimage import ImageReads
from .fitstable import TableReads

__all__ = ["TFITSHDU"]

#: What ``TFITSHDU::LoadHDU`` says of the unit it read.
LOADED = {True: "The selected HDU contains an Image Extension",
          False: "The selected HDU contains a Table Extension"}  # fmt: skip


def _load_warnings(columns: list[Column], rows: int) -> None:
    """ROOT's complaints about the columns it could not read, one for each."""
    for n, column in enumerate(columns, 1):
        if not column.kind:
            Warning("LoadHDU", "error opening FITS file. Column type %d is currently not "
                    "supported", column.typecode)  # fmt: skip
        elif column.kind == VECTOR and -column.typecode not in VECTOR_TYPES:
            for _ in range(rows):
                Error("LoadHDU", "The variable-length array type in column %d is unknown", n)


class TFITSHDU(ImageReads, TableReads, TNamed):
    """``TFITSHDU``: a FITS file's unit, opened by name, number or filter."""

    kImageHDU, kTableHDU = 0, 1

    def __init__(self, filepath: Any, extension: Any = None) -> None:
        super().__init__("", "")
        spec = str(filepath)
        self._base = base_path(spec)
        self._spec = spec if extension is None else f"{self._base}[{extension}]"
        if not self._load(self._spec):
            raise OSError(f"TFITSHDU could not open {self._spec}.")

    def _load(self, spec: str) -> bool:
        """``LoadHDU``: the unit read, or ROOT's warning and ``False``."""
        try:
            opened = open_unit(spec)
        except (OSError, FormatError) as err:
            Warning("LoadHDU", "error opening FITS file. Details: %s", str(err))
            return False
        self._opened, self._unit = opened, opened.unit
        self._kind = opened.unit.kind
        Info("LoadHDU", LOADED[self._kind == "IMAGE"])
        self._sizes = opened.unit.sizes if self._kind == "IMAGE" else []
        self._pixels = image(opened.unit) if self._kind == "IMAGE" else np.zeros(0)
        self._columns, self._rows = opened.columns, opened.rows
        _load_warnings(self._columns, self._rows)
        return True

    def Change(self, filter: Any) -> bool:
        """``Change("[2]")`` or ``Change(2)``: another unit of the file, the old kept on failure."""
        spec = self._base + (f"[{filter}]" if isinstance(filter, (int, np.integer)) else
                             str(filter))  # fmt: skip
        if self._load(spec):
            self._spec = spec
            return True
        Warning("Change", "error changing HDU. Restoring the previous one...")
        self._load(self._spec)
        return False

    # -- the unit ------------------------------------------------------------------------

    def GetType(self) -> int:
        return self.kImageHDU if self._kind == "IMAGE" else self.kTableHDU

    def GetHDUNumber(self) -> int:
        return self._unit.index + 1

    def GetExtensionName(self) -> TString:
        return TString(self._unit.value("EXTNAME") or "PRIMARY")

    def GetKeywordValue(self, keyword: str) -> TString:
        return TString(self._unit.value(str(keyword)) or "")

    def GetFilePath(self) -> TString:
        return TString(self._base)

    def Draw(self, option: str = "") -> None:
        """``Draw``: the picture in a canvas of its own size, named after the unit."""
        from .graphics.canvas import TCanvas

        if self._kind != "IMAGE":
            Warning("Draw", "cannot draw. This is not an image HDU.")
            return
        picture = self.ReadAsImage(0)
        if picture is not None:
            TCanvas(f"{self.GetName()}HDU", f"{self._sizes[0]} x {self._sizes[1]}",
                    self._sizes[0], self._sizes[1])  # fmt: skip
            picture.Draw()

    def Print(self, option: str = "") -> None:
        """``Print``: the file's units (``F``, ``F+``), a table (``T``, ``T+``), or the records."""
        from .fitsprint import print_columns, print_file, print_records, print_table

        opt = str(option)
        if opt[:1] in ("F", "f"):
            print_file(self._base, opt[1:2] == "+")
        elif opt[:1] in ("T", "t"):
            (print_table if opt[1:2] == "+" else print_columns)(self)
        else:
            print_records(self._unit.records)


