"""C's ``printf`` family as a translation calls it, from :mod:`cppstd`."""

from __future__ import annotations

from ...cppstd.printf import BITS as BITS
from ...cppstd.printf import CONVERTERS as CONVERTERS
from ...cppstd.printf import SPEC as SPEC
from ...cppstd.printf import _Arguments as _Arguments
from ...cppstd.printf import _character as _character
from ...cppstd.printf import _conversion as _conversion
from ...cppstd.printf import _decimal as _decimal
from ...cppstd.printf import _integer as _integer
from ...cppstd.printf import _pointer as _pointer
from ...cppstd.printf import _signed as _signed
from ...cppstd.printf import _spec as _spec
from ...cppstd.printf import _text as _text
from ...cppstd.printf import _unsigned as _unsigned
from ...cppstd.printf import _value as _value
from ...cppstd.printf import cformat as cformat
from ...cppstd.printf import fprintf as fprintf
from ...cppstd.printf import printf as printf
from ...cppstd.printf import putchar as putchar
from ...cppstd.printf import puts as puts
from ...cppstd.printf import sprintf as sprintf

__all__ = ["cformat", "printf", "sprintf", "fprintf", "puts", "putchar"]
