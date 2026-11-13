"""What a translation kept from before the split imports: :mod:`xrdroot.pycint.runtime`'s names.

A translation written by the translator under its former name begins
``from xrdroot.cint.runtime import *``; this keeps it running, with the very
objects a new translation imports from :mod:`xrdroot.pycint.runtime`.
"""

from __future__ import annotations

from ...pycint.runtime import *  # noqa: F403
from ...pycint.runtime import __all__ as __all__
