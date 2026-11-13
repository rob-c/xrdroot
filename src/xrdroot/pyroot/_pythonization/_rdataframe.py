"""``ROOT._pythonization._rdataframe``: ``AsNumpy``'s array class and its result's clone.

Imported as ``ROOT._pythonization._rdataframe`` - under whatever name the
namespace was given - so it names what it re-exports absolutely.
"""

from __future__ import annotations

from xrdroot.pyroot.rdf.asnumpy import AsNumpyResult, ndarray
from xrdroot.pyroot.rdf.asnumpy import clone as _clone_asnumpyresult

__all__ = ["AsNumpyResult", "_clone_asnumpyresult", "ndarray"]
