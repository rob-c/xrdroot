"""``ROOT._pythonization``: the private module PyROOT's own tests reach into.

ROOT keeps the Python it adds to its classes here; its tests import helpers
from it by name. What of it this namespace has is the same helper under the
same name - nothing here is part of ``ROOT``'s public namespace.
"""

from __future__ import annotations

__all__: list[str] = []
