"""``ROOT._pythonization._th2``: the 2-D histogram classes PyROOT adds its Python to.

``TH2Poly`` is left out, as ROOT leaves it out; its profile is in.
"""

from __future__ import annotations

__all__: list[str] = []

#: The 2-D classes, by ROOT's names.
_th2_derived_classes_to_pythonize = [
    "TH2C", "TH2S", "TH2I", "TH2L", "TH2F", "TH2D", "TProfile2D", "TProfile2Poly",
]  # fmt: skip
