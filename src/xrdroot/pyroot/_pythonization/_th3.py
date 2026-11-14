"""``ROOT._pythonization._th3``: the 3-D histogram classes PyROOT adds its Python to."""

from __future__ import annotations

__all__: list[str] = []

#: The 3-D classes, by ROOT's names.
_th3_derived_classes_to_pythonize = ["TH3C", "TH3S", "TH3I", "TH3L", "TH3F", "TH3D", "TProfile3D"]
