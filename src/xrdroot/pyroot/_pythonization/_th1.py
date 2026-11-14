"""``ROOT._pythonization._th1``: the 1-D histogram classes PyROOT adds its Python to.

Every histogram has UHI's indexing, plotting and serialization, from
:class:`~xrdroot.pyroot._pythonization._uhi.UnifiedHistogram`; these are
the classes, by ROOT's names, that PyROOT's tests make one of each of.
"""

from __future__ import annotations

__all__: list[str] = []

#: The 1-D classes, by ROOT's names.
_th1_derived_classes_to_pythonize = ["TH1C", "TH1S", "TH1I", "TH1L", "TH1F", "TH1D", "TProfile"]
