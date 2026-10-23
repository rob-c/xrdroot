"""ROOT numeric aliases extend the generic standard-library traits explicitly."""

import os as os
from typing import Any as Any

import numpy as np

from ..cppstd.stdlib import _KINDS as _KINDS
from ..cppstd.stdlib import __all__ as __all__
from ..cppstd.stdlib import _Limits as _Limits
from ..cppstd.stdlib import _NumericLimits as _NumericLimits
from ..cppstd.stdlib import _SmartPointer as _SmartPointer
from ..cppstd.stdlib import numeric_limits as numeric_limits
from ..cppstd.stdlib import remove as remove
from ..cppstd.stdlib import smart_pointer as smart_pointer

_KINDS.update({
    "Long64_t": np.int64,
    "ULong64_t": np.uint64,
    "Float_t": np.float32,
    "Double_t": np.float64,
    "Int_t": np.int32,
    "UInt_t": np.uint32,
})
