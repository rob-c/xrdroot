"""Interpreter imports share standard streams and explicitly supply ROOT's type aliases."""

from ...cppstd.istreams import *  # noqa: F403
from ...cppstd.istreams import KINDS as KINDS
from ...cppstd.istreams import PATTERNS as PATTERNS
from ...cppstd.istreams import __all__ as __all__

#: ROOT's numeric typedefs are supplied by this adapter, outside the standard implementation.
KINDS.update({
    **dict.fromkeys(("Int_t", "Long64_t", "UInt_t"), "integral"),
    **dict.fromkeys(("Double_t", "Float_t"), "floating"),
})
