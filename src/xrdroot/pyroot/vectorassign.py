"""Compatibility imports preserve the generic vector registries and provenance."""

from ..cppstd.vectorassign import _LAYOUTS as _LAYOUTS
from ..cppstd.vectorassign import _array as _array
from ..cppstd.vectorassign import _capacity as _capacity
from ..cppstd.vectorassign import _layout as _layout
from ..cppstd.vectorassign import _storage as _storage
from ..cppstd.vectorassign import _writable as _writable
from ..cppstd.vectorassign import assigned as assigned
from ..cppstd.vectorassign import register as register
from ..cppstd.vectorassign import register_capacity as _register_capacity

#: ROOT's SmallVectorBase stores its size and capacity in signed 32-bit integers.
_RVEC_MAX = (1 << 31) - 1


def _rvec_capacity(current: int, required: int) -> int:
    return min(max(required, 2 * current + 1), _RVEC_MAX)


_register_capacity("rvec", _rvec_capacity)
