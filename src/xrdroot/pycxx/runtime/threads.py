"""Compatibility imports retain generic standard-library objects and state."""

import time as time

from ...cppstd.threads import *  # noqa: F403
from ...cppstd.threads import __all__ as __all__
from ...cppstd.threads import _ThisThread as _ThisThread
