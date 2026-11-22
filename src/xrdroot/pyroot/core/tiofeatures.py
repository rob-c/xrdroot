"""``EIOFeatures``: the bits a basket records what its file used to write it.

``ROOT/TIOFeatures.hxx`` declares the same enum twice: ``ROOT::EIOFeatures`` is
the 6.34 spelling, which 6.40 keeps but no longer sets a bit in, and
``ROOT::Experimental::EIOFeatures`` is the one the current release writes -
``kSupported`` is every bit that may be set, and ``EIOUnsupportedFeatures``
is the ones a reader of this release must refuse. A file's ``TBasket`` says
what it used with those bits, and ``TBasket::kIOBitCount`` is how many of them
a basket has room for: one, in 6.40, where ``EIOBits::kSupported`` is that one
bit set (measured against ROOT 6.40's own ``TIOFeatures.IOBits``).
"""

from __future__ import annotations

#: Only ``EIOFeatures`` is ROOT's at the top of the namespace: the other two are
#: ``ROOT::Experimental``'s alone, measured against ROOT 6.40.
__all__ = ["EIOFeatures"]

#: How many bits a basket spends on saying how it was written: ``TBasket::kIOBitCount``.
IO_BIT_COUNT = 1


class EIOFeatures:
    """``ROOT::EIOFeatures``: the bits 6.40 declares and no longer writes."""

    kSupported = 0


class ExperimentalEIOFeatures:
    """``ROOT::Experimental::EIOFeatures``: the bits 6.40 writes, being all it has room for."""

    kSupported = (1 << IO_BIT_COUNT) - 1


class EIOUnsupportedFeatures:
    """``ROOT::Experimental::EIOUnsupportedFeatures``: the bits this release must refuse."""

    kUnsupported = 0
