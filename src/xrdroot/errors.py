"""What reading a ROOT file can go wrong with.

They are two rather than one because the answers differ: a :class:`FormatError`
means the bytes are wrong, and a :class:`UnsupportedFeatureError` means they are
right and this reader is the one that is missing something. Both are
:class:`ROOTError`, and :class:`ROOTError` is the library's own - the transport's
:class:`~xrdclient.errors.XRootDError` is raised by the transport, at the moment
a URL is opened and read, and is not what a reader of a local file pays for.
"""

from __future__ import annotations

__all__ = ["ROOTError", "FormatError", "UnsupportedFeatureError"]


class ROOTError(Exception):
    """Base of everything :mod:`xrdroot` raises."""


class FormatError(ROOTError):
    """The bytes are not the ROOT format they claim to be."""


class UnsupportedFeatureError(ROOTError):
    """A valid ROOT file using something this reader does not do.

    Raised by name, with the class or algorithm in the message, rather than
    guessed at: a plausible misreading of physics data is worse than a
    refusal.
    """
