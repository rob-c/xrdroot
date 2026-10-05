"""Files named by ``http://`` and ``https://`` URLs, fetched whole as ROOT's web files are read.

ROOT reads such a file through Davix, a range at a time; here the whole of
it is fetched once, into memory, and read from there - which suits the
tutorials' files, a few megabytes each.
"""

from __future__ import annotations

import importlib
import ssl
import urllib.request

__all__ = ["REMOTE", "fetch", "is_remote"]

#: What a URL names, rather than a file on this machine.
REMOTE = ("http://", "https://")

#: How long a fetch waits for the other end, in seconds.
FETCH_TIMEOUT = 120


def is_remote(name: str) -> bool:
    """Is ``name`` a URL this fetches, rather than a path?"""
    return str(name).startswith(REMOTE)


def _context() -> ssl.SSLContext:
    """What an ``https://`` fetch trusts: certifi's authorities where it is installed, since a
    Python built without the system's often has none; else the system's own."""
    try:
        certifi = importlib.import_module("certifi")
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def fetch(name: str) -> bytes:
    """The bytes of a file, or of what a URL names."""
    if is_remote(name):
        with urllib.request.urlopen(name, timeout=FETCH_TIMEOUT, context=_context()) as response:
            return bytes(response.read())
    with open(name, "rb") as stream:
        return stream.read()
