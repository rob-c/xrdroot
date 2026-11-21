"""Where a target's URL is read: the transport, asked for when it is needed.

``xrdclient`` is a hard dependency of this package, but nothing a command that
reads a local file does needs it - and importing it is twenty-six milliseconds
of every process, a tenth of what a command costs before it does anything. A
target is handed to it when one is opened, and never before; what opens and
reads it asks for :mod:`xrdclient.io` itself, at the same moment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xrdclient.url import XRootDURL

__all__ = ["parse"]


def parse(target: Any) -> XRootDURL:
    """A path or URL as the transport reads it: its scheme, host and path."""
    from xrdclient.url import parse as read

    return read(target)
